from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from patrol_lens.config import SearchConfig
from patrol_lens.domain import CandidateInterval, QueryPlan, VerificationResult, VideoAsset
from patrol_lens.history import TrajectoryRecorder, show_history
from patrol_lens.pipeline import SearchPipeline
from patrol_lens.verification import DirectVerificationContext


class FakeStore:
    def __init__(self, candidates):
        self.assets = {
            candidate.video_id: VideoAsset(
                candidate.video_id,
                f"/videos/{candidate.video_id}.mp4",
                f"hash-{candidate.video_id}",
                60_000,
            )
            for candidate in candidates
        }

    def get_asset(self, video_id):
        return self.assets.get(video_id)

    def get_metadata(self, key, default=None):
        return "3.0.0" if key == "index_version" else default


class FakeRetriever:
    def __init__(self, plan, candidates, *, delay=0.0):
        self.plan = plan
        self.candidates = candidates
        self.delay = delay

    def retrieve(self, _query):
        if self.delay:
            time.sleep(self.delay)
        return self.plan, self.candidates


class TrackingMediaProvider:
    def __init__(self, delays=None, direct=None):
        self.delays = delays or {}
        self.direct = {"visual"} if direct is None else set(direct)
        self.started = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def prepare(self, _query, _plan, candidate, _asset, *, cancel_event=None, deadline=None):
        with self.lock:
            self.started.append(candidate.id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            finish = time.monotonic() + self.delays.get(candidate.id, 0.02)
            while time.monotonic() < finish:
                if cancel_event is not None and cancel_event.is_set():
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(0.002)
            return DirectVerificationContext(
                workspace=Path("/tmp/direct-verification-test"),
                media_paths=(f"/tmp/{candidate.id}.mp4",),
                start_ms=candidate.start_ms,
                end_ms=candidate.end_ms,
                direct_modalities=frozenset(self.direct),
            )
        finally:
            with self.lock:
                self.active -= 1


class SequentialMediaProvider:
    """Direct media provider for the default sequential path."""

    def __init__(self):
        self.started = []

    def prepare(
        self,
        _query,
        _plan,
        candidate,
        _asset,
        *,
        cancel_event=None,
        deadline=None,
    ):
        self.started.append(candidate.id)
        return DirectVerificationContext(
            workspace=Path("/tmp/direct-verification-test"),
            media_paths=(f"/tmp/{candidate.id}.mp4",),
            start_ms=candidate.start_ms,
            end_ms=candidate.end_ms,
            direct_modalities=frozenset({"visual"}),
        )


class FakeVerifier:
    def __init__(self, confidences, *, failures=None):
        self.confidences = confidences
        self.failures = set(failures or [])

    def verify(self, _query, _plan, candidate, _asset, _context):
        if candidate.id in self.failures:
            raise RuntimeError("provider failed")
        confidence = self.confidences.get(candidate.id, 0.8)
        return VerificationResult(
            status="supported",
            event_description=f"supported {candidate.id}",
            start_ms=candidate.start_ms,
            end_ms=candidate.end_ms,
            confidence=confidence,
            evidence={"visual": [candidate.id]},
        )


class FakeRefiner:
    def refine(self, _query, _plan, _candidate, _asset, verification, _workspace):
        return verification


def candidates(count=4):
    return [
        CandidateInterval(f"c{index}", f"v{index}", index * 1_000, index * 1_000 + 800, score=1 - index / 10)
        for index in range(count)
    ]


def pipeline(items, media_provider, verifier, config, *, retrieval_delay=0.0, recorder=None):
    plan = QueryPlan(
        "white shirt woman starts crying",
        visual_queries=["white shirt woman crying"],
        required_modalities=["visual"],
        target="onset",
    )
    return SearchPipeline(
        FakeStore(items),
        FakeRetriever(plan, items, delay=retrieval_delay),
        media_provider,
        verifier,
        FakeRefiner(),
        config=config,
        recorder=recorder,
    )


def test_default_search_verifies_candidates_sequentially():
    items = candidates(3)
    media_provider = SequentialMediaProvider()
    response = pipeline(
        items,
        media_provider,
        FakeVerifier({}),
        SearchConfig(),
    ).search("query", max_candidates=3)

    assert media_provider.started == ["c0", "c1", "c2"]
    assert response.candidates_examined == 3
    assert len(response.results) == 3


def test_candidate_parallelism_is_bounded():
    items = candidates(4)
    media_provider = TrackingMediaProvider()
    response = pipeline(
        items,
        media_provider,
        FakeVerifier({}),
        SearchConfig(candidate_parallelism=2),
    ).search("query", max_candidates=4)

    assert media_provider.max_active == 2
    assert response.candidates_examined == 4
    assert len(response.results) == 4


def test_lightweight_reranking_prefers_required_modality_coverage():
    transcript_only = CandidateInterval(
        "transcript-only",
        "v0",
        0,
        800,
        score=0.99,
        covered_modalities=["transcript"],
    )
    visual = CandidateInterval(
        "visual",
        "v1",
        1_000,
        1_800,
        score=0.50,
        covered_modalities=["visual"],
    )
    media_provider = SequentialMediaProvider()

    response = pipeline(
        [transcript_only, visual],
        media_provider,
        FakeVerifier({}),
        SearchConfig(),
    ).search("query", max_candidates=1)

    assert media_provider.started == ["visual"]
    assert response.results[0].description == "supported visual"


def test_high_confidence_direct_result_stops_later_candidates():
    items = candidates(6)
    media_provider = TrackingMediaProvider(delays={"c0": 0.005, "c1": 0.3})
    response = pipeline(
        items,
        media_provider,
        FakeVerifier({"c0": 0.95}),
        SearchConfig(candidate_parallelism=2, early_stop_confidence=0.9),
    ).search("query", max_candidates=6)

    assert response.results[0].description == "supported c0"
    assert response.candidates_examined <= 2
    assert any(item.startswith("early_stop:c0") for item in response.warnings)


def test_early_stop_requires_direct_modalities():
    items = candidates(3)
    media_provider = TrackingMediaProvider(direct=set())
    response = pipeline(
        items,
        media_provider,
        FakeVerifier({"c0": 0.99}),
        SearchConfig(candidate_parallelism=2, early_stop_confidence=0.9),
    ).search("query", max_candidates=3)

    assert response.candidates_examined == 3
    assert not any(item.startswith("early_stop:") for item in response.warnings)


def test_global_timeout_returns_best_supported_result_so_far():
    items = candidates(4)
    media_provider = TrackingMediaProvider(
        delays={"c0": 0.005, "c1": 1.0, "c2": 1.0, "c3": 1.0}
    )
    started = time.monotonic()
    response = pipeline(
        items,
        media_provider,
        FakeVerifier({"c0": 0.82}),
        SearchConfig(candidate_parallelism=2, timeout_s=0.08),
    ).search("query", max_candidates=4)

    assert time.monotonic() - started < 0.4
    assert response.results[0].description == "supported c0"
    assert "search_timeout_returning_best_supported" in response.warnings


def test_global_timeout_covers_retrieval():
    items = candidates(1)
    response = pipeline(
        items,
        TrackingMediaProvider(),
        FakeVerifier({}),
        SearchConfig(candidate_parallelism=2, timeout_s=0.03),
        retrieval_delay=0.2,
    ).search("query", max_candidates=1)

    assert response.results == []
    assert "search_timeout:retrieval" in response.warnings


def test_parallel_candidates_keep_provider_errors_isolated():
    items = candidates(3)
    response = pipeline(
        items,
        TrackingMediaProvider(),
        FakeVerifier({}, failures={"c1"}),
        SearchConfig(candidate_parallelism=2),
    ).search("query", max_candidates=3)

    assert len(response.results) == 2
    assert any(item.startswith("candidate_failed:c1") for item in response.warnings)


def test_preflight_budget_denies_candidate_inference_and_persists_attempt(tmp_path):
    items = candidates(2)
    media_provider = TrackingMediaProvider()
    recorder = TrajectoryRecorder(
        tmp_path,
        query="query",
        command="search",
        max_cost_usd=0.01,
        estimated_model_call_cost_usd=0.02,
    )
    response = pipeline(
        items,
        media_provider,
        FakeVerifier({}),
        SearchConfig(candidate_parallelism=2, max_run_cost_usd=0.01),
        recorder=recorder,
    ).search("query", max_candidates=2)

    assert media_provider.started == []
    assert response.results == []
    assert "run_cost_denied:estimated_upper_bound_exceeds_limit" in response.warnings
    trajectory = show_history(tmp_path, recorder.run_id)["trajectory"]
    assert any(item["event_type"] == "budget_estimated" for item in trajectory)
    assert any(item["event_type"] == "budget_exceeded" for item in trajectory)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"candidate_parallelism": 0}, "parallelism"),
        ({"early_stop_confidence": 1.1}, "confidence"),
        ({"timeout_s": 0}, "timeout"),
        ({"max_run_cost_usd": 0}, "cost"),
    ],
)
def test_search_latency_controls_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SearchConfig(**kwargs)
