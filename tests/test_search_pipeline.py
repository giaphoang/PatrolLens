from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from patrol_lens.config import SearchConfig
from patrol_lens.domain import CandidateInterval, QueryPlan, VerificationResult, VideoAsset
from patrol_lens.pipeline import SearchPipeline


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


class FakeMemory:
    def __init__(self, direct):
        self.direct = set(direct)

    def direct_modalities(self):
        return set(self.direct)


class TrackingAgent:
    def __init__(self, delays=None, direct=None):
        self.delays = delays or {}
        self.direct = {"visual"} if direct is None else set(direct)
        self.started = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def inspect(self, _query, _plan, candidate, _asset, *, cancel_event=None, deadline=None):
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
            return SimpleNamespace(memory=FakeMemory(self.direct))
        finally:
            with self.lock:
                self.active -= 1


class LegacyAgent:
    """Old inspect signature proves omitted controls retain the synchronous path."""

    def __init__(self):
        self.started = []

    def inspect(self, _query, _plan, candidate, _asset):
        self.started.append(candidate.id)
        return SimpleNamespace(memory=FakeMemory({"visual"}))


class FakeVerifier:
    def __init__(self, confidences, *, failures=None):
        self.confidences = confidences
        self.failures = set(failures or [])

    def verify(self, _query, _plan, candidate, _asset, _run):
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
    def refine(self, _query, _plan, _candidate, _asset, verification, _memory):
        return verification


def candidates(count=4):
    return [
        CandidateInterval(f"c{index}", f"v{index}", index * 1_000, index * 1_000 + 800, score=1 - index / 10)
        for index in range(count)
    ]


def pipeline(items, agent, verifier, config, *, retrieval_delay=0.0):
    plan = QueryPlan(
        "white shirt woman starts crying",
        visual_queries=["white shirt woman crying"],
        required_modalities=["visual"],
        target="onset",
    )
    return SearchPipeline(
        FakeStore(items),
        FakeRetriever(plan, items, delay=retrieval_delay),
        agent,
        verifier,
        FakeRefiner(),
        config=config,
    )


def test_default_search_keeps_legacy_sequential_agent_contract():
    items = candidates(3)
    agent = LegacyAgent()
    response = pipeline(
        items,
        agent,
        FakeVerifier({}),
        SearchConfig(),
    ).search("query", max_candidates=3)

    assert agent.started == ["c0", "c1", "c2"]
    assert response.candidates_examined == 3
    assert len(response.results) == 3


def test_candidate_parallelism_is_bounded():
    items = candidates(4)
    agent = TrackingAgent()
    response = pipeline(
        items,
        agent,
        FakeVerifier({}),
        SearchConfig(candidate_parallelism=2),
    ).search("query", max_candidates=4)

    assert agent.max_active == 2
    assert response.candidates_examined == 4
    assert len(response.results) == 4


def test_high_confidence_direct_result_stops_later_candidates():
    items = candidates(6)
    agent = TrackingAgent(delays={"c0": 0.005, "c1": 0.3})
    response = pipeline(
        items,
        agent,
        FakeVerifier({"c0": 0.95}),
        SearchConfig(candidate_parallelism=2, early_stop_confidence=0.9),
    ).search("query", max_candidates=6)

    assert response.results[0].description == "supported c0"
    assert response.candidates_examined <= 2
    assert any(item.startswith("early_stop:c0") for item in response.warnings)


def test_early_stop_requires_direct_modalities():
    items = candidates(3)
    agent = TrackingAgent(direct=set())
    response = pipeline(
        items,
        agent,
        FakeVerifier({"c0": 0.99}),
        SearchConfig(candidate_parallelism=2, early_stop_confidence=0.9),
    ).search("query", max_candidates=3)

    assert response.candidates_examined == 3
    assert not any(item.startswith("early_stop:") for item in response.warnings)


def test_global_timeout_returns_best_supported_result_so_far():
    items = candidates(4)
    agent = TrackingAgent(delays={"c0": 0.005, "c1": 1.0, "c2": 1.0, "c3": 1.0})
    started = time.monotonic()
    response = pipeline(
        items,
        agent,
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
        TrackingAgent(),
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
        TrackingAgent(),
        FakeVerifier({}, failures={"c1"}),
        SearchConfig(candidate_parallelism=2),
    ).search("query", max_candidates=3)

    assert len(response.results) == 2
    assert any(item.startswith("candidate_failed:c1") for item in response.warnings)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"candidate_parallelism": 0}, "parallelism"),
        ({"early_stop_confidence": 1.1}, "confidence"),
        ({"timeout_s": 0}, "timeout"),
    ],
)
def test_search_latency_controls_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SearchConfig(**kwargs)
