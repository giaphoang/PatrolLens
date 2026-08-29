from __future__ import annotations

from patrol_lens.domain import CandidateInterval, QueryPlan, VideoAsset
from patrol_lens.verification import DirectVerificationContext, GeminiEventVerifier


class FakeClient:
    def generate_json(self, _prompt, _schema, *, media_paths=None, model=None):
        assert model == "gemini-test"
        assert media_paths == ["/tmp/candidate.mp4"]
        return {
            "status": "supported",
            "event_description": "person is handcuffed",
            "start_ms": 5_000,
            "end_ms": 40_000,
            "confidence": 0.84,
            "evidence": {
                "visual": ["cuffs close around wrists"],
                "audio": [],
                "transcript": [],
                "ocr": [],
            },
            "missing_evidence": [],
        }


def test_verifier_is_independent_and_clamps_interval(tmp_path):
    candidate = CandidateInterval("c", "v", 10_000, 30_000)
    plan = QueryPlan("handcuffed", visual_queries=["handcuffing"], required_modalities=["visual"])
    context = DirectVerificationContext(
        workspace=tmp_path,
        media_paths=("/tmp/candidate.mp4",),
        start_ms=candidate.start_ms,
        end_ms=candidate.end_ms,
        direct_modalities=frozenset({"visual"}),
    )

    result = GeminiEventVerifier(FakeClient(), model="gemini-test").verify(
        plan.original_text,
        plan,
        candidate,
        VideoAsset("v", "/tmp/video.mp4", "hash", 60_000),
        context,
    )

    assert (result.start_ms, result.end_ms) == (10_000, 30_000)
    assert result.confidence == 0.84
    assert "verifier_interval_clamped" in result.warnings
