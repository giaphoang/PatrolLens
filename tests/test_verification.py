from __future__ import annotations

from patrol_lens.agent.gemini_agent import AgentRunResult
from patrol_lens.agent.memory import EvidenceMemory
from patrol_lens.domain import AgentConclusion, CandidateInterval, QueryPlan, VideoAsset
from patrol_lens.verification import GeminiEventVerifier


class FakeClient:
    def generate_json(self, _prompt, _schema, *, media_paths=None, model=None):
        assert model == "gemini-test"
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
    memory = EvidenceMemory(plan.original_text, plan, candidate, tmp_path)
    run = AgentRunResult(
        AgentConclusion("supported", "proposal", 12_000, 20_000, 0.9),
        memory,
        2,
    )

    result = GeminiEventVerifier(FakeClient(), model="gemini-test").verify(
        plan.original_text,
        plan,
        candidate,
        VideoAsset("v", "/tmp/video.mp4", "hash", 60_000),
        run,
    )

    assert (result.start_ms, result.end_ms) == (10_000, 30_000)
    assert result.confidence == 0.84
    assert "verifier_interval_clamped" in result.warnings
