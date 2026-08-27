from __future__ import annotations

from dataclasses import dataclass

import patrol_lens.agent.gemini_agent as agent_module
from patrol_lens.agent import ActivePerceptionAgent
from patrol_lens.config import AgentConfig
from patrol_lens.domain import (
    AgentAction,
    AgentDecision,
    CandidateInterval,
    QueryPlan,
    ToolObservation,
    VideoAsset,
)


@dataclass
class SequencePolicy:
    decisions: list[AgentDecision]

    def decide(self, _query, _plan, _candidate, _memory, _media_paths):
        return self.decisions.pop(0)


class FakeExecutor:
    def __init__(self, _asset, _candidate, run_dir, *, config):
        self.run_dir = run_dir
        self.turn = 0

    def execute(self, action):
        self.turn += 1
        return ToolObservation(
            f"obs-{self.turn}",
            action,
            action.start_ms or 0,
            action.end_ms or 0,
            [f"/tmp/{action.type}-{self.turn}.bin"],
        )


def test_active_loop_seeks_cross_modal_evidence_and_persists_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, "MediaToolExecutor", FakeExecutor)
    candidate = CandidateInterval("c1", "v1", 10_000, 30_000)
    plan = QueryPlan(
        "red jacket started shouting",
        visual_queries=["red jacket"],
        audio_queries=["shouting"],
        required_modalities=["visual", "audio_event"],
        relation="overlap",
        target="onset",
    )
    policy = SequencePolicy(
        [
            AgentDecision("Need appearance", AgentAction("get_frames", 12_000, 20_000, num_frames=4)),
            AgentDecision("Red jacket visible; need voice", AgentAction("get_audio", 14_000, 22_000)),
            AgentDecision("Need speaker association", AgentAction("get_clip", 15_000, 22_000)),
            AgentDecision(
                "Same person begins shouting",
                AgentAction(
                    "answer", 16_500, 21_000, answer="red-jacket person shouts",
                    status="supported", confidence=0.88,
                ),
            ),
        ]
    )
    config = AgentConfig(max_turns=5, run_root=str(tmp_path / "runs"))
    result = ActivePerceptionAgent(policy, config=config).inspect(
        plan.original_text,
        plan,
        candidate,
        VideoAsset("v1", "/tmp/video.mp4", "hash", 60_000),
    )

    assert result.conclusion.status == "supported"
    assert result.conclusion.start_ms == 16_500
    assert result.memory.direct_modalities() == {"visual", "audio", "audiovisual"}
    assert result.memory.path.exists()
    assert "Red jacket visible" in result.memory.path.read_text()


def test_controller_rejects_premature_cross_modal_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, "MediaToolExecutor", FakeExecutor)
    candidate = CandidateInterval("c1", "v1", 0, 20_000)
    plan = QueryPlan(
        "person shouted",
        visual_queries=["person"],
        audio_queries=["shouting"],
        required_modalities=["visual", "audio_event"],
    )
    policy = SequencePolicy(
        [
            AgentDecision("Assume yes", AgentAction("answer", status="supported", confidence=0.9)),
            AgentDecision("Inspect directly", AgentAction("get_clip", 1_000, 8_000)),
            AgentDecision("Now supported", AgentAction("answer", 2_000, 7_000, status="supported", confidence=0.8)),
        ]
    )
    result = ActivePerceptionAgent(
        policy,
        config=AgentConfig(max_turns=3, run_root=str(tmp_path / "runs")),
    ).inspect("person shouted", plan, candidate, VideoAsset("v1", "/tmp/v.mp4", "h", 20_000))

    assert result.turns == 3
    assert result.conclusion.status == "supported"
    assert any("premature" in note for note in result.memory.notes)
