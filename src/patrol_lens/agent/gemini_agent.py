from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import AgentConfig
from ..domain import (
    AgentAction,
    AgentConclusion,
    AgentDecision,
    CandidateInterval,
    QueryPlan,
    VideoAsset,
)
from ..media_tools import ActionValidationError, MediaToolExecutor
from .memory import EvidenceMemory


class JSONGenerator(Protocol):
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        media_paths: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]: ...


class ActivePolicy(Protocol):
    def decide(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        memory: EvidenceMemory,
        media_paths: list[str],
    ) -> AgentDecision: ...


ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        "action": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["get_frames", "get_audio", "get_clip", "answer"]},
                "start_ms": {"type": "integer"},
                "end_ms": {"type": "integer"},
                "fps": {"type": "number"},
                "num_frames": {"type": "integer"},
                "answer": {"type": "string"},
                "status": {"type": "string", "enum": ["supported", "rejected", "uncertain"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["type"],
            "additionalProperties": False,
        },
    },
    "required": ["assessment", "action"],
    "additionalProperties": False,
}


class GeminiActivePolicy:
    def __init__(self, client: JSONGenerator, *, model: str) -> None:
        self.client = client
        self.model = model

    def decide(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        memory: EvidenceMemory,
        media_paths: list[str],
    ) -> AgentDecision:
        prompt = f"""You are the active-perception controller for evidentiary body-camera search.
Your task is to establish or reject the complete investigator event, not merely a nearby
object or word. Inspect the compact memory and any newly attached media. Return one next
action. Use get_frames for appearance/OCR detail, get_audio for speech/prosody, and get_clip
for motion, causality, speaker association, or temporal order. Request only times inside
[{candidate.start_ms}, {candidate.end_ms}] milliseconds. Avoid repeating an action already
in memory. Use answer only when evidence is sufficient or the candidate is contradicted.
For answer, include status, confidence, start_ms, end_ms, and a concise answer string.
The assessment must be a short evidence summary, not hidden chain-of-thought.

Query: {query}
Memory: {memory.compact_context()}"""
        data = self.client.generate_json(
            prompt,
            ACTION_SCHEMA,
            media_paths=media_paths,
            model=self.model,
        )
        raw = dict(data.get("action", {}))
        action = AgentAction(
            type=raw.get("type", "answer"),
            start_ms=int(raw["start_ms"]) if raw.get("start_ms") is not None else None,
            end_ms=int(raw["end_ms"]) if raw.get("end_ms") is not None else None,
            fps=float(raw["fps"]) if raw.get("fps") is not None else None,
            num_frames=int(raw["num_frames"]) if raw.get("num_frames") is not None else None,
            answer=str(raw["answer"]) if raw.get("answer") is not None else None,
            status=raw.get("status"),
            confidence=float(raw["confidence"]) if raw.get("confidence") is not None else None,
        )
        return AgentDecision(str(data.get("assessment", "")), action)


@dataclass(frozen=True)
class AgentRunResult:
    conclusion: AgentConclusion
    memory: EvidenceMemory
    turns: int


class ActivePerceptionAgent:
    def __init__(self, policy: ActivePolicy, *, config: AgentConfig | None = None) -> None:
        self.policy = policy
        self.config = config or AgentConfig()

    @staticmethod
    def _required_direct(plan: QueryPlan) -> set[str]:
        required: set[str] = set()
        if "visual" in plan.required_modalities:
            required.add("visual")
        if "audio_event" in plan.required_modalities:
            required.add("audio")
        if {"visual", "audio_event"}.issubset(set(plan.required_modalities)):
            required.add("audiovisual")
        return required

    def inspect(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        *,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> AgentRunResult:
        memory = EvidenceMemory(query, plan, candidate, self.config.run_root)
        executor = MediaToolExecutor(asset, candidate, memory.run_dir, config=self.config)
        media_paths: list[str] = []
        required_direct = self._required_direct(plan)

        def cancelled() -> bool:
            return bool(
                (cancel_event is not None and cancel_event.is_set())
                or (deadline is not None and time.monotonic() >= deadline)
            )

        def cancelled_result(turns: int) -> AgentRunResult:
            return AgentRunResult(
                AgentConclusion(
                    status="uncertain",
                    description="Active perception cancelled by the search latency limit",
                    start_ms=candidate.start_ms,
                    end_ms=candidate.end_ms,
                    confidence=0.0,
                    missing_evidence=sorted(required_direct - memory.direct_modalities()),
                ),
                memory,
                turns,
            )

        for turn in range(1, self.config.max_turns + 1):
            if cancelled():
                return cancelled_result(turn - 1)
            decision = self.policy.decide(query, plan, candidate, memory, media_paths)
            if cancelled():
                return cancelled_result(turn)
            memory.record_decision(decision)
            action = decision.action
            if action.type == "answer":
                status = action.status or "uncertain"
                missing_direct = required_direct - memory.direct_modalities()
                if status == "supported" and missing_direct:
                    memory.add_note(
                        "Controller rejected premature supported answer; direct inspection still needed for "
                        + ", ".join(sorted(missing_direct))
                    )
                    media_paths = []
                    continue
                start = candidate.start_ms if action.start_ms is None else action.start_ms
                end = candidate.end_ms if action.end_ms is None else action.end_ms
                start = max(candidate.start_ms, min(candidate.end_ms, start))
                end = max(start, min(candidate.end_ms, end))
                return AgentRunResult(
                    AgentConclusion(
                        status=status,
                        description=action.answer or decision.assessment,
                        start_ms=start,
                        end_ms=end,
                        confidence=max(0.0, min(1.0, action.confidence or 0.0)),
                        missing_evidence=sorted(missing_direct),
                    ),
                    memory,
                    turn,
                )
            if memory.has_action(action.signature()):
                memory.add_note(f"Repeated action rejected: {action.signature()}")
                media_paths = []
                continue
            try:
                observation = executor.execute(action)
            except ActionValidationError as exc:
                memory.add_note(f"Invalid action rejected: {exc}")
                media_paths = []
                continue
            memory.add_observation(observation)
            if cancelled():
                return cancelled_result(turn)
            media_paths = observation.media_paths

        return AgentRunResult(
            AgentConclusion(
                status="uncertain",
                description="Active-perception turn budget exhausted without sufficient evidence",
                start_ms=candidate.start_ms,
                end_ms=candidate.end_ms,
                confidence=0.0,
                missing_evidence=sorted(required_direct - memory.direct_modalities()),
            ),
            memory,
            self.config.max_turns,
        )
