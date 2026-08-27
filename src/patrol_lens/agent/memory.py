from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from ..domain import AgentDecision, CandidateInterval, QueryPlan, ToolObservation


class EvidenceMemory:
    """Compact, durable state for one candidate's perception loop."""

    def __init__(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        run_root: str | Path,
        *,
        run_id: str | None = None,
    ) -> None:
        self.query = query
        self.plan = plan
        self.candidate = candidate
        self.run_id = run_id or uuid.uuid4().hex
        self.run_dir = Path(run_root).expanduser().resolve() / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "memory.json"
        self.observations: list[ToolObservation] = []
        self.decisions: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.action_signatures: set[str] = set()
        self.persist()

    def add_observation(self, observation: ToolObservation) -> None:
        self.observations.append(observation)
        self.action_signatures.add(observation.action.signature())
        self.persist()

    def record_decision(self, decision: AgentDecision) -> None:
        if self.observations and not self.observations[-1].summary:
            self.observations[-1].summary = decision.assessment
        self.decisions.append(
            {"assessment": decision.assessment, "action": decision.action.to_dict()}
        )
        self.persist()

    def add_note(self, note: str) -> None:
        self.notes.append(note)
        self.persist()

    def has_action(self, signature: str) -> bool:
        return signature in self.action_signatures

    def direct_modalities(self) -> set[str]:
        found: set[str] = set()
        for item in self.observations:
            if item.action.type in {"get_frames", "get_clip"}:
                found.add("visual")
            if item.action.type in {"get_audio", "get_clip"}:
                found.add("audio")
            if item.action.type == "get_clip":
                found.add("audiovisual")
        return found

    def media_paths(self, *, limit: int = 8) -> list[str]:
        paths = [path for item in self.observations for path in item.media_paths]
        return paths[-limit:]

    def compact_context(self) -> str:
        seed = [
            {
                "id": item.id,
                "time_ms": [item.start_ms, item.end_ms],
                "modality": item.modality,
                "content": item.content[:600],
                "confidence": round(item.confidence, 3),
                "source": item.source,
            }
            for item in self.candidate.evidence[:40]
        ]
        observations = [
            {
                "id": item.id,
                "action": item.action.to_dict(),
                "time_ms": [item.start_ms, item.end_ms],
                "summary": item.summary[:800],
                "media_count": len(item.media_paths),
            }
            for item in self.observations
        ]
        return json.dumps(
            {
                "candidate": {
                    "video_id": self.candidate.video_id,
                    "start_ms": self.candidate.start_ms,
                    "end_ms": self.candidate.end_ms,
                    "retrieval_score": self.candidate.score,
                },
                "query_plan": self.plan.to_dict(),
                "retrieved_evidence": seed,
                "active_observations": observations,
                "controller_notes": self.notes,
            },
            separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "query_plan": self.plan.to_dict(),
            "candidate": self.candidate.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "decisions": list(self.decisions),
            "notes": list(self.notes),
        }

    def persist(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2))
        temporary.replace(self.path)
