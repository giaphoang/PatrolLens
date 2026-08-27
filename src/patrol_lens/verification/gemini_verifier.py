from __future__ import annotations

from typing import Any, Protocol

from ..agent.gemini_agent import AgentRunResult
from ..domain import CandidateInterval, QueryPlan, VerificationResult, VideoAsset


class JSONGenerator(Protocol):
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        media_paths: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]: ...


VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["supported", "rejected", "uncertain"]},
        "event_description": {"type": "string"},
        "start_ms": {"type": "integer"},
        "end_ms": {"type": "integer"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {
            "type": "object",
            "properties": {
                "visual": {"type": "array", "items": {"type": "string"}},
                "audio": {"type": "array", "items": {"type": "string"}},
                "transcript": {"type": "array", "items": {"type": "string"}},
                "ocr": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["visual", "audio", "transcript", "ocr"],
            "additionalProperties": False,
        },
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "status", "event_description", "start_ms", "end_ms", "confidence", "evidence", "missing_evidence"
    ],
    "additionalProperties": False,
}


def _selected_media(run: AgentRunResult) -> list[str]:
    selected: list[str] = []
    seen_kinds: set[str] = set()
    frame_budget = 4
    for observation in reversed(run.memory.observations):
        kind = observation.action.type
        if kind == "get_frames":
            if frame_budget:
                selected[0:0] = observation.media_paths[-frame_budget:]
                frame_budget -= min(frame_budget, len(observation.media_paths))
        elif kind not in seen_kinds:
            selected[0:0] = observation.media_paths
            seen_kinds.add(kind)
    return selected[-8:]


class GeminiEventVerifier:
    """Final semantic gate: retrieval scores never become evidence confidence."""

    def __init__(self, client: JSONGenerator, *, model: str) -> None:
        self.client = client
        self.model = model

    def verify(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        run: AgentRunResult,
    ) -> VerificationResult:
        if run.conclusion.status == "rejected":
            return VerificationResult(
                status="rejected",
                event_description=run.conclusion.description,
                start_ms=run.conclusion.start_ms,
                end_ms=run.conclusion.end_ms,
                confidence=run.conclusion.confidence,
                missing_evidence=run.conclusion.missing_evidence,
            )
        prompt = f"""You are the independent event verifier for a body-camera search result.
Decide whether the complete investigator query is directly supported within this candidate.
Distinguish co-occurrence from attribution and causality: for example, a red jacket plus an
unassociated loud voice does not prove that the red-jacket person shouted. Treat retrieved
ASR/OCR/audio labels as fallible auxiliary evidence. Use attached direct media and the active
observation summaries to resolve contradictions. Prefer rejected or uncertain over unsupported
inference. Return absolute millisecond boundaries inside [{candidate.start_ms}, {candidate.end_ms}].

Query: {query}
Structured plan: {plan.to_dict()}
Agent proposal: {run.conclusion}
Evidence memory: {run.memory.compact_context()}"""
        data = self.client.generate_json(
            prompt,
            VERIFICATION_SCHEMA,
            media_paths=_selected_media(run),
            model=self.model,
        )
        warnings: list[str] = []
        start = int(data.get("start_ms", candidate.start_ms))
        end = int(data.get("end_ms", candidate.end_ms))
        if end < start:
            start, end = end, start
            warnings.append("verifier_interval_reordered")
        clamped_start = max(candidate.start_ms, min(candidate.end_ms, start))
        clamped_end = max(clamped_start, min(candidate.end_ms, end))
        if (clamped_start, clamped_end) != (start, end):
            warnings.append("verifier_interval_clamped")
        raw_evidence = dict(data.get("evidence", {}))
        evidence = {
            key: [str(item) for item in raw_evidence.get(key, [])]
            for key in ("visual", "audio", "transcript", "ocr")
        }
        return VerificationResult(
            status=data.get("status", "uncertain"),
            event_description=str(data.get("event_description", "")),
            start_ms=clamped_start,
            end_ms=clamped_end,
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            evidence=evidence,
            missing_evidence=[str(item) for item in data.get("missing_evidence", [])],
            warnings=warnings,
        )
