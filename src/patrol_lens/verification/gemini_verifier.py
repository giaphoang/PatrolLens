from __future__ import annotations

import json
from typing import Any, Protocol

from ..domain import CandidateInterval, QueryPlan, VerificationResult, VideoAsset
from .direct_media import DirectVerificationContext


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


class GeminiEventVerifier:
    """One-shot multimodal gate: retrieval scores never become confidence."""

    def __init__(self, client: JSONGenerator, *, model: str) -> None:
        self.client = client
        self.model = model

    def verify(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        context: DirectVerificationContext,
    ) -> VerificationResult:
        evidence = [
            {
                "id": item.id,
                "time_ms": [item.start_ms, item.end_ms],
                "modality": item.modality,
                "content": item.content[:1_200],
                "confidence": round(item.confidence, 3),
                "source": item.source,
            }
            for item in candidate.evidence[:60]
        ]
        candidate_summary = {
            "video_id": asset.id,
            "start_ms": candidate.start_ms,
            "end_ms": candidate.end_ms,
            "retrieval_score": candidate.score,
        }
        prompt = f"""You are the independent event verifier for a body-camera search result.
Decide whether the complete investigator query is directly supported within this candidate.
Distinguish co-occurrence from attribution and causality: for example, a red jacket plus an
unassociated loud voice does not prove that the red-jacket person shouted. Treat retrieved
ASR/OCR/audio labels as fallible auxiliary evidence. Inspect the attached candidate clip directly
for appearance, speech, acoustic events, action, temporal order, and speaker attribution. Prefer
rejected or uncertain over unsupported inference. Return absolute millisecond boundaries inside
[{context.start_ms}, {context.end_ms}]. Include exact transcript excerpts relevant to the event.

Query: {query}
Structured plan: {plan.to_dict()}
Candidate: {json.dumps(candidate_summary, separators=(",", ":"))}
Retrieved timestamped evidence: {json.dumps(evidence, separators=(",", ":"))}"""
        data = self.client.generate_json(
            prompt,
            VERIFICATION_SCHEMA,
            media_paths=list(context.media_paths),
            model=self.model,
        )
        warnings: list[str] = []
        start = int(data.get("start_ms", candidate.start_ms))
        end = int(data.get("end_ms", candidate.end_ms))
        if end < start:
            start, end = end, start
            warnings.append("verifier_interval_reordered")
        clamped_start = max(context.start_ms, min(context.end_ms, start))
        clamped_end = max(clamped_start, min(context.end_ms, end))
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
