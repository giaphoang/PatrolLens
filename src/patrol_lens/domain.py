from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Modality = Literal["visual", "clip", "text", "ocr", "audio", "metadata"]


def _clean(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class VideoAsset:
    id: str
    path: str
    sha256: str
    duration_ms: int
    fps: float | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Segment:
    id: str
    video_id: str
    start_ms: int
    end_ms: int
    kind: str = "coarse"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    id: str
    segment_id: str
    video_id: str
    modality: Modality
    start_ms: int
    end_ms: int
    text: str | None = None
    label: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingRecord:
    id: str
    segment_id: str
    modality: Modality
    model: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryPlan:
    original_text: str
    modality_weights: dict[str, float]
    text_terms: list[str] = field(default_factory=list)
    visual_concepts: list[str] = field(default_factory=list)
    ocr_terms: list[str] = field(default_factory=list)
    audio_intent: str | None = None
    temporal_constraints: list[str] = field(default_factory=list)
    conjunctions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    segment: Segment
    score: float = 0.0
    modality_scores: dict[str, float] = field(default_factory=dict)
    evidence: list[Observation] = field(default_factory=list)
    rerank_score: float | None = None
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment.to_dict(),
            "score": self.score,
            "modality_scores": self.modality_scores,
            "evidence": [_clean(item) for item in self.evidence],
            "rerank_score": self.rerank_score,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class RerankDecision:
    match: Literal["yes", "no", "uncertain"]
    event_start_offset_ms: int = 0
    event_end_offset_ms: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    warning: str | None = None


def result_dict(query: str, plan: QueryPlan, candidates: list[Candidate], *, index_version: str, rerank_status: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        segment = candidate.segment
        results.append(
            {
                "video_id": segment.video_id,
                "segment_id": segment.id,
                "start_s": round(segment.start_ms / 1000, 3),
                "end_s": round(segment.end_ms / 1000, 3),
                "score": round(candidate.score, 6),
                "rerank_score": None if candidate.rerank_score is None else round(candidate.rerank_score, 6),
                "confidence": None if candidate.confidence is None else round(candidate.confidence, 6),
                "modality_scores": candidate.modality_scores,
                "evidence": [_clean(item) for item in candidate.evidence],
                "warnings": candidate.warnings,
            }
        )
    return {
        "query": query,
        "query_plan": plan.to_dict(),
        "index_version": index_version,
        "rerank_status": rerank_status,
        "results": results,
    }
