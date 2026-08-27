from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Modality = Literal[
    "visual",
    "transcript",
    "ocr",
    "audio",
    "audio_event",
    "metadata",
]
Relation = Literal["overlap", "before", "after", "sequence", "any"]
SupportStatus = Literal["supported", "rejected", "uncertain"]
ActionType = Literal["get_frames", "get_audio", "get_clip", "answer"]


def jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def hash_evidence(evidence: Evidence) -> str:
    """Stable hash of the raw, timestamped evidence record.

    This deliberately excludes derived embeddings. It lets a database row
    prove which evidence payload and timestamp produced an embedding.
    """

    payload = {
        "id": evidence.id,
        "video_id": evidence.video_id,
        "segment_id": evidence.segment_id,
        "start_ms": evidence.start_ms,
        "end_ms": evidence.end_ms,
        "modality": evidence.modality,
        "content": evidence.content,
        "confidence": evidence.confidence,
        "source": evidence.source,
        "metadata": evidence.metadata,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VideoAsset:
    id: str
    path: str
    sha256: str
    duration_ms: int
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Segment:
    """A deterministic processing window, not a final search result."""

    id: str
    video_id: str
    start_ms: int
    end_ms: int
    kind: str = "coarse"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Evidence:
    """Canonical timestamped observation shared by every modality."""

    id: str
    video_id: str
    start_ms: int
    end_ms: int
    modality: Modality
    content: str
    confidence: float
    source: str
    segment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingRecord:
    id: str
    evidence_id: str
    modality: Modality
    model: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryPlan:
    original_text: str
    visual_queries: list[str] = field(default_factory=list)
    transcript_queries: list[str] = field(default_factory=list)
    ocr_queries: list[str] = field(default_factory=list)
    audio_queries: list[str] = field(default_factory=list)
    required_modalities: list[Modality] = field(default_factory=list)
    modality_weights: dict[str, float] = field(default_factory=dict)
    relation: Relation = "any"
    relation_tolerance_ms: int = 4_000
    target: str = "event"
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalHit:
    branch: str
    rank: int
    score: float
    evidence: Evidence


@dataclass
class CandidateInterval:
    id: str
    video_id: str
    start_ms: int
    end_ms: int
    score: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    branch_scores: dict[str, float] = field(default_factory=dict)
    covered_modalities: list[str] = field(default_factory=list)
    missing_modalities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_s": round(self.start_ms / 1000, 3),
            "end_s": round(self.end_ms / 1000, 3),
            "score": self.score,
            "evidence": [item.to_dict() for item in self.evidence],
            "branch_scores": dict(self.branch_scores),
            "covered_modalities": list(self.covered_modalities),
            "missing_modalities": list(self.missing_modalities),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentAction:
    type: ActionType
    start_ms: int | None = None
    end_ms: int | None = None
    fps: float | None = None
    num_frames: int | None = None
    answer: str | None = None
    status: SupportStatus | None = None
    confidence: float | None = None

    def signature(self) -> str:
        return f"{self.type}:{self.start_ms}:{self.end_ms}:{self.fps}:{self.num_frames}:{self.answer}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolObservation:
    id: str
    action: AgentAction
    start_ms: int
    end_ms: int
    media_paths: list[str] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action.to_dict(),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "media_paths": list(self.media_paths),
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentDecision:
    assessment: str
    action: AgentAction


@dataclass(frozen=True)
class AgentConclusion:
    status: SupportStatus
    description: str
    start_ms: int
    end_ms: int
    confidence: float
    missing_evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VerificationResult:
    status: SupportStatus
    event_description: str
    start_ms: int
    end_ms: int
    confidence: float
    evidence: dict[str, list[str]] = field(default_factory=dict)
    missing_evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceResult:
    video_id: str
    video_path: str
    start_ms: int
    end_ms: int
    confidence: float
    description: str
    evidence: dict[str, list[str]]
    retrieval_score: float
    grounding_method: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start_s"] = round(self.start_ms / 1000, 3)
        payload["end_s"] = round(self.end_ms / 1000, 3)
        return payload


@dataclass(frozen=True)
class SearchResponse:
    query: str
    plan: QueryPlan
    results: list[EvidenceResult]
    candidates_examined: int
    index_version: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_plan": self.plan.to_dict(),
            "results": [item.to_dict() for item in self.results],
            "candidates_examined": self.candidates_examined,
            "index_version": self.index_version,
            "warnings": list(self.warnings),
        }
