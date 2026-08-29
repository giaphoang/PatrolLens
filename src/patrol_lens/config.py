from __future__ import annotations

from dataclasses import dataclass

DEFAULT_GEMINI_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_GEMINI_EMBEDDING_MODEL = "google/gemini-embedding-2"
DEFAULT_GEMINI_EMBEDDING_BATCH_MODEL = "google/gemini-embedding-2:batch"


@dataclass(frozen=True)
class IngestionConfig:
    window_ms: int = 16_000
    stride_ms: int = 8_000
    frame_step_ms: int = 1_000
    clap_window_ms: int = 10_000
    clap_stride_ms: int = 5_000
    batch_size: int = 16
    ocr_min_confidence: float = 0.45
    visual_duplicate_distance: int = 8
    visual_scene_change_distance: int = 18
    schema_version: str = "3.0.0"
    embedding_preprocessing_version: str = "scene-keyframe-v1"
    clap_preprocessing_version: str = "coreml-int8-48khz-mono-peaknorm-v1"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 6
    embed_images: bool = True
    reuse_existing_transcripts: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    branch_k: int = 60
    top_k: int = 12
    rrf_constant: int = 60
    temporal_tolerance_ms: int = 4_000
    merge_gap_ms: int = 3_000
    candidate_padding_ms: int = 5_000
    max_candidate_ms: int = 45_000
    require_conjunctive_modalities: bool = True


@dataclass(frozen=True)
class SearchConfig:
    """Optional latency and cost controls for full candidate verification."""

    candidate_parallelism: int = 1
    early_stop_confidence: float | None = None
    timeout_s: float | None = None
    max_run_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.candidate_parallelism <= 0:
            raise ValueError("candidate parallelism must be positive")
        if self.early_stop_confidence is not None and not 0 <= self.early_stop_confidence <= 1:
            raise ValueError("early-stop confidence must be between 0 and 1")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("search timeout must be positive")
        if self.max_run_cost_usd is not None and self.max_run_cost_usd <= 0:
            raise ValueError("maximum run cost must be positive")


@dataclass(frozen=True)
class RefinementConfig:
    context_ms: int = 7_000
    frame_fps: float = 6.0
    max_interval_ms_without_specialist: int = 20_000
    timelens_confidence_threshold: float = 0.65
