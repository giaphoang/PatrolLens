from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_GEMINI_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_GEMINI_EMBEDDING_MODEL = "google/gemini-embedding-2"
DEFAULT_GEMINI_EMBEDDING_BATCH_MODEL = "google/gemini-embedding-2:batch"


@dataclass(frozen=True)
class IngestionConfig:
    window_ms: int = 16_000
    stride_ms: int = 8_000
    frame_step_ms: int = 1_000
    audio_window_ms: int = 4_000
    audio_stride_ms: int = 2_000
    batch_size: int = 16
    ocr_min_confidence: float = 0.45
    visual_duplicate_distance: int = 8
    visual_scene_change_distance: int = 18
    schema_version: str = "3.0.0"
    embedding_preprocessing_version: str = "scene-keyframe-v1"
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
class AgentConfig:
    model: str = DEFAULT_GEMINI_MODEL
    planner_model: str = DEFAULT_GEMINI_MODEL
    max_turns: int = 5
    max_frames_per_action: int = 12
    max_audio_ms: int = 30_000
    max_clip_ms: int = 20_000
    max_inline_media_bytes: int = 18 * 1024 * 1024
    run_root: str = ".patrol-lens/runs"

    @classmethod
    def from_env(cls, *, model: str | None = None, planner_model: str | None = None) -> AgentConfig:
        resolved = model or os.getenv("PATROLLENS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        return cls(
            model=resolved,
            planner_model=planner_model or os.getenv("PATROLLENS_GEMINI_PLANNER_MODEL", resolved),
            max_turns=int(os.getenv("PATROLLENS_AGENT_MAX_TURNS", "5")),
            run_root=os.getenv("PATROLLENS_RUN_ROOT", ".patrol-lens/runs"),
        )


@dataclass(frozen=True)
class RefinementConfig:
    context_ms: int = 7_000
    frame_fps: float = 6.0
    max_interval_ms_without_specialist: int = 20_000
    timelens_confidence_threshold: float = 0.65
