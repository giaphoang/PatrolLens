from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from ..config import IngestionConfig
from ..domain import VideoAsset


def _utc_timestamp(timestamp: float | None = None) -> str:
    value = datetime.fromtimestamp(timestamp, UTC) if timestamp is not None else datetime.now(UTC)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class IngestionCostRates:
    """Configurable pricing assumptions used for a conservative USD estimate."""

    asr_usd_per_hour: float = 0.04
    text_usd_per_million_tokens: float = 0.20
    image_usd_per_million_tokens: float = 0.45
    transcript_tokens_per_minute: float = 240.0
    image_tokens_per_tile: int = 258

    @classmethod
    def from_env(cls) -> IngestionCostRates:
        return cls(
            asr_usd_per_hour=float(
                os.getenv("PATROLLENS_ESTIMATED_ASR_USD_PER_HOUR", "0.04")
            ),
            text_usd_per_million_tokens=float(
                os.getenv(
                    "PATROLLENS_ESTIMATED_EMBEDDING_TEXT_USD_PER_MILLION_TOKENS",
                    "0.20",
                )
            ),
            image_usd_per_million_tokens=float(
                os.getenv(
                    "PATROLLENS_ESTIMATED_EMBEDDING_IMAGE_USD_PER_MILLION_TOKENS",
                    "0.45",
                )
            ),
            transcript_tokens_per_minute=float(
                os.getenv("PATROLLENS_ESTIMATED_TRANSCRIPT_TOKENS_PER_MINUTE", "240")
            ),
            image_tokens_per_tile=int(
                os.getenv("PATROLLENS_ESTIMATED_IMAGE_TOKENS_PER_TILE", "258")
            ),
        )

    def validate(self) -> None:
        values = asdict(self)
        if any(float(value) < 0 for value in values.values()):
            raise ValueError("ingestion cost-estimation rates cannot be negative")
        if self.image_tokens_per_tile <= 0:
            raise ValueError("estimated image tokens per tile must be positive")


def prioritize_oldest(paths: Iterable[Path]) -> list[Path]:
    """Return videos by earliest filesystem update, with path as a stable tie-breaker."""

    return sorted(
        (Path(path).resolve() for path in paths),
        key=lambda path: (path.stat().st_mtime_ns, str(path).casefold()),
    )


def select_pending_batch(
    entries: list[dict[str, Any]],
    video_batch_size: int | None,
) -> list[dict[str, Any]]:
    if video_batch_size is not None and video_batch_size <= 0:
        raise ValueError("video batch size must be positive")
    pending = [entry for entry in entries if entry.get("scheduling_state") != "complete"]
    return pending if video_batch_size is None else pending[:video_batch_size]


def _estimated_image_tiles(width: int | None, height: int | None) -> int:
    if not width or not height or (width <= 384 and height <= 384):
        return 1
    crop = max(256, min(768, math.floor(min(width, height) / 1.5)))
    return max(1, math.ceil(width / crop) * math.ceil(height / crop))


def _existing_keyframe_count(
    asset: VideoAsset,
    artifact_root: Path,
    config: IngestionConfig,
) -> int | None:
    manifest_path = artifact_root / "media" / "keyframes" / asset.id / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not (
        manifest.get("video_sha256") == asset.sha256
        and manifest.get("frame_step_ms") == config.frame_step_ms
        and manifest.get("duplicate_distance") == config.visual_duplicate_distance
        and manifest.get("scene_change_distance") == config.visual_scene_change_distance
    ):
        return None
    records = manifest.get("keyframes")
    return len(records) if isinstance(records, list) else None


def estimate_video_cost(
    asset: VideoAsset,
    *,
    artifact_root: str | Path,
    config: IngestionConfig,
    scheduling_state: str,
    asr_enabled: bool,
    remote_asr_enabled: bool,
    embedding_enabled: bool,
    image_embedding_enabled: bool,
    clap_enabled: bool,
    rates: IngestionCostRates,
) -> dict[str, Any]:
    """Estimate gross and incremental remote indexing cost for one video."""

    rates.validate()
    duration_hours = asset.duration_ms / 3_600_000
    duration_minutes = asset.duration_ms / 60_000
    sampled_frames = math.ceil(asset.duration_ms / config.frame_step_ms)
    known_keyframes = _existing_keyframe_count(
        asset, Path(artifact_root), config
    ) if image_embedding_enabled else None
    estimated_images = (
        known_keyframes
        if known_keyframes is not None
        else sampled_frames
    ) if image_embedding_enabled else 0
    image_basis = (
        "existing_keyframe_manifest"
        if known_keyframes is not None
        else "conservative_all_sampled_frames_unique"
    )
    image_tiles = _estimated_image_tiles(asset.width, asset.height)
    image_tokens = estimated_images * image_tiles * rates.image_tokens_per_tile
    transcript_tokens = (
        math.ceil(duration_minutes * rates.transcript_tokens_per_minute)
        if asr_enabled and embedding_enabled and asset.has_audio
        else 0
    )
    asr_cost = (
        duration_hours * rates.asr_usd_per_hour
        if remote_asr_enabled and asset.has_audio
        else 0.0
    )
    text_cost = transcript_tokens / 1_000_000 * rates.text_usd_per_million_tokens
    image_cost = image_tokens / 1_000_000 * rates.image_usd_per_million_tokens
    gross = asr_cost + text_cost + image_cost
    incremental = 0.0 if scheduling_state in {"complete", "clap_backfill"} else gross
    clap_windows = (
        max(
            0,
            math.ceil(
                max(0, asset.duration_ms - config.clap_window_ms)
                / config.clap_stride_ms
            )
            + 1,
        )
        if clap_enabled and asset.has_audio
        else 0
    )
    return {
        "currency": "USD",
        "scheduling_state": scheduling_state,
        "gross_estimated_cost_usd": round(gross, 8),
        "incremental_estimated_cost_usd": round(incremental, 8),
        "asr": {
            "enabled": asr_enabled and asset.has_audio,
            "remote_pricing_applied": remote_asr_enabled and asset.has_audio,
            "audio_hours": round(duration_hours, 6),
            "rate_usd_per_hour": rates.asr_usd_per_hour,
            "estimated_cost_usd": round(asr_cost, 8),
        },
        "text_embeddings": {
            "enabled": asr_enabled and embedding_enabled and asset.has_audio,
            "estimated_tokens": transcript_tokens,
            "tokens_per_minute_assumption": rates.transcript_tokens_per_minute,
            "rate_usd_per_million_tokens": rates.text_usd_per_million_tokens,
            "estimated_cost_usd": round(text_cost, 8),
        },
        "image_embeddings": {
            "enabled": image_embedding_enabled,
            "estimated_images": estimated_images,
            "estimate_basis": image_basis,
            "sampled_frames_upper_bound": sampled_frames if image_embedding_enabled else 0,
            "estimated_tiles_per_image": image_tiles if image_embedding_enabled else 0,
            "tokens_per_tile_assumption": rates.image_tokens_per_tile,
            "estimated_tokens": image_tokens,
            "rate_usd_per_million_tokens": rates.image_usd_per_million_tokens,
            "estimated_cost_usd": round(image_cost, 8),
        },
        "local_clap": {
            "enabled": clap_enabled and asset.has_audio,
            "estimated_windows": clap_windows,
            "estimated_remote_cost_usd": 0.0,
        },
        "cache_note": (
            "No remote work expected because this video is already complete."
            if scheduling_state == "complete"
            else "Only local CLAP backfill is expected."
            if scheduling_state == "clap_backfill"
            else "Incremental cost is conservative; durable ASR/embedding cache hits can reduce it."
        ),
    }


def build_cost_report(
    entries: list[dict[str, Any]],
    *,
    input_root: str | Path,
    artifact_root: str | Path,
    video_batch_size: int | None,
    rates: IngestionCostRates,
) -> dict[str, Any]:
    selected = [entry for entry in entries if entry.get("selected_for_batch")]
    pending = [entry for entry in entries if entry.get("scheduling_state") != "complete"]
    remaining = [entry for entry in pending if not entry.get("selected_for_batch")]

    def total(items: list[dict[str, Any]], field: str) -> float:
        return round(sum(float(item["cost_estimate"][field]) for item in items), 8)

    return {
        "schema_version": "1.0",
        "generated_at": _utc_timestamp(),
        "input": str(Path(input_root).expanduser().resolve()),
        "artifact_root": str(Path(artifact_root).expanduser().resolve()),
        "priority": "oldest_file_modification_first",
        "video_batch_size": video_batch_size,
        "pricing_assumptions": {
            **asdict(rates),
            "pricing_snapshot_date": "2026-08-28",
            "asr_source": "https://openrouter.ai/openai/whisper-large-v3-turbo/pricing",
            "embedding_source": "https://openrouter.ai/google/gemini-embedding-2/pricing",
        },
        "summary": {
            "video_count": len(entries),
            "pending_video_count": len(pending),
            "selected_video_count": len(selected),
            "selected_batch_estimated_cost_usd": total(
                selected, "incremental_estimated_cost_usd"
            ),
            "pending_corpus_estimated_cost_usd": total(
                pending, "incremental_estimated_cost_usd"
            ),
            "remaining_after_selected_batch_estimated_cost_usd": total(
                remaining, "incremental_estimated_cost_usd"
            ),
            "gross_corpus_estimated_cost_usd": total(
                entries, "gross_estimated_cost_usd"
            ),
        },
        "videos": entries,
    }


def write_cost_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at"] = _utc_timestamp()
    temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def file_update_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "updated_at": _utc_timestamp(stat.st_mtime),
        "updated_at_ns": stat.st_mtime_ns,
        "source_bytes": stat.st_size,
    }
