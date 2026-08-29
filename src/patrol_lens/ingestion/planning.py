from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from ..config import IngestionConfig
from ..domain import VideoAsset


_KNOWN_EMBEDDING_PRICES: dict[str, dict[str, float]] = {
    # OpenRouter prices in USD per million input tokens as of 2026-08-28.
    "google/gemini-embedding-2": {
        "text": 0.20,
        "image": 0.45,
    },
    "google/gemini-embedding-2:batch": {
        "text": 0.10,
        "image": 0.225,
    },
}


def _model_env_suffix(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").upper()


def _embedding_model_for_pricing(embedding_model: str | None) -> str | None:
    return embedding_model or os.getenv("PATROLLENS_EMBEDDING_BATCH_MODEL") or os.getenv(
        "PATROLLENS_EMBEDDING_MODEL"
    )


def _embedding_rate_defaults(embedding_model: str | None) -> dict[str, float]:
    model = _embedding_model_for_pricing(embedding_model)
    if model in _KNOWN_EMBEDDING_PRICES:
        return dict(_KNOWN_EMBEDDING_PRICES[model])
    return {"text": 0.20, "image": 0.45}


def _embedding_rate_env_names(kind: str, embedding_model: str | None) -> tuple[str, str]:
    generic = f"PATROLLENS_ESTIMATED_EMBEDDING_{kind.upper()}_USD_PER_MILLION_TOKENS"
    model = _embedding_model_for_pricing(embedding_model)
    specific = (
        f"{generic}_{_model_env_suffix(model)}"
        if model
        else generic
    )
    return specific, generic


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
    def from_env(
        cls,
        *,
        embedding_model: str | None = None,
    ) -> IngestionCostRates:
        defaults = _embedding_rate_defaults(embedding_model)

        def rate(kind: str) -> float:
            specific, generic = _embedding_rate_env_names(kind, embedding_model)
            value = os.getenv(specific)
            if value is None:
                value = os.getenv(generic, str(defaults[kind]))
            return float(value)

        return cls(
            asr_usd_per_hour=float(
                os.getenv("PATROLLENS_ESTIMATED_ASR_USD_PER_HOUR", "0.04")
            ),
            text_usd_per_million_tokens=rate("text"),
            image_usd_per_million_tokens=rate("image"),
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

    @classmethod
    def pricing_metadata(cls, embedding_model: str | None = None) -> dict[str, Any]:
        """Describe where model-specific estimate rates came from."""

        model = _embedding_model_for_pricing(embedding_model)
        defaults = _embedding_rate_defaults(model)
        metadata: dict[str, Any] = {
            "embedding_model": model,
            "pricing_catalog": "openrouter",
            "pricing_snapshot_date": datetime.now(UTC).date().isoformat(),
        }
        if not model:
            metadata.update(
                {
                    "text_rate_source": "default_assumption",
                    "image_rate_source": "default_assumption",
                    "pricing_status": "no_embedding_model",
                }
            )
            return metadata

        for kind, label in (("text", "text_rate_source"), ("image", "image_rate_source")):
            specific, generic = _embedding_rate_env_names(kind, model)
            if os.getenv(specific) is not None:
                metadata[label] = f"environment:{specific}"
            elif os.getenv(generic) is not None:
                metadata[label] = f"environment:{generic}"
            elif model in _KNOWN_EMBEDDING_PRICES:
                metadata[label] = "openrouter_catalog"
            else:
                metadata[label] = "default_assumption"

        metadata["pricing_status"] = (
            "catalog" if model in _KNOWN_EMBEDDING_PRICES else "unverified_fallback"
        )
        has_model_specific_override = any(
            os.getenv(_embedding_rate_env_names(kind, model)[0]) is not None
            for kind in ("text", "image")
        )
        if model not in _KNOWN_EMBEDDING_PRICES and not has_model_specific_override:
            metadata["warning"] = (
                "No model-specific OpenRouter rate is known; set the model-specific "
                "PATROLLENS_ESTIMATED_EMBEDDING_* environment variables."
            )
        metadata["catalog_rates_usd_per_million_tokens"] = defaults
        return metadata


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


def new_cost_report_path(artifact_root: str | Path) -> Path:
    """Return a collision-resistant report path for one ingestion invocation."""

    root = Path(artifact_root).expanduser().resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"
    return root / "reports" / f"ingestion-cost-estimate-{timestamp}-{uuid.uuid4().hex[:8]}.json"


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
    embedding_model: str | None = None,
) -> dict[str, Any]:
    selected = [entry for entry in entries if entry.get("selected_for_batch")]
    pending = [entry for entry in entries if entry.get("scheduling_state") != "complete"]
    remaining = [entry for entry in pending if not entry.get("selected_for_batch")]

    def total(items: list[dict[str, Any]], field: str) -> float:
        return round(sum(float(item["cost_estimate"][field]) for item in items), 8)

    report = {
        "schema_version": "1.1",
        "generated_at": _utc_timestamp(),
        "input": str(Path(input_root).expanduser().resolve()),
        "artifact_root": str(Path(artifact_root).expanduser().resolve()),
        "priority": "oldest_file_modification_first",
        "video_batch_size": video_batch_size,
        "pricing_assumptions": {
            **asdict(rates),
            **IngestionCostRates.pricing_metadata(embedding_model),
            "asr_source": "https://openrouter.ai/openai/whisper-large-v3-turbo/pricing",
            "embedding_source": (
                f"https://openrouter.ai/{embedding_model}/pricing"
                if embedding_model
                else None
            ),
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
        "cost_reconciliation": {
            "provider_usage_source": "openrouter_response.usage",
            "provider_usage_note": (
                "Provider-reported token usage and cost are read from each OpenRouter "
                "response; no extra usage API call is required."
            ),
            "local_cost_policy": (
                "Local model execution is recorded with actual cost 0.0 USD; its "
                "latency and memory remain part of runtime telemetry."
            ),
            "estimate_fallback_policy": (
                "A reconciled total falls back to the configured estimate whenever a "
                "remote response omits provider cost or execution is incomplete."
            ),
        },
    }
    return reconcile_cost_report(report)


def _round_cost(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return round(max(0.0, float(value)), 8)
    except (TypeError, ValueError):
        return None


def _runtime_cost(runtime: Any) -> tuple[float | None, str]:
    """Return a runtime cost and source without treating missing cost as zero."""

    if not isinstance(runtime, dict):
        return None, "unavailable"
    source = str(runtime.get("cost_source") or "")
    if source == "local_zero":
        return 0.0, source
    api_calls = int(runtime.get("api_calls", runtime.get("calls", 0)) or 0)
    if runtime.get("cost_available") is False:
        return None, "unavailable"
    cost = _round_cost(runtime.get("reported_cost_usd"))
    if cost is not None:
        return cost, source or "provider"
    if api_calls == 0 and source in {
        "not_called",
        "cache_only",
        "checkpoint_only",
        "no_remote_work",
    }:
        return 0.0, source
    return None, "unavailable"


def _execution_runtime_components(execution: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(execution, dict):
        return {}
    runtime = execution.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        # Keep reports generated by the previous implementation readable.
        if isinstance(execution.get("asr_runtime"), dict):
            runtime["asr"] = execution["asr_runtime"]
        if isinstance(execution.get("embedding_runtime"), dict):
            runtime["embedding"] = execution["embedding_runtime"]
    components: dict[str, dict[str, Any]] = {}
    for name in ("asr", "embedding"):
        value = runtime.get(name)
        if isinstance(value, dict):
            components[name] = value
    local = runtime.get("local")
    if isinstance(local, dict):
        for name, value in local.items():
            if isinstance(value, dict):
                components[f"local.{name}"] = value
    return components


def _expected_remote_components(entry: dict[str, Any]) -> set[str]:
    estimate = entry.get("cost_estimate")
    if not isinstance(estimate, dict):
        return set()
    expected: set[str] = set()
    asr = estimate.get("asr")
    if isinstance(asr, dict) and asr.get("remote_pricing_applied"):
        expected.add("asr")
    text = estimate.get("text_embeddings")
    image = estimate.get("image_embeddings")
    if (
        isinstance(text, dict) and text.get("enabled")
    ) or (
        isinstance(image, dict) and image.get("enabled")
    ):
        expected.add("embedding")
    return expected


def reconcile_video_cost(entry: dict[str, Any]) -> dict[str, Any]:
    """Merge one video estimate with runtime/provider usage, if executed."""

    estimate = entry.get("cost_estimate")
    estimated = 0.0
    if isinstance(estimate, dict):
        estimated = _round_cost(estimate.get("incremental_estimated_cost_usd")) or 0.0
    execution = entry.get("execution")
    status = execution.get("status") if isinstance(execution, dict) else None
    base = {
        "status": "not_executed",
        "estimated_cost_usd": estimated,
        "provider_reported_cost_usd": None,
        "provider_observed_cost_usd": None,
        "provider_cost_complete": False,
        "local_cost_usd": 0.0,
        "actual_cost_usd": None,
        "reconciled_cost_usd": estimated,
        "cost_source": "estimate_pending",
        "components": {},
    }
    if status not in {"complete", "failed"}:
        return base

    components = _execution_runtime_components(execution)
    expected_remote = _expected_remote_components(entry)
    missing_remote = expected_remote - set(components)
    component_reports: dict[str, Any] = {}
    provider_observed = 0.0
    local_cost = 0.0
    unknown_provider_cost = bool(missing_remote)
    has_provider_component = False
    for name, runtime in components.items():
        cost, source = _runtime_cost(runtime)
        is_local = name.startswith("local.") or source == "local_zero"
        has_provider_component = has_provider_component or not is_local
        if cost is None and not is_local:
            unknown_provider_cost = True
        if cost is not None:
            if is_local:
                local_cost += cost
            else:
                provider_observed += cost
        component_reports[name] = {
            "reported_cost_usd": cost,
            "cost_source": source,
            "api_calls": int(runtime.get("api_calls", runtime.get("calls", 0)) or 0),
            "latency_ms": _round_cost(runtime.get("latency_ms")),
        }
    for name in sorted(missing_remote):
        unknown_provider_cost = True
        component_reports[name] = {
            "reported_cost_usd": None,
            "cost_source": "missing_runtime",
            "api_calls": None,
            "latency_ms": None,
        }

    executed_runtime = bool(components) or status == "complete"
    actual = None if unknown_provider_cost else _round_cost(provider_observed + local_cost)
    if has_provider_component and local_cost:
        source = "provider+local_zero"
    elif has_provider_component:
        source = "provider"
    elif components:
        source = "local_zero"
    else:
        source = "zero_remote_work"
    if actual is not None and status == "complete":
        reconciliation_status = "actual"
    elif actual is not None:
        reconciliation_status = "partial_execution"
    elif executed_runtime:
        reconciliation_status = "partial"
        source = "estimate_fallback"
    else:
        reconciliation_status = "unavailable"
        source = "estimate_fallback"
    return {
        "status": reconciliation_status,
        "estimated_cost_usd": estimated,
        "provider_reported_cost_usd": (
            _round_cost(provider_observed) if not unknown_provider_cost else None
        ),
        "provider_observed_cost_usd": _round_cost(provider_observed),
        "provider_cost_complete": not unknown_provider_cost,
        "local_cost_usd": _round_cost(local_cost) or 0.0,
        "actual_cost_usd": actual,
        "reconciled_cost_usd": actual if actual is not None else estimated,
        "cost_source": source,
        "components": component_reports,
    }


def _aggregate_reconciliations(items: list[dict[str, Any]]) -> dict[str, Any]:
    reconciliations = [
        item.get("cost_reconciliation")
        for item in items
        if isinstance(item.get("cost_reconciliation"), dict)
    ]
    estimated = round(
        sum(float(item.get("estimated_cost_usd") or 0.0) for item in reconciliations),
        8,
    )
    reconciled = round(
        sum(float(item.get("reconciled_cost_usd") or 0.0) for item in reconciliations),
        8,
    )
    executed = [item for item in reconciliations if item.get("status") != "not_executed"]
    actual_values = [item.get("actual_cost_usd") for item in executed]
    actual = (
        round(sum(float(value) for value in actual_values), 8)
        if executed and all(value is not None for value in actual_values)
        else None
    )
    observed = (
        round(
            sum(float(item.get("provider_observed_cost_usd") or 0.0) for item in executed),
            8,
        )
        if executed
        else None
    )
    local = round(
        sum(float(item.get("local_cost_usd") or 0.0) for item in executed),
        8,
    )
    if not executed:
        status = "estimate_only"
    elif actual is not None and len(executed) == len(reconciliations):
        status = "actual"
    else:
        status = "partial"
    return {
        "estimated_cost_usd": estimated,
        "reported_provider_cost_usd": observed,
        "local_cost_usd": local,
        "actual_cost_usd": actual,
        "reconciled_cost_usd": reconciled,
        "cost_status": status,
    }


def reconcile_cost_report(report: dict[str, Any]) -> dict[str, Any]:
    """Refresh report-level cost fields after an ingestion step completes."""

    entries = report.get("videos")
    if not isinstance(entries, list):
        return report
    for entry in entries:
        if isinstance(entry, dict):
            entry["cost_reconciliation"] = reconcile_video_cost(entry)
    selected = [entry for entry in entries if entry.get("selected_for_batch")]
    pending = [entry for entry in entries if entry.get("scheduling_state") != "complete"]
    remaining = [entry for entry in pending if not entry.get("selected_for_batch")]
    summary = report.setdefault("summary", {})
    for label, items in (
        ("selected_batch", selected),
        ("pending_corpus", pending),
        ("remaining_after_selected_batch", remaining),
    ):
        aggregate = _aggregate_reconciliations(items)
        for field, value in aggregate.items():
            summary[f"{label}_{field}"] = value

    execution = report.get("execution")
    canary = execution.get("canary_runtime") if isinstance(execution, dict) else None
    selected_aggregate = _aggregate_reconciliations(selected)
    if isinstance(canary, dict):
        canary_cost, canary_source = _runtime_cost(canary)
    else:
        canary_cost, canary_source = None, "not_executed"
    selected_observed = selected_aggregate["reported_provider_cost_usd"]
    turn_observed = (
        _round_cost(
            float(selected_observed or 0.0)
            + float(canary_cost or 0.0)
        )
        if selected_observed is not None or canary_cost is not None
        else None
    )
    selected_actual = selected_aggregate["actual_cost_usd"]
    if canary is None:
        turn_actual = selected_actual
        turn_reconciled = selected_aggregate["reconciled_cost_usd"]
    elif selected_actual is not None and canary_cost is not None:
        turn_actual = _round_cost(float(selected_actual) + float(canary_cost))
        turn_reconciled = _round_cost(
            float(selected_aggregate["reconciled_cost_usd"])
            + float(canary_cost)
        )
    else:
        turn_actual = None
        # There is no safe estimate for a canary whose provider cost is absent.
        turn_reconciled = None
    if selected_aggregate["cost_status"] == "estimate_only" and canary is None:
        turn_status = "estimate_only"
    elif turn_actual is not None and selected_aggregate["cost_status"] == "actual":
        turn_status = "actual"
    else:
        turn_status = "partial"
    summary.update(
        {
            "ingestion_turn_reported_provider_cost_usd": turn_observed,
            "ingestion_turn_actual_cost_usd": turn_actual,
            "ingestion_turn_local_cost_usd": selected_aggregate["local_cost_usd"],
            "ingestion_turn_reconciled_cost_usd": turn_reconciled,
            "ingestion_turn_cost_status": turn_status,
        }
    )
    report["execution_cost_reconciliation"] = {
        "canary": {
            "reported_cost_usd": canary_cost,
            "cost_source": canary_source,
            "api_calls": int(canary.get("api_calls", 0) or 0)
            if isinstance(canary, dict)
            else 0,
            "latency_ms": _round_cost(canary.get("latency_ms"))
            if isinstance(canary, dict)
            else None,
        },
        "selected_batch": selected_aggregate,
        "turn": {
            "reported_provider_cost_usd": turn_observed,
            "actual_cost_usd": turn_actual,
            "local_cost_usd": selected_aggregate["local_cost_usd"],
            "reconciled_cost_usd": turn_reconciled,
            "cost_status": turn_status,
        },
        "note": "Embedding canary usage is tracked separately from per-video estimates.",
    }
    return report


def write_cost_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    reconcile_cost_report(report)
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
