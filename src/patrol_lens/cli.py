from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .adapters.asr import (
    DEFAULT_OPENROUTER_ASR_MODEL,
    FasterWhisperASR,
    OpenRouterASR,
)
from .adapters.clap import DEFAULT_CLAP_MODEL, ClapCoreMLBackend
from .adapters.media import compress_video_480p, extract_audio, iter_video_files, probe_video
from .adapters.openrouter import (
    EmbeddingDimensionError,
    OpenRouterEmbeddingClient,
    OpenRouterJSONClient,
)
from .adapters.visual import SigLIP2Encoder
from .agent import ActivePerceptionAgent, GeminiActivePolicy
from .asr_benchmark import benchmark_backend, transcript_text, word_error_rate
from .config import (
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_MODEL,
    AgentConfig,
    IngestionConfig,
    RetrievalConfig,
    SearchConfig,
)
from .domain import VideoAsset
from .evaluate import evaluate_file
from .index import AutoVectorIndex, IndexStore, PostgresIndexStore, PostgresVectorIndex
from .ingestion import IngestionBackends, IngestionPipeline
from .ingestion.planning import (
    IngestionCostRates,
    build_cost_report,
    estimate_video_cost,
    file_update_metadata,
    new_cost_report_path,
    prioritize_oldest,
    reconcile_cost_report,
    select_pending_batch,
    write_cost_report,
)
from .history import TrajectoryRecorder, list_history, show_history
from .pipeline import SearchPipeline
from .retrieval import CoarseRetriever, GeminiQueryPlanner, HeuristicQueryPlanner
from .temporal import LightweightTimestampRefiner, TimeLens2Adapter
from .verification import GeminiEventVerifier


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _redact_database_url(value: object) -> object:
    if not isinstance(value, str):
        return value
    if "://" not in value:
        return re.sub(
            r"(?i)(\bpassword\s*=\s*)(?:'[^']*'|\"[^\"]*\"|\S+)",
            r"\1***",
            value,
        )
    try:
        parsed = urlsplit(value)
        if parsed.password is None:
            return value
        hostname = parsed.hostname or ""
        if parsed.port:
            hostname = f"{hostname}:{parsed.port}"
        username = f"{parsed.username}:***@" if parsed.username else "***@"
        return urlunsplit((parsed.scheme, username + hostname, parsed.path, parsed.query, parsed.fragment))
    except ValueError:
        return "<redacted-database-url>"


def _history_parameters(args: argparse.Namespace) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for key, value in vars(args).items():
        if key in {"func", "query"}:
            continue
        parameters[key] = _redact_database_url(value) if key == "database_url" else value
    return parameters


def _recorder(args: argparse.Namespace) -> TrajectoryRecorder:
    return TrajectoryRecorder(
        args.index,
        query=args.query,
        command=args.command,
        parameters=_history_parameters(args),
        max_cost_usd=getattr(args, "max_run_cost_usd", None),
        estimated_model_call_cost_usd=float(
            os.getenv("PATROLLENS_ESTIMATED_MODEL_CALL_COST_USD", "0.02")
        ),
    )


def _run_metadata(recorder: TrajectoryRecorder) -> dict[str, object]:
    return {
        "run_id": recorder.run_id,
        "status": recorder.summary["status"],
        "last_completed_stage": recorder.summary["last_completed_stage"],
        "candidates_retrieved": recorder.summary["candidates_retrieved"],
        "candidates_examined": recorder.summary["candidates_examined"],
        "best_partial_result": recorder.summary["best_partial_result"],
        "total_cost_usd": recorder.summary["total_cost_usd"],
        "estimated_upper_bound_cost_usd": recorder.summary[
            "estimated_upper_bound_cost_usd"
        ],
        "elapsed_seconds": recorder.summary["elapsed_seconds"],
        "termination_reason": recorder.summary["termination_reason"],
        "trajectory_path": recorder.summary["trajectory_path"],
    }


Store = IndexStore | PostgresIndexStore
VectorIndex = AutoVectorIndex | PostgresVectorIndex


def _resolved_transcriber(value: str) -> str:
    return "openrouter" if value == "auto" else value


def _asr_backend(args: argparse.Namespace, *, selection: str | None = None):
    selected = _resolved_transcriber(selection or args.transcriber)
    if selected == "faster_whisper":
        return FasterWhisperASR(
            args.faster_whisper_model,
            device=args.device,
            compute_type=args.compute_type,
            word_timestamps=False,
        )
    return OpenRouterASR(
        model=args.asr_model,
        base_url=args.openrouter_base_url,
        http_referer=args.openrouter_http_referer,
        title=args.openrouter_title,
        language=args.asr_language,
        chunk_seconds=args.asr_chunk_seconds,
        timeout_s=args.asr_timeout_seconds,
        max_retries=args.asr_max_retries,
    )


def _load_project_env() -> None:
    """Load the nearest repository ``.env`` before CLI defaults are parsed.

    ``override=True`` is intentional: the local project file is the
    authoritative configuration for this CLI, which prevents a stale
    OPENROUTER_API_KEY exported by an older shell from being selected.
    """

    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "python-dotenv is not installed; run uv sync before using the CLI"
        ) from exc
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)


def _open_store(args: argparse.Namespace) -> Store:
    if args.backend == "postgres":
        return PostgresIndexStore(args.database_url, root=args.index)
    return IndexStore(args.index)


def _vector_index(store: Store) -> VectorIndex:
    if isinstance(store, PostgresIndexStore):
        return PostgresVectorIndex(store)
    return AutoVectorIndex(store)


def _gemini(
    model: str,
    args: argparse.Namespace,
    *,
    recorder: TrajectoryRecorder | None = None,
) -> OpenRouterJSONClient:
    return OpenRouterJSONClient(
        model=model,
        base_url=args.openrouter_base_url,
        http_referer=args.openrouter_http_referer,
        title=args.openrouter_title,
        recorder=recorder,
    )


def _embedding(
    args: argparse.Namespace,
    *,
    recorder: TrajectoryRecorder | None = None,
) -> OpenRouterEmbeddingClient:
    model = os.getenv("PATROLLENS_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL)
    batch_model = os.getenv("PATROLLENS_EMBEDDING_BATCH_MODEL", model)
    query_model = os.getenv("PATROLLENS_EMBEDDING_QUERY_MODEL", model)
    mode = getattr(args, "embedding_mode", "sync")
    batch_api = mode == "batch"
    return OpenRouterEmbeddingClient(
        model=model,
        batch_model=batch_model if batch_api else model,
        query_model=query_model,
        dimensions=args.embedding_dimensions,
        base_url=args.openrouter_base_url,
        http_referer=args.openrouter_http_referer,
        title=args.openrouter_title,
        media_batch_size=getattr(args, "embedding_batch_size", 6),
        batch_api=batch_api,
        batch_poll_interval_s=getattr(args, "embedding_batch_poll_s", 10.0),
        batch_timeout_s=getattr(args, "embedding_batch_timeout_s", 86_400.0),
        batch_checkpoint_dir=(
            Path(args.index) / "batches" / "openrouter-embeddings"
            if batch_api
            else None
        ),
        recorder=recorder,
    )


def _embedding_canary(embedding: OpenRouterEmbeddingClient) -> None:
    """Verify the provider's output size before ingestion can write evidence."""

    vector = embedding.canary_ingestion()
    if len(vector) != embedding.dimensions:
        raise EmbeddingDimensionError(embedding.dimensions, len(vector))


def _reset_runtime_info(backend: object | None) -> None:
    if backend is None:
        return
    reset = getattr(backend, "reset_runtime_info", None)
    if callable(reset):
        reset()
    elif hasattr(backend, "last_runtime_info"):
        setattr(backend, "last_runtime_info", {})


def _runtime_snapshot(backend: object | None) -> dict[str, object] | None:
    if backend is None:
        return None
    value = getattr(backend, "last_runtime_info", None)
    return dict(value) if isinstance(value, dict) else None


def _reset_ingestion_runtime(backends: IngestionBackends) -> None:
    for backend in (
        backends.asr,
        backends.embedding,
        backends.visual,
        backends.audio_embedding,
    ):
        _reset_runtime_info(backend)


def _ingestion_runtime(
    backends: IngestionBackends,
    stats: dict[str, object] | None = None,
) -> dict[str, object]:
    """Collect provider ledgers and explicit zero-cost local model entries."""

    stats = stats or {}
    runtime: dict[str, object] = {}
    for name, backend in (
        ("asr", backends.asr),
        ("embedding", backends.embedding),
    ):
        snapshot = _runtime_snapshot(backend)
        if snapshot is not None:
            if name == "embedding":
                snapshot["cache_hits"] = int(
                    stats.get("embedding_cache_hits", 0) or 0
                )
                snapshot["cache_misses"] = int(
                    stats.get("embedding_cache_misses", 0) or 0
                )
            runtime[name] = snapshot

    local: dict[str, object] = {}
    if backends.audio_embedding is not None:
        local["audio_embedding"] = {
            "provider": "local",
            "model": backends.audio_embedding.model_name,
            "api_calls": int(stats.get("clap_model_calls", 0) or 0),
            "cache_hits": int(stats.get("clap_cache_hits", 0) or 0),
            "reported_cost_usd": 0.0,
            "cost_available": True,
            "cost_source": "local_zero",
        }
    if backends.visual is not None and backends.embedding is None:
        local["visual"] = {
            "provider": "local",
            "model": backends.visual.model_name,
            "api_calls": int(stats.get("visual_vectors", 0) or 0),
            "reported_cost_usd": 0.0,
            "cost_available": True,
            "cost_source": "local_zero",
        }
    if local:
        runtime["local"] = local
    return runtime


def _clap_backend(
    args: argparse.Namespace,
    *,
    required: bool,
) -> ClapCoreMLBackend | None:
    root = Path(args.clap_model_root).expanduser().resolve()
    backend = ClapCoreMLBackend(
        root / "model" / "clap_audio_encoder.mlpackage",
        root / "model" / "text_model.onnx",
        root / "tokenizer",
        model_name=args.clap_model,
        compute_units=args.clap_compute_units,
    )
    paths_exist = all(
        path.exists()
        for path in (
            backend.audio_model_path,
            backend.text_model_path,
            backend.tokenizer_path,
        )
    )
    if not required and not paths_exist:
        return None
    backend.validate_setup()
    return backend


def _ingestion_backends(
    args: argparse.Namespace,
    *,
    run_embedding_canary: bool = True,
) -> IngestionBackends:
    if args.profile == "metadata":
        return IngestionBackends()
    use_clap = args.clap if args.clap is not None else args.profile == "full"
    audio_embedding = _clap_backend(args, required=True) if use_clap else None
    asr = None if args.no_asr else _asr_backend(args)
    embedding = None if args.no_embeddings else _embedding(args)
    if embedding is not None and run_embedding_canary:
        _embedding_canary(embedding)
    visual = (
        None
        if args.no_visual or embedding is not None
        else SigLIP2Encoder(args.visual_model, device=args.device)
    )
    return IngestionBackends(
        visual=visual,
        embedding=embedding,
        asr=asr,
        audio_embedding=audio_embedding,
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    if min(
        args.frame_fps,
        args.window_s,
        args.stride_s,
        args.clap_window_s,
        args.clap_stride_s,
    ) <= 0:
        raise ValueError("frame rate, windows, and strides must be positive")
    if args.clap_window_s != 10.0:
        raise ValueError("larger_clap_general CoreML requires --clap-window-s 10")
    if args.clap_stride_s > args.clap_window_s:
        raise ValueError("--clap-stride-s cannot exceed --clap-window-s")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.video_batch_size is not None and args.video_batch_size <= 0:
        raise ValueError("--video-batch-size must be positive")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive")
    if args.embedding_batch_poll_s < 0:
        raise ValueError("--embedding-batch-poll-s cannot be negative")
    if args.embedding_batch_timeout_s <= 0:
        raise ValueError("--embedding-batch-timeout-s must be positive")
    if args.asr_chunk_seconds <= 0 or args.asr_chunk_seconds > 600:
        raise ValueError("--asr-chunk-seconds must be between 1 and 600")
    if args.asr_timeout_seconds <= 0 or args.asr_max_retries < 0:
        raise ValueError("ASR timeout must be positive and retries cannot be negative")
    if args.embedding_dimensions <= 0 or args.embedding_dimensions > 3072:
        raise ValueError("--embedding-dimensions must be between 1 and 3072")
    if args.backend == "postgres" and args.embedding_dimensions != 768:
        raise ValueError("PostgreSQL ingestion requires --embedding-dimensions 768")
    with _open_store(args) as store:
        config = IngestionConfig(
            window_ms=round(args.window_s * 1000),
            stride_ms=round(args.stride_s * 1000),
            frame_step_ms=max(1, round(1000 / args.frame_fps)),
            clap_window_ms=round(args.clap_window_s * 1000),
            clap_stride_ms=round(args.clap_stride_s * 1000),
            batch_size=args.batch_size,
            embedding_dimensions=args.embedding_dimensions,
            embedding_batch_size=args.embedding_batch_size,
            embed_images=not args.no_visual and not args.no_embedding_images,
        )
        backends = _ingestion_backends(args, run_embedding_canary=False)
        pipeline = IngestionPipeline(
            store,
            backends=backends,
            config=config,
            vector_index=_vector_index(store),
        )
        videos = prioritize_oldest(iter_video_files(args.input))
        if not videos:
            raise RuntimeError(f"no supported video files found under {args.input}")
        embedding_batch_model = getattr(backends.embedding, "batch_model", None)
        rates = IngestionCostRates.from_env(
            embedding_model=embedding_batch_model,
        )
        assets: dict[str, VideoAsset] = {}
        entries: list[dict[str, object]] = []
        pending_rank = 0
        for priority_rank, path in enumerate(videos, start=1):
            asset = probe_video(path)
            assets[asset.path] = asset
            scheduling_state = pipeline.ingestion_state(asset, force=args.force)
            if scheduling_state != "complete":
                pending_rank += 1
            entry: dict[str, object] = {
                "priority_rank": priority_rank,
                "pending_rank": pending_rank if scheduling_state != "complete" else None,
                "video_id": asset.id,
                "path": asset.path,
                "duration_seconds": round(asset.duration_ms / 1000, 3),
                "resolution": [asset.width, asset.height],
                "has_audio": asset.has_audio,
                **file_update_metadata(path),
                "scheduling_state": scheduling_state,
                "selected_for_batch": False,
                "cost_estimate": estimate_video_cost(
                    asset,
                    artifact_root=store.root,
                    config=config,
                    scheduling_state=scheduling_state,
                    asr_enabled=backends.asr is not None,
                    remote_asr_enabled=isinstance(backends.asr, OpenRouterASR),
                    embedding_enabled=backends.embedding is not None,
                    image_embedding_enabled=(
                        backends.embedding is not None and config.embed_images
                    ),
                    clap_enabled=backends.audio_embedding is not None,
                    rates=rates,
                ),
                "execution": {"status": "not_selected"},
            }
            entries.append(entry)

        selected_entries = select_pending_batch(entries, args.video_batch_size)
        for entry in selected_entries:
            entry["selected_for_batch"] = True
            entry["execution"] = {"status": "planned"}

        report = build_cost_report(
            entries,
            input_root=args.input,
            artifact_root=store.root,
            video_batch_size=args.video_batch_size,
            rates=rates,
            embedding_model=embedding_batch_model,
        )
        report["execution"] = {
            "status": "estimate_only" if args.estimate_only else "planned",
            "completed_videos": 0,
            "failed_videos": 0,
        }
        report["models"] = {
            "asr": getattr(backends.asr, "model_name", None),
            "embedding": getattr(backends.embedding, "model_name", None),
            "embedding_batch": getattr(backends.embedding, "batch_model", None),
            "embedding_query": getattr(backends.embedding, "query_model", None),
            "embedding_mode": args.embedding_mode,
            "embedding_dimensions": config.embedding_dimensions,
            "audio_embedding": getattr(backends.audio_embedding, "model_name", None),
        }
        report_path = write_cost_report(
            report,
            args.cost_report or new_cost_report_path(store.root),
        )
        if args.estimate_only or not selected_entries:
            _print(
                {
                    "index": str(store.root),
                    "cost_report": str(report_path),
                    "estimate": report["summary"],
                    "videos": [],
                }
            )
            return

        if backends.embedding is not None:
            _reset_runtime_info(backends.embedding)
            try:
                _embedding_canary(backends.embedding)
            except Exception as exc:
                report["execution"]["canary_runtime"] = _runtime_snapshot(
                    backends.embedding
                )
                report["execution"] = {
                    "status": "failed",
                    "failed_stage": "embedding_canary",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "completed_videos": 0,
                    "failed_videos": 0,
                    "canary_runtime": report["execution"].get("canary_runtime"),
                }
                write_cost_report(report, report_path)
                raise
            report["execution"]["canary_runtime"] = _runtime_snapshot(
                backends.embedding
            )
            reconcile_cost_report(report)
            write_cost_report(report, report_path)
        stats: list[dict[str, object]] = []
        report["execution"]["status"] = "running"
        write_cost_report(report, report_path)
        for entry in selected_entries:
            asset = assets[str(entry["path"])]
            entry["execution"] = {"status": "running"}
            write_cost_report(report, report_path)
            _reset_ingestion_runtime(backends)
            try:
                video_stats = pipeline.ingest_asset(asset, force=args.force)
            except Exception as exc:
                runtime = _ingestion_runtime(backends)
                entry["execution"] = {
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "runtime": runtime,
                }
                report["execution"]["status"] = "failed"
                report["execution"]["failed_videos"] += 1
                write_cost_report(report, report_path)
                raise
            runtime = _ingestion_runtime(backends, video_stats)
            entry["execution"] = {
                "status": "complete",
                "ingestion_stats": video_stats,
                "runtime": runtime,
            }
            stats.append(video_stats)
            report["execution"]["completed_videos"] += 1
            write_cost_report(report, report_path)
        report["execution"]["status"] = "complete"
        write_cost_report(report, report_path)
        _print(
            {
                "index": str(store.root),
                "cost_report": str(report_path),
                "estimate": report["summary"],
                "videos": stats,
            }
        )


def cmd_compress(args: argparse.Namespace) -> None:
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    source_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if source_root.is_dir() and (
        output_root == source_root or source_root in output_root.parents
    ):
        raise ValueError("--output must be outside the input corpus")

    videos = list(iter_video_files(source_root))
    if not videos:
        raise RuntimeError(f"no supported video files found under {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    destinations: set[Path] = set()
    for source in videos:
        relative = Path(source.name) if source_root.is_file() else source.relative_to(source_root)
        destination = (output_root / relative).with_suffix(".mp4")
        if destination in destinations:
            raise RuntimeError(f"multiple source videos map to {destination}")
        destinations.add(destination)
        skipped = destination.is_file() and destination.stat().st_size > 0 and not args.overwrite
        source_bytes = source.stat().st_size
        compress_video_480p(
            source,
            destination,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )
        output_bytes = destination.stat().st_size
        entries.append(
            {
                "source": str(source),
                "output": str(destination),
                "skipped": skipped,
                "source_bytes": source_bytes,
                "output_bytes": output_bytes,
                "size_reduction_percent": round(
                    (1 - output_bytes / source_bytes) * 100,
                    2,
                ) if source_bytes else 0.0,
            }
        )

    manifest = {
        "input": str(source_root),
        "output": str(output_root),
        "format": "H.264/AAC MP4",
        "maximum_resolution": [854, 480],
        "crf": args.crf,
        "preset": args.preset,
        "videos": entries,
    }
    manifest_path = output_root / "compression-manifest.json"
    temporary_manifest = output_root / ".compression-manifest.partial.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2))
    temporary_manifest.replace(manifest_path)
    _print({**manifest, "manifest": str(manifest_path)})


def cmd_migrate_embeddings(args: argparse.Namespace) -> None:
    if args.backend != "postgres":
        raise ValueError("embedding migration requires --backend postgres")
    if args.embedding_dimensions != 768:
        raise ValueError("embedding migration is fixed at 768 dimensions")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive")
    with _open_store(args) as store:
        if not isinstance(store, PostgresIndexStore):
            raise TypeError("embedding migration requires a PostgreSQL store")
        embedding = _embedding(args)
        _embedding_canary(embedding)
        stats = store.migrate_embeddings_768(
            embedding,
            batch_size=args.embedding_batch_size,
        )
        _print(
            {
                "index": str(store.root),
                "model": embedding.model_name,
                "dimensions": 768,
                **stats,
            }
        )


def cmd_benchmark_asr(args: argparse.Namespace) -> None:
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    backend = _asr_backend(args, selection="openrouter")
    asset = probe_video(source)
    with tempfile.TemporaryDirectory(prefix="patrol-lens-asr-canary-") as temp_dir:
        audio_path = extract_audio(source, Path(temp_dir) / "audio.wav")
        spans, stats = benchmark_backend(
            backend,
            str(audio_path),
            duration_seconds=asset.duration_ms / 1000,
        )
    result: dict[str, object] = {
        "input": str(source),
        "transcription": stats,
    }
    if args.reference_transcript:
        reference = Path(args.reference_transcript).expanduser().read_text()
        result["reference_word_error_rate"] = round(
            word_error_rate(reference, transcript_text(spans)),
            4,
        )
    _print(result)


def _build_retriever(
    args: argparse.Namespace,
    store: Store,
    *,
    client: OpenRouterJSONClient | None = None,
    recorder: TrajectoryRecorder | None = None,
) -> CoarseRetriever:
    if isinstance(store, PostgresIndexStore) and args.embedding_dimensions != 768:
        raise ValueError("PostgreSQL retrieval requires --embedding-dimensions 768")
    if args.planner == "gemini":
        client = client or _gemini(args.planner_model, args, recorder=recorder)
        planner = GeminiQueryPlanner(client, model=args.planner_model)
    else:
        planner = HeuristicQueryPlanner()
    embedding = None if args.no_embeddings else _embedding(args, recorder=recorder)
    audio_encoder = None if args.clap is False else _clap_backend(
        args,
        required=args.clap is True,
    )
    visual = (
        None
        if args.no_visual
        else embedding or SigLIP2Encoder(args.visual_model, device=args.device)
    )
    config = RetrievalConfig(
        branch_k=args.branch_k,
        top_k=args.top_k,
        temporal_tolerance_ms=round(args.temporal_tolerance_s * 1000),
        merge_gap_ms=round(args.merge_gap_s * 1000),
        candidate_padding_ms=round(args.candidate_padding_s * 1000),
    )
    return CoarseRetriever(
        store,
        planner=planner,
        visual_encoder=visual,
        semantic_encoder=embedding,
        audio_encoder=audio_encoder,
        vector_index=_vector_index(store),
        config=config,
        recorder=recorder,
    )


def cmd_retrieve(args: argparse.Namespace) -> None:
    recorder = _recorder(args)
    try:
        with _open_store(args) as store:
            retriever = _build_retriever(args, store, recorder=recorder)
            payload = retriever.search_json(args.query)
        payload["run_id"] = recorder.run_id
        results = payload.get("results", [])
        recorder.finish(
            status="completed",
            termination_reason="retrieval_completed",
            result=payload,
            candidates_retrieved=len(results) if isinstance(results, list) else 0,
            candidates_examined=0,
            result_count=len(results) if isinstance(results, list) else 0,
        )
        payload["run"] = _run_metadata(recorder)
        _print(payload)
    except KeyboardInterrupt:
        recorder.finish(status="cancelled", termination_reason="keyboard_interrupt")
        _print({"run": _run_metadata(recorder)})
        raise
    except Exception as exc:
        recorder.finish(status="failed", termination_reason="error", error=exc)
        _print({"run": _run_metadata(recorder), "error": str(exc)})
        raise


def cmd_search(args: argparse.Namespace) -> None:
    recorder = _recorder(args)
    try:
        with _open_store(args) as store:
            client = (
                None
                if args.coarse_only and args.planner == "heuristic"
                else _gemini(args.model, args, recorder=recorder)
            )
            retriever = _build_retriever(
                args, store, client=client, recorder=recorder
            )
            if args.coarse_only:
                payload = retriever.search_json(args.query)
            else:
                base_agent_config = AgentConfig.from_env(
                    model=args.model, planner_model=args.planner_model
                )
                agent_config = replace(
                    base_agent_config,
                    max_turns=args.max_turns,
                    run_root=str(store.root / "runs"),
                )
                assert client is not None
                agent = ActivePerceptionAgent(
                    GeminiActivePolicy(client, model=args.model),
                    config=agent_config,
                    recorder=recorder,
                )
                verifier = GeminiEventVerifier(client, model=args.model)
                refiner = LightweightTimestampRefiner(client, model=args.model)
                timelens = None
                if args.timelens_command:
                    timelens = TimeLens2Adapter(
                        shlex.split(args.timelens_command),
                        acknowledge_restricted_license=args.acknowledge_timelens_license,
                        timeout_s=args.timelens_timeout,
                    )
                pipeline = SearchPipeline(
                    store,
                    retriever,
                    agent,
                    verifier,
                    refiner,
                    timelens2=timelens,
                    config=SearchConfig(
                        candidate_parallelism=(
                            1
                            if args.candidate_parallelism is None
                            else args.candidate_parallelism
                        ),
                        early_stop_confidence=args.early_stop_confidence,
                        timeout_s=args.search_timeout_s,
                        max_run_cost_usd=args.max_run_cost_usd,
                    ),
                    recorder=recorder,
                )
                payload = pipeline.search(
                    args.query, max_candidates=args.max_candidates
                ).to_dict()
        payload["run_id"] = recorder.run_id
        results = payload.get("results", [])
        result_list = results if isinstance(results, list) else []
        best = result_list[0] if result_list else None
        warnings = payload.get("warnings", [])
        warning_list = warnings if isinstance(warnings, list) else []
        if any(str(item).startswith("run_cost_denied") for item in warning_list):
            status, reason = "denied", "estimated_cost_exceeds_budget"
        elif recorder.budget_exceeded:
            status, reason = "budget_exceeded", "runtime_cost_budget_reached"
        elif any(str(item).startswith("search_timeout") for item in warning_list):
            status, reason = "timeout", "search_timeout"
        else:
            status, reason = "completed", "search_completed"
        recorder.finish(
            status=status,
            termination_reason=reason,
            result=payload if status == "completed" else best,
            best_partial_result=best,
            candidates_examined=int(payload.get("candidates_examined", 0) or 0),
            result_count=len(result_list),
            best_confidence=(
                float(best["confidence"])
                if isinstance(best, dict) and best.get("confidence") is not None
                else None
            ),
        )
        payload["run"] = _run_metadata(recorder)
        _print(payload)
    except KeyboardInterrupt:
        recorder.finish(status="cancelled", termination_reason="keyboard_interrupt")
        _print({"run": _run_metadata(recorder)})
        raise
    except Exception as exc:
        recorder.finish(status="failed", termination_reason="error", error=exc)
        _print({"run": _run_metadata(recorder), "error": str(exc)})
        raise


def cmd_evaluate(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        args.planner = "heuristic"
        retriever = _build_retriever(args, store)
        _print(evaluate_file(args.queries, retriever, args.top_k))


def cmd_history(args: argparse.Namespace) -> None:
    if args.history_action == "show":
        if not args.run_id:
            raise ValueError("history show requires a run_id")
        _print(show_history(args.index, args.run_id))
        return
    if args.run_id:
        raise ValueError("a run_id is only valid with 'history show'")
    _print({"history_root": str(Path(args.index).expanduser() / "history"), "runs": list_history(args.index)})


def cmd_doctor(_args: argparse.Namespace) -> None:
    modules = {
        "openai/OpenRouter": "openai",
        "psycopg": "psycopg",
        "pgvector": "pgvector",
        "faiss": "faiss",
        "faster-whisper": "faster_whisper",
        "SigLIP2/transformers": "transformers",
        "CoreML CLAP": "coremltools",
        "CLAP text/ONNX": "onnxruntime",
    }
    def installed(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            return False

    clap_root = Path(
        os.getenv(
            "PATROLLENS_CLAP_MODEL_ROOT",
            ".patrol-lens-models/larger_clap_general_coreml",
        )
    ).expanduser().resolve()
    _print(
        {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
            "openrouter_api_key_configured": bool(os.getenv("OPENROUTER_API_KEY")),
            "postgres_database_url_configured": bool(os.getenv("PATROLLENS_DATABASE_URL")),
            "openrouter_base_url": os.getenv(
                "PATROLLENS_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            "gemini_model": os.getenv("PATROLLENS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            "default_ingestion_transcriber": _resolved_transcriber(
                os.getenv("PATROLLENS_TRANSCRIBER", "auto")
            ),
            "asr_model": os.getenv(
                "PATROLLENS_ASR_MODEL",
                DEFAULT_OPENROUTER_ASR_MODEL,
            ),
            "clap_model": os.getenv("PATROLLENS_CLAP_MODEL", DEFAULT_CLAP_MODEL),
            "clap_compute_units": os.getenv(
                "PATROLLENS_CLAP_COMPUTE_UNITS",
                "cpu_only",
            ),
            "clap_artifacts": {
                "audio_coreml": (
                    clap_root / "model" / "clap_audio_encoder.mlpackage"
                ).exists(),
                "text_onnx": (clap_root / "model" / "text_model.onnx").is_file(),
                "tokenizer": (clap_root / "tokenizer").is_dir(),
            },
            "python_modules": {name: installed(module) for name, module in modules.items()},
        }
    )


def _add_transcriber_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transcriber",
        "--asr-backend",
        dest="transcriber",
        choices=["auto", "openrouter", "faster_whisper"],
        default=os.getenv("PATROLLENS_TRANSCRIBER", "auto"),
        help="Offline ASR backend; auto selects OpenRouter transcription",
    )
    parser.add_argument(
        "--asr-model",
        default=os.getenv("PATROLLENS_ASR_MODEL", DEFAULT_OPENROUTER_ASR_MODEL),
    )
    parser.add_argument(
        "--asr-language",
        default=os.getenv("PATROLLENS_ASR_LANGUAGE", "auto"),
    )
    parser.add_argument(
        "--asr-chunk-seconds",
        type=int,
        default=int(os.getenv("PATROLLENS_ASR_CHUNK_SECONDS", "300")),
        help="Local checkpoint chunk size before remote transcription",
    )
    parser.add_argument(
        "--asr-timeout-seconds",
        type=float,
        default=float(os.getenv("PATROLLENS_ASR_TIMEOUT_SECONDS", "90")),
    )
    parser.add_argument(
        "--asr-max-retries",
        type=int,
        default=int(os.getenv("PATROLLENS_ASR_MAX_RETRIES", "3")),
    )
    parser.add_argument(
        "--faster-whisper-model",
        default=os.getenv("PATROLLENS_FASTER_WHISPER_MODEL", "large-v3-turbo"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="default")


def _add_ingest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Video file or directory")
    _add_storage_arguments(parser)
    parser.add_argument("--profile", choices=["full", "core", "metadata"], default="core")
    _add_openrouter_transport_arguments(parser)
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=int(os.getenv("PATROLLENS_EMBEDDING_DIMENSIONS", "768")),
    )
    parser.add_argument(
        "--embedding-mode",
        choices=["sync", "batch"],
        default="sync",
        help=(
            "Bulk embedding transport; Batch API is used only when explicitly set to batch"
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=6,
        help="Maximum embedding items submitted together in one local/API batch",
    )
    parser.add_argument(
        "--embedding-batch-poll-s",
        type=float,
        default=float(os.getenv("PATROLLENS_EMBEDDING_BATCH_POLL_SECONDS", "10")),
        help="Seconds between OpenRouter Batch API status checks",
    )
    parser.add_argument(
        "--embedding-batch-timeout-s",
        type=float,
        default=float(os.getenv("PATROLLENS_EMBEDDING_BATCH_TIMEOUT_SECONDS", "86400")),
        help="Maximum wait for each OpenRouter embedding batch",
    )
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--no-embedding-images", action="store_true")
    parser.add_argument("--visual-model", default="google/siglip2-base-patch16-224")
    _add_transcriber_arguments(parser)
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--no-asr", action="store_true")
    _add_clap_arguments(parser)
    parser.add_argument("--window-s", type=float, default=16.0)
    parser.add_argument("--stride-s", type=float, default=8.0)
    parser.add_argument("--frame-fps", type=float, default=1.0)
    parser.add_argument(
        "--clap-window-s",
        type=float,
        default=float(os.getenv("PATROLLENS_CLAP_WINDOW_SECONDS", "10")),
    )
    parser.add_argument(
        "--clap-stride-s",
        type=float,
        default=float(os.getenv("PATROLLENS_CLAP_STRIDE_SECONDS", "5")),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--video-batch-size",
        type=int,
        default=None,
        help="Process only this many pending videos, oldest modification first",
    )
    parser.add_argument(
        "--cost-report",
        help="Cost report path (default: INDEX/reports/ingestion-cost-estimate.json)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Generate the ordered per-video cost report without indexing",
    )
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(func=cmd_ingest)


def _add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    _add_storage_arguments(parser)
    parser.add_argument("--planner", choices=["gemini", "heuristic"], default="gemini")
    parser.add_argument("--planner-model", default=os.getenv("PATROLLENS_GEMINI_PLANNER_MODEL", DEFAULT_GEMINI_MODEL))
    _add_openrouter_transport_arguments(parser)
    parser.add_argument("--visual-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=int(os.getenv("PATROLLENS_EMBEDDING_DIMENSIONS", "768")),
    )
    parser.add_argument("--no-embeddings", action="store_true")
    _add_clap_arguments(parser)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--branch-k", type=int, default=60)
    parser.add_argument("--temporal-tolerance-s", type=float, default=4.0)
    parser.add_argument("--merge-gap-s", type=float, default=3.0)
    parser.add_argument("--candidate-padding-s", type=float, default=5.0)


def _add_clap_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--clap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable larger_clap_general audio indexing/search; full ingestion "
            "enables it by default and retrieval auto-detects installed artifacts"
        ),
    )
    parser.add_argument(
        "--clap-model-root",
        default=os.getenv(
            "PATROLLENS_CLAP_MODEL_ROOT",
            ".patrol-lens-models/larger_clap_general_coreml",
        ),
    )
    parser.add_argument(
        "--clap-model",
        default=os.getenv("PATROLLENS_CLAP_MODEL", DEFAULT_CLAP_MODEL),
        help="CLAP model namespace stored with 512-d vectors",
    )
    parser.add_argument(
        "--clap-compute-units",
        choices=["cpu_only", "cpu_and_gpu", "all"],
        default=os.getenv("PATROLLENS_CLAP_COMPUTE_UNITS", "cpu_only"),
        help="CoreML execution policy; CPU-only is the safe default on macOS 26",
    )


def _add_openrouter_transport_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--openrouter-base-url",
        default=os.getenv("PATROLLENS_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter-compatible API base URL",
    )
    parser.add_argument(
        "--openrouter-http-referer",
        default=os.getenv("PATROLLENS_OPENROUTER_HTTP_REFERER"),
        help="Optional HTTP-Referer attribution header",
    )
    parser.add_argument(
        "--openrouter-title",
        default=os.getenv("PATROLLENS_OPENROUTER_TITLE"),
        help="Optional X-OpenRouter-Title attribution header",
    )


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index",
        default=os.getenv("PATROLLENS_ARTIFACT_ROOT", ".patrol-lens"),
        help="Local artifact directory for SQLite, extracted media, or Postgres runs",
    )
    parser.add_argument(
        "--backend",
        choices=["sqlite", "postgres"],
        default=os.getenv("PATROLLENS_STORE_BACKEND", "sqlite"),
        help="Evidence/index backend",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("PATROLLENS_DATABASE_URL"),
        help="PostgreSQL DSN (or set PATROLLENS_DATABASE_URL)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patrol-lens",
        description="Search body-camera footage with retrieval-guided Gemini active perception",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Build timestamped multimodal evidence indexes")
    _add_ingest_arguments(ingest)
    index = commands.add_parser("index", help="Alias for ingest")
    _add_ingest_arguments(index)

    compress = commands.add_parser(
        "compress",
        help="Create a separate resumable 480p video corpus",
    )
    compress.add_argument("input", help="Source video file or corpus directory")
    compress.add_argument(
        "--output",
        default="compressed_video_corpus",
        help="Destination corpus directory",
    )
    compress.add_argument(
        "--crf",
        type=int,
        default=int(os.getenv("PATROLLENS_VIDEO_COMPRESSION_CRF", "23")),
        help="H.264 quality: lower is higher quality and larger (0-51)",
    )
    compress.add_argument(
        "--preset",
        choices=[
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        ],
        default="veryfast",
    )
    compress.add_argument("--overwrite", action="store_true")
    compress.set_defaults(func=cmd_compress)

    retrieve = commands.add_parser("retrieve", help="Run cheap multimodal candidate retrieval only")
    retrieve.add_argument("query")
    _add_retrieval_arguments(retrieve)
    retrieve.set_defaults(func=cmd_retrieve)

    search = commands.add_parser("search", help="Run retrieval, active perception, verification, and grounding")
    search.add_argument("query")
    _add_retrieval_arguments(search)
    search.add_argument("--model", default=os.getenv("PATROLLENS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    search.add_argument("--max-candidates", type=int, default=12)
    search.add_argument("--max-turns", type=int, default=5)
    search.add_argument(
        "--candidate-parallelism",
        type=int,
        default=None,
        help="Maximum candidates inspected concurrently; omitted keeps sequential search",
    )
    search.add_argument(
        "--early-stop-confidence",
        type=float,
        default=None,
        help="Stop after directly grounded supported evidence reaches this confidence",
    )
    search.add_argument(
        "--search-timeout-s",
        type=float,
        default=None,
        help="Global deadline covering retrieval, inspection, verification, and refinement",
    )
    search.add_argument(
        "--max-run-cost-usd",
        type=float,
        default=None,
        help="Deny or stop candidate inference when this run-cost budget is reached",
    )
    search.add_argument("--coarse-only", action="store_true")
    search.add_argument("--timelens-command")
    search.add_argument("--acknowledge-timelens-license", action="store_true")
    search.add_argument("--timelens-timeout", type=int, default=300)
    search.set_defaults(func=cmd_search)

    history = commands.add_parser(
        "history",
        help="List durable search/retrieval threads or show one trajectory",
    )
    history.add_argument("history_action", nargs="?", choices=["show"])
    history.add_argument("run_id", nargs="?")
    history.add_argument(
        "--index",
        default=os.getenv("PATROLLENS_ARTIFACT_ROOT", ".patrol-lens"),
        help="Artifact directory containing the history folder",
    )
    history.set_defaults(func=cmd_history)

    evaluate = commands.add_parser("evaluate", help="Evaluate coarse retrieval against JSONL intervals")
    evaluate.add_argument("queries")
    _add_retrieval_arguments(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    asr_canary = commands.add_parser(
        "benchmark-asr",
        help="Benchmark the configured OpenRouter transcription model",
    )
    asr_canary.add_argument("input", help="One canary video")
    _add_openrouter_transport_arguments(asr_canary)
    _add_transcriber_arguments(asr_canary)
    asr_canary.add_argument(
        "--reference-transcript",
        help="Optional ground-truth transcript text file for WER comparison",
    )
    asr_canary.set_defaults(func=cmd_benchmark_asr)

    migrate = commands.add_parser(
        "migrate-embeddings",
        help="Re-embed PostgreSQL source evidence into the 768-dimensional index",
    )
    _add_storage_arguments(migrate)
    _add_openrouter_transport_arguments(migrate)
    migrate.add_argument("--embedding-dimensions", type=int, default=768)
    migrate.add_argument("--embedding-batch-size", type=int, default=6)
    migrate.set_defaults(func=cmd_migrate_embeddings)

    doctor = commands.add_parser("doctor", help="Report optional runtime capabilities")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_project_env()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
