from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

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
)
from .evaluate import evaluate_file
from .index import AutoVectorIndex, IndexStore, PostgresIndexStore, PostgresVectorIndex
from .ingestion import IngestionBackends, IngestionPipeline
from .pipeline import SearchPipeline
from .retrieval import CoarseRetriever, GeminiQueryPlanner, HeuristicQueryPlanner
from .temporal import LightweightTimestampRefiner, TimeLens2Adapter
from .verification import GeminiEventVerifier


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


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


def _gemini(model: str, args: argparse.Namespace) -> OpenRouterJSONClient:
    return OpenRouterJSONClient(
        model=model,
        base_url=args.openrouter_base_url,
        http_referer=args.openrouter_http_referer,
        title=args.openrouter_title,
    )


def _embedding(args: argparse.Namespace) -> OpenRouterEmbeddingClient:
    model = os.getenv("PATROLLENS_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL)
    query_model = os.getenv("PATROLLENS_EMBEDDING_QUERY_MODEL", model)
    return OpenRouterEmbeddingClient(
        model=model,
        # Ingestion uses the synchronous embeddings endpoint. The client
        # defaults the document/media route to the same canonical model.
        query_model=query_model,
        dimensions=args.embedding_dimensions,
        base_url=args.openrouter_base_url,
        http_referer=args.openrouter_http_referer,
        title=args.openrouter_title,
        media_batch_size=getattr(args, "embedding_batch_size", 6),
    )


def _embedding_canary(embedding: OpenRouterEmbeddingClient) -> None:
    """Verify the provider's output size before ingestion can write evidence."""

    vector = embedding.encode_text("PatrolLens embedding dimension canary")
    if len(vector) != embedding.dimensions:
        raise EmbeddingDimensionError(embedding.dimensions, len(vector))


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


def _ingestion_backends(args: argparse.Namespace) -> IngestionBackends:
    if args.profile == "metadata":
        return IngestionBackends()
    use_clap = args.clap if args.clap is not None else args.profile == "full"
    audio_embedding = _clap_backend(args, required=True) if use_clap else None
    asr = None if args.no_asr else _asr_backend(args)
    embedding = None if args.no_embeddings else _embedding(args)
    if embedding is not None:
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
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive")
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
        pipeline = IngestionPipeline(
            store,
            backends=_ingestion_backends(args),
            config=config,
            vector_index=_vector_index(store),
        )
        videos = list(iter_video_files(args.input))
        if not videos:
            raise RuntimeError(f"no supported video files found under {args.input}")
        stats = [pipeline.ingest_path(path, force=args.force) for path in videos]
        _print({"index": str(store.root), "videos": stats})


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
) -> CoarseRetriever:
    if isinstance(store, PostgresIndexStore) and args.embedding_dimensions != 768:
        raise ValueError("PostgreSQL retrieval requires --embedding-dimensions 768")
    if args.planner == "gemini":
        client = client or _gemini(args.planner_model, args)
        planner = GeminiQueryPlanner(client, model=args.planner_model)
    else:
        planner = HeuristicQueryPlanner()
    embedding = None if args.no_embeddings else _embedding(args)
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
    )


def cmd_retrieve(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        retriever = _build_retriever(args, store)
        _print(retriever.search_json(args.query))


def cmd_search(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        client = None if args.coarse_only and args.planner == "heuristic" else _gemini(args.model, args)
        retriever = _build_retriever(args, store, client=client)
        if args.coarse_only:
            _print(retriever.search_json(args.query))
            return
        base_agent_config = AgentConfig.from_env(model=args.model, planner_model=args.planner_model)
        agent_config = replace(
            base_agent_config,
            max_turns=args.max_turns,
            run_root=str(store.root / "runs"),
        )
        assert client is not None
        agent = ActivePerceptionAgent(
            GeminiActivePolicy(client, model=args.model),
            config=agent_config,
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
        )
        _print(pipeline.search(args.query, max_candidates=args.max_candidates).to_dict())


def cmd_evaluate(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        args.planner = "heuristic"
        retriever = _build_retriever(args, store)
        _print(evaluate_file(args.queries, retriever, args.top_k))


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
    parser.add_argument("--embedding-batch-size", type=int, default=6)
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
    search.add_argument("--coarse-only", action="store_true")
    search.add_argument("--timelens-command")
    search.add_argument("--acknowledge-timelens-license", action="store_true")
    search.add_argument("--timelens-timeout", type=int, default=300)
    search.set_defaults(func=cmd_search)

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
