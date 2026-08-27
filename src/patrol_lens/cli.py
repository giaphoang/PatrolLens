from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import sys
from dataclasses import replace

from .adapters.asr import FasterWhisperASR
from .adapters.audio import (
    CompositeAudioAnalyzer,
    SileroVADAnalyzer,
    WaveAudioAnalyzer,
    YAMNetAnalyzer,
)
from .adapters.media import iter_video_files
from .adapters.ocr import PaddleOCRBackend
from .adapters.openrouter import OpenRouterJSONClient
from .adapters.visual import SigLIP2Encoder
from .agent import ActivePerceptionAgent, GeminiActivePolicy
from .config import DEFAULT_GEMINI_MODEL, AgentConfig, IngestionConfig, RetrievalConfig
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


def _ingestion_backends(args: argparse.Namespace) -> IngestionBackends:
    if args.profile == "metadata":
        return IngestionBackends()
    visual = None if args.no_visual else SigLIP2Encoder(args.visual_model, device=args.device)
    asr = None if args.no_asr else FasterWhisperASR(
        args.asr_model,
        device=args.device,
        compute_type=args.compute_type,
    )
    ocr = None if args.no_ocr else PaddleOCRBackend(language=args.ocr_language)
    audio = None
    if not args.no_audio:
        analyzers = [WaveAudioAnalyzer(raised_db=args.raised_voice_db)]
        use_silero = args.silero_vad if args.silero_vad is not None else args.profile == "full"
        use_yamnet = args.yamnet if args.yamnet is not None else args.profile == "full"
        if use_silero:
            analyzers.append(SileroVADAnalyzer())
        if use_yamnet:
            analyzers.append(YAMNetAnalyzer())
        audio = CompositeAudioAnalyzer(analyzers)
    return IngestionBackends(visual=visual, asr=asr, ocr=ocr, audio=audio)


def cmd_ingest(args: argparse.Namespace) -> None:
    if min(args.frame_fps, args.window_s, args.stride_s, args.audio_window_s, args.audio_stride_s) <= 0:
        raise ValueError("frame rate, windows, and strides must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    with _open_store(args) as store:
        config = IngestionConfig(
            window_ms=round(args.window_s * 1000),
            stride_ms=round(args.stride_s * 1000),
            frame_step_ms=max(1, round(1000 / args.frame_fps)),
            audio_window_ms=round(args.audio_window_s * 1000),
            audio_stride_ms=round(args.audio_stride_s * 1000),
            batch_size=args.batch_size,
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


def _build_retriever(
    args: argparse.Namespace,
    store: Store,
    *,
    client: OpenRouterJSONClient | None = None,
) -> CoarseRetriever:
    if args.planner == "gemini":
        client = client or _gemini(args.planner_model, args)
        planner = GeminiQueryPlanner(client, model=args.planner_model)
    else:
        planner = HeuristicQueryPlanner()
    visual = None if args.no_visual else SigLIP2Encoder(args.visual_model, device=args.device)
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
        "PaddleOCR": "paddleocr",
        "SigLIP2/transformers": "transformers",
        "Silero VAD": "silero_vad",
        "YAMNet/TensorFlow Hub": "tensorflow_hub",
    }
    def installed(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            return False

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
            "python_modules": {name: installed(module) for name, module in modules.items()},
        }
    )


def _add_ingest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Video file or directory")
    _add_storage_arguments(parser)
    parser.add_argument("--profile", choices=["full", "core", "metadata"], default="core")
    parser.add_argument("--visual-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--ocr-language", default="en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--silero-vad", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--yamnet", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--raised-voice-db", type=float, default=-24.0)
    parser.add_argument("--window-s", type=float, default=16.0)
    parser.add_argument("--stride-s", type=float, default=8.0)
    parser.add_argument("--frame-fps", type=float, default=1.0)
    parser.add_argument("--audio-window-s", type=float, default=4.0)
    parser.add_argument("--audio-stride-s", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(func=cmd_ingest)


def _add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    _add_storage_arguments(parser)
    parser.add_argument("--planner", choices=["gemini", "heuristic"], default="gemini")
    parser.add_argument("--planner-model", default=os.getenv("PATROLLENS_GEMINI_PLANNER_MODEL", DEFAULT_GEMINI_MODEL))
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
    parser.add_argument("--visual-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--branch-k", type=int, default=60)
    parser.add_argument("--temporal-tolerance-s", type=float, default=4.0)
    parser.add_argument("--merge-gap-s", type=float, default=3.0)
    parser.add_argument("--candidate-padding-s", type=float, default=5.0)


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

    retrieve = commands.add_parser("retrieve", help="Run cheap multimodal candidate retrieval only")
    retrieve.add_argument("query")
    _add_retrieval_arguments(retrieve)
    retrieve.set_defaults(func=cmd_retrieve)

    search = commands.add_parser("search", help="Run retrieval, active perception, verification, and grounding")
    search.add_argument("query")
    _add_retrieval_arguments(search)
    search.add_argument("--model", default=os.getenv("PATROLLENS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    search.add_argument("--max-candidates", type=int, default=12)
    search.add_argument("--max-turns", type=int, default=6)
    search.add_argument("--coarse-only", action="store_true")
    search.add_argument("--timelens-command")
    search.add_argument("--acknowledge-timelens-license", action="store_true")
    search.add_argument("--timelens-timeout", type=int, default=300)
    search.set_defaults(func=cmd_search)

    evaluate = commands.add_parser("evaluate", help="Evaluate coarse retrieval against JSONL intervals")
    evaluate.add_argument("queries")
    _add_retrieval_arguments(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    doctor = commands.add_parser("doctor", help="Report optional runtime capabilities")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
