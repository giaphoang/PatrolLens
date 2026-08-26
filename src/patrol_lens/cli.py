from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .adapters.asr import FasterWhisperASR
from .adapters.audio import WaveAudioAnalyzer
from .adapters.media import iter_video_files
from .adapters.ocr import PaddleOCRBackend
from .adapters.openrouter import OpenRouterReranker
from .adapters.visual import SigLIP2Encoder
from .evaluate import evaluate_file
from .ingest import Indexer, IngestConfig
from .retrieval import Retriever
from .storage import IndexStore
from .text import HashEmbeddingEncoder


def _encoder(name: str):
    if name == "hash":
        return HashEmbeddingEncoder()
    if name == "siglip2":
        return SigLIP2Encoder()
    raise ValueError(f"Unknown encoder: {name}")


def cmd_index(args: argparse.Namespace) -> None:
    store = IndexStore(args.index)
    text_encoder = _encoder(args.text_encoder)
    visual = SigLIP2Encoder(model_name=args.visual_model) if args.visual == "siglip2" else None
    asr = FasterWhisperASR(args.asr_model, device=args.device) if args.asr == "faster-whisper" else None
    audio = WaveAudioAnalyzer() if args.audio == "wave" else None
    ocr = PaddleOCRBackend() if args.ocr == "paddle" else None
    remote = None
    if args.remote_annotate:
        remote = OpenRouterReranker(args.openrouter_model or os.environ.get("OPENROUTER_VLM_MODEL", ""), text_encoder=text_encoder)
    indexer = Indexer(store, config=IngestConfig(enable_remote_annotations=args.remote_annotate), asr=asr, audio=audio, visual=visual, ocr=ocr, text_encoder=text_encoder, remote_annotator=remote)
    stats = []
    for video in iter_video_files(args.input):
        stats.append(indexer.index_path(video))
    print(json.dumps({"index": str(Path(args.index).resolve()), "videos": stats}, indent=2))
    store.close()


def cmd_search(args: argparse.Namespace) -> None:
    store = IndexStore(args.index)
    encoder = _encoder(args.encoder)
    reranker = OpenRouterReranker(args.openrouter_model or "") if args.rerank else None
    retriever = Retriever(store, text_encoder=encoder, visual_encoder=encoder, clip_encoder=encoder, reranker=reranker)
    output = retriever.search_json(args.query, top_k=args.top_k, retrieve_k=args.retrieve_k, max_rerank=args.max_rerank)
    print(json.dumps(output, indent=2))
    store.close()


def cmd_evaluate(args: argparse.Namespace) -> None:
    store = IndexStore(args.index)
    encoder = _encoder(args.encoder)
    retriever = Retriever(store, text_encoder=encoder, visual_encoder=encoder)
    print(json.dumps(evaluate_file(args.queries, retriever, args.top_k), indent=2))
    store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patrol-lens", description="Search body-camera footage with natural-language queries")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Build local modality indexes")
    index.add_argument("input", help="Video file or directory")
    index.add_argument("--index", default=".patrol-lens")
    index.add_argument("--text-encoder", choices=["hash", "siglip2"], default="hash")
    index.add_argument("--visual", choices=["none", "siglip2"], default="none")
    index.add_argument("--visual-model", default="google/siglip2-base-patch16-224")
    index.add_argument("--asr", choices=["none", "faster-whisper"], default="none")
    index.add_argument("--asr-model", default="small.en")
    index.add_argument("--device", default="cpu")
    index.add_argument("--audio", choices=["none", "wave"], default="none")
    index.add_argument("--ocr", choices=["none", "paddle"], default="none")
    index.add_argument("--remote-annotate", action="store_true")
    index.add_argument("--openrouter-model", default=None)
    index.set_defaults(func=cmd_index)

    search = subparsers.add_parser("search", help="Search an existing index")
    search.add_argument("query")
    search.add_argument("--index", default=".patrol-lens")
    search.add_argument("--encoder", choices=["hash", "siglip2"], default="hash")
    search.add_argument("--top-k", type=int, default=20)
    search.add_argument("--retrieve-k", type=int, default=100)
    search.add_argument("--max-rerank", type=int, default=20)
    search.add_argument("--rerank", action="store_true")
    search.add_argument("--openrouter-model", default=os.environ.get("OPENROUTER_VLM_MODEL"))
    search.set_defaults(func=cmd_search)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate against JSONL interval labels")
    evaluate.add_argument("queries")
    evaluate.add_argument("--index", default=".patrol-lens")
    evaluate.add_argument("--encoder", choices=["hash", "siglip2"], default="hash")
    evaluate.add_argument("--top-k", type=int, default=10)
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
