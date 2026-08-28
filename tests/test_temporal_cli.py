from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import patrol_lens.temporal.timelens2_adapter as adapter_module
from patrol_lens.adapters.asr import FasterWhisperASR, OpenRouterASR
from patrol_lens.cli import (
    _asr_backend,
    _embedding,
    _ingestion_backends,
    _load_project_env,
    build_parser,
)
from patrol_lens.domain import VerificationResult
from patrol_lens.temporal import TimeLens2Adapter
from patrol_lens.temporal.timelens2_adapter import should_use_timelens2


def test_cli_exposes_new_lifecycle():
    parser = build_parser()

    ingest = parser.parse_args(["ingest", "videos"])
    assert ingest.command == "ingest"
    assert ingest.video_batch_size is None
    assert ingest.cost_report is None
    assert ingest.estimate_only is False
    batched_ingest = parser.parse_args(
        ["ingest", "videos", "--video-batch-size", "3", "--estimate-only"]
    )
    assert batched_ingest.video_batch_size == 3
    assert batched_ingest.estimate_only is True
    compress = parser.parse_args(["compress", "videos_corpus"])
    assert compress.output == "compressed_video_corpus"
    assert parser.parse_args(["retrieve", "red shirt"]).command == "retrieve"
    search = parser.parse_args(["search", "Miranda rights"])
    assert search.model == "google/gemini-3.1-pro-preview"
    assert search.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert search.candidate_parallelism is None
    assert search.early_stop_confidence is None
    assert search.search_timeout_s is None
    bounded = parser.parse_args(
        [
            "search", "crying", "--candidate-parallelism", "4",
            "--early-stop-confidence", "0.90", "--search-timeout-s", "300",
        ]
    )
    assert bounded.candidate_parallelism == 4
    assert bounded.early_stop_confidence == 0.9
    assert bounded.search_timeout_s == 300
    assert parser.parse_args(["doctor"]).command == "doctor"


def test_ingestion_embedding_uses_base_environment_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("PATROLLENS_EMBEDDING_MODEL", "google/gemini-embedding-2")
    monkeypatch.setenv("PATROLLENS_EMBEDDING_BATCH_MODEL", "google/gemini-embedding-2:batch")
    monkeypatch.setenv("PATROLLENS_EMBEDDING_QUERY_MODEL", "google/gemini-embedding-2")

    args = build_parser().parse_args(["ingest", "videos"])
    embedding = _embedding(args)

    assert embedding.model_name == "google/gemini-embedding-2"
    assert embedding.batch_model == "google/gemini-embedding-2"
    assert embedding.query_model == "google/gemini-embedding-2"
    assert not hasattr(args, "embedding_batch_model")


def test_offline_ingestion_defaults_to_openrouter_asr(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    args = build_parser().parse_args(["ingest", "videos"])

    assert args.transcriber == "auto"
    backend = _asr_backend(args)
    assert isinstance(backend, OpenRouterASR)
    assert backend.model_name == "openai/whisper-large-v3-turbo"


def test_ingestion_profiles_exclude_removed_auxiliary_stacks():
    parser = build_parser()
    common = ["ingest", "videos", "--no-embeddings", "--no-asr", "--no-visual"]

    core_args = parser.parse_args([*common, "--profile", "core"])
    core = _ingestion_backends(core_args)
    full = _ingestion_backends(
        parser.parse_args([*common, "--profile", "full", "--no-clap"])
    )

    assert core.ocr is None
    assert full.ocr is None
    assert not hasattr(core, "audio")
    assert not hasattr(core_args, "audio_window_s")
    assert not hasattr(core_args, "audio_stride_s")
    assert not hasattr(core_args, "ocr_language")
    assert not hasattr(core_args, "yamnet")
    assert not hasattr(core_args, "raised_voice_db")


def test_faster_whisper_remains_an_explicit_ingestion_fallback():
    args = build_parser().parse_args(
        ["ingest", "videos", "--transcriber", "faster_whisper"]
    )

    assert isinstance(_asr_backend(args), FasterWhisperASR)


def test_cli_dotenv_overrides_stale_shell_configuration(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=dotenv-key\n"
        "PATROLLENS_EMBEDDING_DIMENSIONS=768\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "stale-shell-key")
    monkeypatch.setenv("PATROLLENS_EMBEDDING_DIMENSIONS", "3072")

    _load_project_env()

    assert os.environ["OPENROUTER_API_KEY"] == "dotenv-key"
    assert os.environ["PATROLLENS_EMBEDDING_DIMENSIONS"] == "768"


def test_timelens_requires_explicit_license_acknowledgement():
    with pytest.raises(RuntimeError, match="academic-only"):
        TimeLens2Adapter(["timelens2-wrapper"])


def test_timelens_adapter_clamps_external_intervals(monkeypatch):
    payload = json.dumps({"intervals_ms": [[900, 2200, 0.9], [4000, 9000]]})
    monkeypatch.setattr(
        adapter_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=payload, stderr=""),
    )
    adapter = TimeLens2Adapter(["wrapper"], acknowledge_restricted_license=True)

    intervals = adapter.ground("video.mp4", "query", 1000, 5000)

    assert (intervals[0].start_ms, intervals[0].end_ms) == (1000, 2200)
    assert (intervals[1].start_ms, intervals[1].end_ms) == (4000, 5000)


def test_timelens_activation_is_quality_triggered():
    broad = VerificationResult("supported", "event", 0, 30_000, 0.9)
    precise = VerificationResult("supported", "event", 10_000, 14_000, 0.9)

    assert should_use_timelens2(broad)
    assert not should_use_timelens2(precise)
