from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import patrol_lens.temporal.timelens2_adapter as adapter_module
from patrol_lens.cli import _embedding, _load_project_env, build_parser
from patrol_lens.domain import VerificationResult
from patrol_lens.temporal import TimeLens2Adapter
from patrol_lens.temporal.timelens2_adapter import should_use_timelens2


def test_cli_exposes_new_lifecycle():
    parser = build_parser()

    assert parser.parse_args(["ingest", "videos"]).command == "ingest"
    assert parser.parse_args(["retrieve", "red shirt"]).command == "retrieve"
    search = parser.parse_args(["search", "Miranda rights"])
    assert search.model == "google/gemini-3.1-pro-preview"
    assert search.openrouter_base_url == "https://openrouter.ai/api/v1"
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
