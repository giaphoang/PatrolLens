from __future__ import annotations

import wave

import pytest

from patrol_lens.adapters.asr import OpenRouterASR, WordSpan
from patrol_lens.asr_benchmark import compare_asr_results, word_error_rate


def _write_wav(path, *, seconds: float, sample_rate: int = 100) -> None:
    frame_count = round(seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count)


def test_openrouter_asr_chunks_offsets_and_checkpoints(monkeypatch, tmp_path):
    audio = tmp_path / "media" / "audio" / "video.wav"
    _write_wav(audio, seconds=2.5)
    calls: list[dict] = []
    backend = OpenRouterASR(
        api_key="test-key",
        chunk_seconds=1,
        http_referer="https://patrol-lens.test",
        title="PatrolLens",
    )

    def fake_post(payload):
        calls.append(payload)
        ordinal = len(calls)
        end = 0.4 if ordinal == 3 else 0.9
        return {
            "text": f"chunk {ordinal}",
            "segments": [
                {
                    "start": 0.1,
                    "end": end,
                    "text": f" chunk {ordinal} ",
                    "avg_logprob": -0.1,
                }
            ],
            "usage": {"cost": 0.001},
        }

    monkeypatch.setattr(backend, "_post_json", fake_post)

    first = backend.transcribe(str(audio))
    first_runtime = dict(backend.last_runtime_info)
    second = backend.transcribe(str(audio))

    assert [(item.start_ms, item.end_ms, item.text) for item in first] == [
        (100, 900, "chunk 1"),
        (1100, 1900, "chunk 2"),
        (2100, 2400, "chunk 3"),
    ]
    assert second == first
    assert len(calls) == 3
    assert calls[0]["model"] == "openai/whisper-large-v3-turbo"
    assert calls[0]["response_format"] == "verbose_json"
    assert calls[0]["input_audio"]["format"] == "wav"
    assert "language" not in calls[0]
    assert first_runtime["api_calls"] == 3
    assert first_runtime["cost_source"] == "provider"
    assert first_runtime["latency_ms"] >= 0
    assert backend.last_runtime_info["api_calls"] == 0
    assert backend.last_runtime_info["cache_hits"] == 3
    assert backend.last_runtime_info["reported_cost_usd"] == 0
    assert backend.last_runtime_info["cost_source"] == "cache_only"


def test_openrouter_asr_rejects_response_without_segment_timestamps(
    monkeypatch,
    tmp_path,
):
    audio = tmp_path / "media" / "audio" / "video.wav"
    _write_wav(audio, seconds=0.5)
    backend = OpenRouterASR(api_key="test-key", chunk_seconds=1)
    monkeypatch.setattr(
        backend,
        "_post_json",
        lambda _payload: {"text": "untimestamped transcript"},
    )

    with pytest.raises(RuntimeError, match="without segment timestamps"):
        backend.transcribe(str(audio))


def test_openrouter_asr_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterASR()


def test_asr_canary_metrics_compare_speed_quality_and_drift():
    baseline = [WordSpan(0, 1000, "right to remain silent")]
    candidate = [WordSpan(100, 1200, "right to remain silent")]

    comparison = compare_asr_results(baseline, candidate)

    assert word_error_rate("a b", "a c") == 0.5
    assert comparison["cross_backend_word_error_rate"] == 0.0
    assert comparison["timestamp_drift_vs_baseline"]["median_seconds"] == 0.15
