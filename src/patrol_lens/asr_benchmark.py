from __future__ import annotations

import hashlib
import re
import time
from difflib import SequenceMatcher
from statistics import median
from typing import Any

from .adapters.asr import ASRBackend, WordSpan


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = _words(reference)
    actual = _words(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, reference_word in enumerate(expected, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def transcript_text(spans: list[WordSpan]) -> str:
    return " ".join(item.text.strip() for item in spans if item.text.strip())


def _timestamp_drift(
    baseline: list[WordSpan],
    candidate: list[WordSpan],
) -> dict[str, Any]:
    available = set(range(len(candidate)))
    drifts: list[float] = []
    for expected in baseline:
        matches = [
            (
                SequenceMatcher(None, expected.text.lower(), candidate[index].text.lower()).ratio(),
                index,
            )
            for index in available
        ]
        if not matches:
            break
        similarity, index = max(matches)
        if similarity < 0.35:
            continue
        available.remove(index)
        actual = candidate[index]
        drifts.extend(
            (
                abs(actual.start_ms - expected.start_ms) / 1000,
                abs(actual.end_ms - expected.end_ms) / 1000,
            )
        )
    return {
        "matched_boundaries": len(drifts),
        "median_seconds": round(median(drifts), 3) if drifts else None,
        "max_seconds": round(max(drifts), 3) if drifts else None,
    }


def benchmark_backend(
    backend: ASRBackend,
    audio_path: str,
    *,
    duration_seconds: float,
) -> tuple[list[WordSpan], dict[str, Any]]:
    started = time.perf_counter()
    spans = backend.transcribe(audio_path)
    elapsed = time.perf_counter() - started
    transcript = transcript_text(spans)
    result: dict[str, Any] = {
        "backend": backend.model_name,
        "wall_seconds": round(elapsed, 3),
        "audio_seconds": round(duration_seconds, 3),
        "real_time_factor": round(elapsed / max(duration_seconds, 0.001), 4),
        "segments": len(spans),
        "characters": len(transcript),
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "preview": transcript[:500],
    }
    runtime_info = getattr(backend, "last_runtime_info", None)
    if runtime_info:
        result["runtime"] = runtime_info
    return spans, result


def compare_asr_results(
    baseline_spans: list[WordSpan],
    candidate_spans: list[WordSpan],
    *,
    reference_text: str | None = None,
) -> dict[str, Any]:
    baseline_text = transcript_text(baseline_spans)
    candidate_text = transcript_text(candidate_spans)
    comparison: dict[str, Any] = {
        "cross_backend_word_error_rate": round(
            word_error_rate(baseline_text, candidate_text),
            4,
        ),
        "timestamp_drift_vs_baseline": _timestamp_drift(
            baseline_spans,
            candidate_spans,
        ),
    }
    if reference_text is not None:
        comparison["reference_word_error_rate"] = {
            "baseline": round(word_error_rate(reference_text, baseline_text), 4),
            "candidate": round(word_error_rate(reference_text, candidate_text), 4),
        }
    else:
        comparison["quality_note"] = (
            "No reference transcript supplied; cross-backend WER measures agreement, "
            "not ground-truth accuracy."
        )
    return comparison
