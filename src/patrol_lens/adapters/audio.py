from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class AudioAnalysis:
    """Speech-presence evidence emitted by optional audio adapters."""

    speech_activity: float
    labels: list[str] = field(default_factory=list)
    confidence: float = 0.0
    event_scores: dict[str, float] = field(default_factory=dict)

    @property
    def content(self) -> str:
        features = [*self.labels, f"speech_activity={self.speech_activity:.2f}"]
        return "; ".join(features)


class AudioBackend(Protocol):
    model_name: str

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis: ...


class NullAudio:
    model_name = "none"

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis:
        return AudioAnalysis(0.0)


class SileroVADAnalyzer:
    """Optional speech-presence adapter used by the full ingestion profile."""

    model_name = "silero-vad"

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from silero_vad import (
                    get_speech_timestamps,
                    load_silero_vad,
                    read_audio,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "silero-vad is not installed; install patrol-lens[audio]"
                ) from exc
            self._model = load_silero_vad()
            self._read_audio = read_audio
            self._get_speech_timestamps = get_speech_timestamps
        return self._model

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis:
        return self.analyze_many(audio_path, [(start_ms, end_ms)])[0]

    def analyze_many(
        self,
        audio_path: str,
        intervals: list[tuple[int, int]],
    ) -> list[AudioAnalysis]:
        model = self._load()
        waveform = self._read_audio(audio_path, sampling_rate=16_000)
        spans = self._get_speech_timestamps(waveform, model, sampling_rate=16_000)
        results: list[AudioAnalysis] = []
        for start_ms, end_ms in intervals:
            start = max(0, round(start_ms * 16))
            end = max(start + 1, round(end_ms * 16))
            speech_samples = sum(
                max(0, min(end, int(item["end"])) - max(start, int(item["start"])))
                for item in spans
            )
            ratio = min(1.0, speech_samples / max(1, end - start))
            results.append(
                AudioAnalysis(
                    ratio,
                    ["speech"] if ratio else [],
                    ratio,
                )
            )
        return results
