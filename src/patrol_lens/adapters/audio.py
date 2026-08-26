from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AudioAnalysis:
    rms_db: float
    speech_activity: float
    label: str | None
    confidence: float


class AudioBackend(Protocol):
    model_name: str

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis: ...


class NullAudio:
    model_name = "none"

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis:
        return AudioAnalysis(-80.0, 0.0, None, 0.0)


class WaveAudioAnalyzer:
    """Dependency-free loudness/VAD baseline for raised-voice retrieval."""

    model_name = "wave-rms-vad-1"

    def __init__(self, raised_db: float = -24.0) -> None:
        self.raised_db = raised_db

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis:
        try:
            with wave.open(audio_path, "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                start_frame = int(start_ms / 1000 * sample_rate)
                frame_count = max(1, int((end_ms - start_ms) / 1000 * sample_rate))
                handle.setpos(min(start_frame, handle.getnframes()))
                raw = handle.readframes(frame_count)
        except (OSError, wave.Error):
            return AudioAnalysis(-80.0, 0.0, None, 0.0)
        if not raw or sample_width != 2:
            return AudioAnalysis(-80.0, 0.0, None, 0.0)
        samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)
        if channels > 1:
            samples = samples[::channels]
        rms = math.sqrt(sum(sample * sample for sample in samples) / max(1, len(samples))) / 32768
        rms_db = 20 * math.log10(max(rms, 1e-6))
        activity = min(1.0, max(0.0, (rms_db + 55) / 35))
        confidence = min(1.0, max(0.0, (rms_db - self.raised_db) / 18 + 0.5)) if activity else 0.0
        label = "elevated vocal intensity" if rms_db >= self.raised_db and activity >= 0.25 else None
        return AudioAnalysis(rms_db, activity, label, confidence if label else 0.0)
