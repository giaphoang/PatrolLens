from __future__ import annotations

import csv
import math
import struct
import wave
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Protocol


@dataclass(frozen=True)
class AudioAnalysis:
    rms_db: float
    speech_activity: float
    pitch_hz: float | None
    labels: list[str] = field(default_factory=list)
    confidence: float = 0.0
    event_scores: dict[str, float] = field(default_factory=dict)

    @property
    def content(self) -> str:
        features = [f"rms_db={self.rms_db:.1f}", f"speech_activity={self.speech_activity:.2f}"]
        if self.pitch_hz:
            features.append(f"pitch_hz={self.pitch_hz:.1f}")
        return "; ".join([*self.labels, *features])


class AudioBackend(Protocol):
    model_name: str

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis: ...


class NullAudio:
    model_name = "none"

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis:
        return AudioAnalysis(-80.0, 0.0, None)


def _wave_segment(audio_path: str, start_ms: int, end_ms: int) -> tuple[list[int], int]:
    try:
        with wave.open(audio_path, "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            if sample_width != 2:
                return [], sample_rate
            start_frame = max(0, int(start_ms / 1000 * sample_rate))
            frame_count = max(1, int((end_ms - start_ms) / 1000 * sample_rate))
            handle.setpos(min(start_frame, handle.getnframes()))
            raw = handle.readframes(frame_count)
    except (OSError, wave.Error):
        return [], 16_000
    if not raw:
        return [], sample_rate
    samples = list(struct.unpack("<" + "h" * (len(raw) // 2), raw))
    return (samples[::channels] if channels > 1 else samples), sample_rate


def _pitch_from_crossings(samples: list[int], sample_rate: int) -> float | None:
    """Cheap voiced-pitch cue; Gemini remains responsible for semantic prosody."""

    if len(samples) < sample_rate // 10:
        return None
    mean = sum(samples) / len(samples)
    signs = [sample >= mean for sample in samples]
    crossings = sum(left != right for left, right in pairwise(signs))
    estimate = crossings * sample_rate / (2 * len(samples))
    return estimate if 60 <= estimate <= 500 else None


class WaveAudioAnalyzer:
    """Dependency-free RMS, crude pitch, and activity cues for prosody retrieval."""

    model_name = "wave-rms-pitch-v2"

    def __init__(self, raised_db: float = -24.0) -> None:
        self.raised_db = raised_db

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis:
        samples, sample_rate = _wave_segment(audio_path, start_ms, end_ms)
        if not samples:
            return AudioAnalysis(-80.0, 0.0, None)
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768
        rms_db = 20 * math.log10(max(rms, 1e-6))
        activity = min(1.0, max(0.0, (rms_db + 55) / 35))
        labels: list[str] = []
        confidence = 0.0
        if rms_db >= self.raised_db and activity >= 0.25:
            labels.append("elevated vocal intensity")
            confidence = min(1.0, max(0.0, (rms_db - self.raised_db) / 18 + 0.5))
        return AudioAnalysis(
            rms_db=rms_db,
            speech_activity=activity,
            pitch_hz=_pitch_from_crossings(samples, sample_rate),
            labels=labels,
            confidence=confidence,
        )


class SileroVADAnalyzer:
    """Optional Silero speech-presence adapter, loaded only when selected."""

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
                raise RuntimeError("silero-vad is not installed; install patrol-lens[audio]") from exc
            self._model = load_silero_vad()
            self._read_audio = read_audio
            self._get_speech_timestamps = get_speech_timestamps
        return self._model

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis:
        return self.analyze_many(audio_path, [(start_ms, end_ms)])[0]

    def analyze_many(self, audio_path: str, intervals: list[tuple[int, int]]) -> list[AudioAnalysis]:
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
            results.append(AudioAnalysis(-80.0, ratio, None, ["speech"] if ratio else [], ratio))
        return results


class YAMNetAnalyzer:
    """Optional AudioSet event classifier using Google's YAMNet TF Hub model."""

    model_name = "yamnet-1"

    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n
        self._model = None
        self._cached_path: str | None = None
        self._cached_waveform = None

    def _load(self):
        if self._model is None:
            try:
                import tensorflow as tf
                import tensorflow_hub as hub
            except ImportError as exc:
                raise RuntimeError("tensorflow and tensorflow-hub are required for YAMNet") from exc
            self._tf = tf
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            class_map_path = self._model.class_map_path().numpy().decode()
            with tf.io.gfile.GFile(class_map_path) as handle:
                self._class_names = [row["display_name"] for row in csv.DictReader(handle)]
        return self._model

    def _waveform(self, audio_path: str):
        self._load()
        if self._cached_path != audio_path:
            audio_bytes = self._tf.io.read_file(audio_path)
            waveform, sample_rate = self._tf.audio.decode_wav(audio_bytes, desired_channels=1)
            waveform = self._tf.squeeze(waveform, axis=-1)
            if int(sample_rate) != 16_000:
                raise RuntimeError("YAMNet adapter expects the ingestion WAV to be 16 kHz")
            self._cached_path = audio_path
            self._cached_waveform = waveform
        return self._cached_waveform

    def _analyze_waveform(self, waveform, start_ms: int, end_ms: int) -> AudioAnalysis:
        model = self._load()
        start = max(0, round(start_ms * 16))
        end = max(start + 1, round(end_ms * 16))
        scores, _embeddings, _spectrogram = model(waveform[start:end])
        mean_scores = self._tf.reduce_mean(scores, axis=0).numpy()
        positions = mean_scores.argsort()[-self.top_n:][::-1]
        event_scores = {self._class_names[int(pos)]: float(mean_scores[pos]) for pos in positions}
        labels = [label for label, score in event_scores.items() if score >= 0.1]
        confidence = max(event_scores.values(), default=0.0)
        return AudioAnalysis(-80.0, 0.0, None, labels, confidence, event_scores)

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis:
        return self._analyze_waveform(self._waveform(audio_path), start_ms, end_ms)

    def analyze_many(self, audio_path: str, intervals: list[tuple[int, int]]) -> list[AudioAnalysis]:
        waveform = self._waveform(audio_path)
        return [self._analyze_waveform(waveform, start, end) for start, end in intervals]


class CompositeAudioAnalyzer:
    """Merge cheap prosody, Silero VAD, and YAMNet observations behind one interface."""

    def __init__(self, analyzers: list[AudioBackend]) -> None:
        if not analyzers:
            raise ValueError("at least one audio analyzer is required")
        self.analyzers = analyzers
        self.model_name = "+".join(item.model_name for item in analyzers)

    @staticmethod
    def _merge(parts: list[AudioAnalysis]) -> AudioAnalysis:
        labels = list(dict.fromkeys(label for item in parts for label in item.labels))
        event_scores = {key: value for item in parts for key, value in item.event_scores.items()}
        rms = next((item.rms_db for item in parts if item.rms_db > -80), -80.0)
        pitch = next((item.pitch_hz for item in parts if item.pitch_hz), None)
        return AudioAnalysis(
            rms_db=rms,
            speech_activity=max((item.speech_activity for item in parts), default=0.0),
            pitch_hz=pitch,
            labels=labels,
            confidence=max((item.confidence for item in parts), default=0.0),
            event_scores=event_scores,
        )

    def analyze(self, audio_path: str, start_ms: int, end_ms: int) -> AudioAnalysis:
        return self._merge([item.analyze(audio_path, start_ms, end_ms) for item in self.analyzers])

    def analyze_many(self, audio_path: str, intervals: list[tuple[int, int]]) -> list[AudioAnalysis]:
        per_analyzer: list[list[AudioAnalysis]] = []
        for analyzer in self.analyzers:
            method = getattr(analyzer, "analyze_many", None)
            if callable(method):
                per_analyzer.append(method(audio_path, intervals))
            else:
                per_analyzer.append(
                    [analyzer.analyze(audio_path, start, end) for start, end in intervals]
                )
        return [self._merge(list(parts)) for parts in zip(*per_analyzer)]
