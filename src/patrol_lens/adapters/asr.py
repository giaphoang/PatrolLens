from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WordSpan:
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None


class ASRBackend(Protocol):
    model_name: str

    def transcribe(self, audio_path: str) -> list[WordSpan]: ...


class NullASR:
    model_name = "none"

    def transcribe(self, audio_path: str) -> list[WordSpan]:
        return []


class FasterWhisperASR:
    def __init__(self, model_name: str = "small.en", device: str = "auto", compute_type: str = "default") -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("faster-whisper is not installed; install patrol-lens[media]") from exc
            device = self.device
            if device == "auto":
                device = "cpu"
            self._model = WhisperModel(self.model_name, device=device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio_path: str) -> list[WordSpan]:
        model = self._load()
        segments, _info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
        words: list[WordSpan] = []
        for segment in segments:
            segment_words = getattr(segment, "words", None) or []
            if not segment_words:
                words.append(WordSpan(int(segment.start * 1000), int(segment.end * 1000), segment.text.strip(), None))
                continue
            for word in segment_words:
                words.append(
                    WordSpan(
                        int(float(word.start) * 1000),
                        int(float(word.end) * 1000),
                        str(word.word).strip(),
                        float(word.probability) if getattr(word, "probability", None) is not None else None,
                    )
                )
        return words


def words_for_interval(words: list[WordSpan], start_ms: int, end_ms: int) -> list[WordSpan]:
    return [word for word in words if word.end_ms > start_ms and word.start_ms < end_ms]
