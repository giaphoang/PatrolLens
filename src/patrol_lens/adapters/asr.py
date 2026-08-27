from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Protocol


DEFAULT_OPENROUTER_ASR_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RAW_AUDIO_CHUNK_BYTES = 18 * 1024 * 1024


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
    """Optional local fallback retained for explicit canary comparisons."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "default",
        *,
        word_timestamps: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.word_timestamps = word_timestamps
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed; run uv sync --extra speech"
                ) from exc
            device = "cpu" if self.device == "auto" else self.device
            self._model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str) -> list[WordSpan]:
        segments, _info = self._load().transcribe(
            audio_path,
            word_timestamps=self.word_timestamps,
            vad_filter=True,
        )
        spans: list[WordSpan] = []
        for segment in segments:
            segment_words = (
                (getattr(segment, "words", None) or [])
                if self.word_timestamps
                else []
            )
            if segment_words:
                for word in segment_words:
                    spans.append(
                        WordSpan(
                            int(float(word.start) * 1000),
                            int(float(word.end) * 1000),
                            str(word.word).strip(),
                            float(word.probability)
                            if getattr(word, "probability", None) is not None
                            else None,
                        )
                    )
                continue
            confidence = None
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is not None:
                confidence = min(
                    1.0,
                    max(0.0, math.exp(min(0.0, float(avg_logprob)))),
                )
            spans.append(
                WordSpan(
                    int(segment.start * 1000),
                    int(segment.end * 1000),
                    segment.text.strip(),
                    confidence,
                )
            )
        return spans


@dataclass(frozen=True)
class _AudioChunk:
    path: Path
    start_ms: int
    duration_ms: int
    content_hash: str


class OpenRouterASR:
    """Checkpointed segment transcription through OpenRouter's STT endpoint."""

    cache_version = "openrouter-stt-v1"

    def __init__(
        self,
        model: str = DEFAULT_OPENROUTER_ASR_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_referer: str | None = None,
        title: str | None = None,
        language: str = "auto",
        chunk_seconds: int = 300,
        timeout_s: float = 90.0,
        max_retries: int = 3,
    ) -> None:
        if chunk_seconds <= 0 or chunk_seconds > 600:
            raise ValueError("OpenRouter ASR chunk seconds must be between 1 and 600")
        if timeout_s <= 0:
            raise ValueError("OpenRouter ASR timeout must be positive")
        if max_retries < 0:
            raise ValueError("OpenRouter ASR retries cannot be negative")
        self.model_name = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv(
            "PATROLLENS_OPENROUTER_BASE_URL",
            DEFAULT_OPENROUTER_BASE_URL,
        )
        self.http_referer = http_referer or os.getenv(
            "PATROLLENS_OPENROUTER_HTTP_REFERER"
        )
        self.title = title or os.getenv("PATROLLENS_OPENROUTER_TITLE")
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.last_runtime_info: dict[str, Any] = {}
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for remote transcription")

    @staticmethod
    def _cache_root(audio_path: Path) -> Path:
        media_root = audio_path.parent.parent
        return media_root / "transcripts" / audio_path.stem

    def _cache_path(self, audio_path: Path, chunk: _AudioChunk) -> Path:
        descriptor = "\0".join(
            (
                chunk.content_hash,
                self.model_name,
                self.language,
                self.cache_version,
            )
        )
        key = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[:24]
        model_dir = self.model_name.replace("/", "--").replace(":", "--")
        return self._cache_root(audio_path) / model_dir / (
            f"{chunk.start_ms:012d}-{key}.json"
        )

    def _iter_chunks(self, audio_path: Path, output_dir: Path) -> Iterator[_AudioChunk]:
        try:
            source = wave.open(str(audio_path), "rb")
        except (wave.Error, EOFError) as exc:
            raise RuntimeError(
                f"OpenRouter ASR requires the extracted PCM WAV audio: {audio_path}"
            ) from exc
        with source:
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            if (
                sample_rate <= 0
                or channels <= 0
                or sample_width <= 0
                or source.getcomptype() != "NONE"
            ):
                raise RuntimeError(f"invalid WAV format: {audio_path}")
            frames_per_chunk = min(
                sample_rate * self.chunk_seconds,
                MAX_RAW_AUDIO_CHUNK_BYTES // (channels * sample_width),
            )
            start_frame = 0
            ordinal = 0
            while True:
                frames = source.readframes(frames_per_chunk)
                if not frames:
                    break
                frame_count = len(frames) // (channels * sample_width)
                chunk_path = output_dir / f"chunk-{ordinal:06d}.wav"
                with wave.open(str(chunk_path), "wb") as output:
                    output.setnchannels(channels)
                    output.setsampwidth(sample_width)
                    output.setframerate(sample_rate)
                    output.writeframes(frames)
                yield _AudioChunk(
                    path=chunk_path,
                    start_ms=round(start_frame * 1000 / sample_rate),
                    duration_ms=round(frame_count * 1000 / sample_rate),
                    content_hash=hashlib.sha256(frames).hexdigest(),
                )
                start_frame += frame_count
                ordinal += 1

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.title:
            headers["X-OpenRouter-Title"] = self.title
        return headers

    def _request_payload(self, chunk: _AudioChunk) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input_audio": {
                "data": base64.b64encode(chunk.path.read_bytes()).decode("ascii"),
                "format": "wav",
            },
            "response_format": "verbose_json",
            "temperature": 0,
        }
        if self.language and self.language != "auto":
            payload["language"] = self.language
        return payload

    @staticmethod
    def _error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", errors="replace")
        except Exception:
            return str(exc)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.base_url.rstrip('/')}/audio/transcriptions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise RuntimeError("OpenRouter returned an invalid transcription response")
                return result
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {408, 409, 429} or exc.code >= 500
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OpenRouter transcription failed ({exc.code}): "
                        f"{self._error_detail(exc)}"
                    ) from exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OpenRouter transcription request failed: {exc.reason}"
                    ) from exc
            time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable OpenRouter retry state")

    @staticmethod
    def _field(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    @classmethod
    def _confidence(cls, segment: Any) -> float | None:
        direct = cls._field(segment, "confidence")
        if direct is not None:
            return min(1.0, max(0.0, float(direct)))
        avg_logprob = cls._field(segment, "avg_logprob")
        if avg_logprob is not None:
            return min(1.0, max(0.0, math.exp(min(0.0, float(avg_logprob)))))
        return None

    def _parse_segments(
        self,
        response: dict[str, Any],
        chunk: _AudioChunk,
    ) -> list[WordSpan]:
        segments = response.get("segments") or []
        transcript = str(response.get("text", "")).strip()
        if transcript and not segments:
            raise RuntimeError(
                "OpenRouter returned transcript text without segment timestamps; "
                "the response must support verbose_json"
            )
        spans: list[WordSpan] = []
        for segment in segments:
            text = str(self._field(segment, "text", "")).strip()
            if not text:
                continue
            local_start = max(0, round(float(self._field(segment, "start", 0)) * 1000))
            local_end = max(
                local_start,
                round(float(self._field(segment, "end", local_start / 1000)) * 1000),
            )
            local_start = min(local_start, chunk.duration_ms)
            local_end = min(local_end, chunk.duration_ms)
            spans.append(
                WordSpan(
                    start_ms=chunk.start_ms + local_start,
                    end_ms=chunk.start_ms + local_end,
                    text=text,
                    confidence=self._confidence(segment),
                )
            )
        return spans

    @staticmethod
    def _usage_cost(response: dict[str, Any]) -> float:
        usage = response.get("usage") or {}
        try:
            return float(usage.get("cost") or 0)
        except (TypeError, ValueError):
            return 0.0

    def transcribe(self, audio_path: str) -> list[WordSpan]:
        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        spans: list[WordSpan] = []
        api_calls = 0
        cache_hits = 0
        cost = 0.0
        with TemporaryDirectory(prefix="patrol-lens-openrouter-asr-") as temp_dir:
            for chunk in self._iter_chunks(source, Path(temp_dir)):
                cache_path = self._cache_path(source, chunk)
                if cache_path.is_file():
                    response = json.loads(cache_path.read_text())
                    cache_hits += 1
                else:
                    response = self._post_json(self._request_payload(chunk))
                    parsed = self._parse_segments(response, chunk)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(response, separators=(",", ":")))
                    temporary.replace(cache_path)
                    spans.extend(parsed)
                    api_calls += 1
                    cost += self._usage_cost(response)
                    continue
                spans.extend(self._parse_segments(response, chunk))
        self.last_runtime_info = {
            "api_calls": api_calls,
            "cache_hits": cache_hits,
            "reported_cost_usd": round(cost, 8),
            "chunk_seconds": self.chunk_seconds,
        }
        return sorted(spans, key=lambda item: (item.start_ms, item.end_ms))


def words_for_interval(words: list[WordSpan], start_ms: int, end_ms: int) -> list[WordSpan]:
    return [word for word in words if word.end_ms > start_ms and word.start_ms < end_ms]
