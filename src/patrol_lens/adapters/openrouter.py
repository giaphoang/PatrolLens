from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import (
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_MODEL,
)

if TYPE_CHECKING:
    from ..history import TrajectoryRecorder

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_INLINE_MEDIA_BYTES = 18 * 1024 * 1024


class EmbeddingDimensionError(RuntimeError):
    """Raised when a provider returns a vector of the wrong size."""

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Expected {expected}, got {actual}")


def _parse_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.removeprefix("```json").removeprefix("```")
        value = value.removesuffix("```").strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("OpenRouter structured response must be a JSON object")
    return parsed


def _message_text(message: Any) -> str:
    """Normalize the OpenAI SDK's string or content-part response shape."""

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                chunks.append(str(text))
        return "".join(chunks)
    if content is not None:
        return str(content)
    return ""


class OpenRouterJSONClient:
    """OpenAI-compatible OpenRouter boundary for structured multimodal calls.

    The rest of PatrolLens only depends on ``generate_json``. Keeping the
    OpenRouter SDK details here makes the planner, active policy, verifier,
    and timestamp refiner provider-neutral.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_referer: str | None = None,
        title: str | None = None,
        timeout_s: float = 120.0,
        max_inline_media_bytes: int = DEFAULT_MAX_INLINE_MEDIA_BYTES,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("PATROLLENS_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        self.http_referer = http_referer or os.getenv("PATROLLENS_OPENROUTER_HTTP_REFERER")
        self.title = title or os.getenv("PATROLLENS_OPENROUTER_TITLE")
        self.timeout_s = timeout_s
        self.max_inline_media_bytes = max_inline_media_bytes
        self.recorder = recorder
        self._client: Any = None
        self._client_lock = threading.Lock()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for Gemini reasoning through OpenRouter")

    def _load(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        from openai import OpenAI
                    except ImportError as exc:
                        raise RuntimeError(
                            "openai is not installed; install patrol-lens[openrouter]"
                        ) from exc
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout_s,
                    )
        return self._client

    @staticmethod
    def _mime(path: Path) -> str:
        guessed, _encoding = mimetypes.guess_type(path.name)
        if guessed:
            return guessed
        return {
            ".aac": "audio/aac",
            ".aiff": "audio/aiff",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".mp3": "audio/mpeg",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".ogg": "audio/ogg",
            ".wav": "audio/wav",
            ".webm": "video/webm",
        }.get(path.suffix.lower(), "application/octet-stream")

    @staticmethod
    def _audio_format(path: Path, mime_type: str) -> str:
        extension = path.suffix.lower().lstrip(".")
        if extension:
            return extension
        return mime_type.split("/", 1)[-1]

    def _media_part(self, path: Path) -> dict[str, Any]:
        mime_type = self._mime(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        if mime_type.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": data_url}}
        if mime_type.startswith("video/"):
            return {"type": "video_url", "video_url": {"url": data_url}}
        if mime_type.startswith("audio/"):
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": encoded,
                    "format": self._audio_format(path, mime_type),
                },
            }
        raise ValueError(f"unsupported multimodal media type for {path}: {mime_type}")

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.title:
            headers["X-OpenRouter-Title"] = self.title
        return headers

    @staticmethod
    def _schema_response_format(schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "patrol_lens_response",
                "strict": True,
                "schema": schema,
            },
        }

    def _create_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
        }
        if schema is not None:
            request["response_format"] = self._schema_response_format(schema)
        headers = self._headers()
        if headers:
            request["extra_headers"] = headers
        return self._load().chat.completions.create(**request)

    @staticmethod
    def _supports_plain_json_retry(exc: Exception) -> bool:
        detail = str(exc).lower()
        return any(
            marker in detail
            for marker in (
                "response_format",
                "json_schema",
                "structured output",
                "structured_output",
                "does not support",
            )
        )

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        media_paths: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        total_bytes = 0
        for raw_path in media_paths or []:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            total_bytes += path.stat().st_size
            if total_bytes > self.max_inline_media_bytes:
                raise ValueError(
                    f"selected media exceeds {self.max_inline_media_bytes} inline bytes; request a shorter interval"
                )
            parts.append(self._media_part(path))

        messages = [{"role": "user", "content": parts}]
        selected_model = model or self.model

        def complete(
            request_messages: list[dict[str, Any]],
            request_schema: dict[str, Any] | None,
            *,
            retry: bool = False,
        ) -> Any:
            request_id: str | None = None
            started = time.monotonic()
            if self.recorder:
                request_id, started = self.recorder.model_request(
                    model=selected_model,
                    input_summary={
                        "prompt": prompt,
                        "structured_output": request_schema is not None,
                        "plain_json_retry": retry,
                    },
                    media_references=list(media_paths or []),
                )
            try:
                result = self._create_completion(
                    model=selected_model,
                    messages=request_messages,
                    schema=request_schema,
                )
            except Exception as exc:
                if self.recorder:
                    self.recorder.provider_error(
                        exc,
                        model=selected_model,
                        request_event_id=request_id,
                        started_monotonic=started,
                    )
                raise
            if self.recorder and request_id is not None:
                self.recorder.model_response(
                    result,
                    request_event_id=request_id,
                    started_monotonic=started,
                    model=selected_model,
                    output_summary={"choice_count": len(getattr(result, "choices", None) or [])},
                )
            return result

        try:
            response = complete(messages, schema)
        except Exception as exc:
            if not self._supports_plain_json_retry(exc):
                raise
            fallback_prompt = (
                f"{prompt}\n\nReturn only a valid JSON object matching this JSON Schema:\n"
                f"{json.dumps(schema, separators=(',', ':'))}"
            )
            fallback_messages = [{
                "role": "user",
                "content": [{"type": "text", "text": fallback_prompt}, *parts[1:]],
            }]
            response = complete(fallback_messages, None, retry=True)

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("OpenRouter returned no completion choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else choice.message
        text = _message_text(message)
        if not text:
            raise RuntimeError("OpenRouter returned no structured text")
        return _parse_json(text)


class OpenRouterEmbeddingClient:
    """OpenRouter client for Gemini Embedding 2 text and media vectors.

    ``model_name`` is the canonical model namespace stored with an embedding.
    The synchronous embeddings endpoint is the safe default for both offline
    inputs and queries. Callers may explicitly provide a ``:batch`` model for
    a separate asynchronous Batch API implementation.

    OpenRouter's embeddings endpoint represents multimodal input as an input
    object containing a ``content`` array. The object is deliberately kept at
    this provider boundary so the ingestion and retrieval layers only depend
    on ``encode_*`` methods.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
        *,
        batch_model: str | None = None,
        query_model: str | None = None,
        dimensions: int = 3072,
        api_key: str | None = None,
        base_url: str | None = None,
        http_referer: str | None = None,
        title: str | None = None,
        timeout_s: float = 120.0,
        max_inline_media_bytes: int = DEFAULT_MAX_INLINE_MEDIA_BYTES,
        media_batch_size: int = 6,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        if dimensions <= 0 or dimensions > 3072:
            raise ValueError("Gemini Embedding 2 dimensions must be between 1 and 3072")
        if media_batch_size <= 0:
            raise ValueError("embedding media batch size must be positive")
        self.model_name = model.removesuffix(":batch")
        self.query_model = query_model or self.model_name
        # ``:batch`` models are not accepted by the synchronous
        # ``/api/v1/embeddings`` endpoint. Keep sync ingestion safe by using
        # the canonical model unless a caller explicitly opts into a separate
        # Batch API transport.
        self.batch_model = batch_model or self.model_name
        self.dimensions = dimensions
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("PATROLLENS_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        self.http_referer = http_referer or os.getenv("PATROLLENS_OPENROUTER_HTTP_REFERER")
        self.title = title or os.getenv("PATROLLENS_OPENROUTER_TITLE")
        self.timeout_s = timeout_s
        self.max_inline_media_bytes = max_inline_media_bytes
        self.media_batch_size = media_batch_size
        self.recorder = recorder
        self._client: Any = None
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for Gemini Embedding 2")

    def _load(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai is not installed; install patrol-lens[openrouter]") from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
            )
        return self._client

    def _embedding_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.title:
            headers["X-OpenRouter-Title"] = self.title
        return headers

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    _mime = staticmethod(OpenRouterJSONClient._mime)
    _audio_format = staticmethod(OpenRouterJSONClient._audio_format)

    def _request(
        self,
        inputs: list[Any],
        *,
        model: str,
        single_text: bool = False,
    ) -> list[list[float]]:
        if not inputs:
            return []
        payload: dict[str, Any] = {
            "model": model,
            "input": inputs[0] if single_text else inputs,
            "dimensions": self.dimensions,
            "encoding_format": "float",
            # OpenRouter documents the OpenAI-compatible ``dimensions`` field,
            # while Gemini's native API names the same control
            # ``output_dimensionality``. Send both so the selected Google
            # provider cannot silently fall back to its 3072 default.
            "extra_body": {"output_dimensionality": self.dimensions},
        }
        headers = self._embedding_headers()
        if headers:
            payload["extra_headers"] = headers
        request_id: str | None = None
        started = time.monotonic()
        if self.recorder:
            request_id, started = self.recorder.model_request(
                model=model,
                input_summary={
                    "operation": "embedding",
                    "input_count": len(inputs),
                    "dimensions": self.dimensions,
                },
            )
        try:
            response = self._load().embeddings.create(**payload)
        except Exception as exc:
            if self.recorder:
                self.recorder.provider_error(
                    exc,
                    model=model,
                    request_event_id=request_id,
                    started_monotonic=started,
                )
            raise
        if self.recorder and request_id is not None:
            self.recorder.model_response(
                response,
                request_event_id=request_id,
                started_monotonic=started,
                model=model,
                output_summary={"embedding_count": len(self._field(response, "data", []) or [])},
            )
        data = self._field(response, "data", []) or []
        if not isinstance(data, list):
            raise RuntimeError("OpenRouter returned an invalid embeddings response")
        ordered = sorted(data, key=lambda item: int(self._field(item, "index", 0)))
        if len(ordered) != len(inputs):
            raise RuntimeError(
                f"OpenRouter returned {len(ordered)} embeddings for {len(inputs)} inputs"
            )
        vectors: list[list[float]] = []
        for item in ordered:
            raw_vector = self._field(item, "embedding")
            if not isinstance(raw_vector, (list, tuple)) or not raw_vector:
                raise RuntimeError("OpenRouter returned an empty embedding vector")
            vector = [float(value) for value in raw_vector]
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError("OpenRouter returned a non-finite embedding vector")
            if len(vector) != self.dimensions:
                raise EmbeddingDimensionError(self.dimensions, len(vector))
            vectors.append(vector)
        return vectors

    @staticmethod
    def _document_text(text: str) -> str:
        return f"title: none | text: {text}"

    @staticmethod
    def _query_text(text: str) -> str:
        return f"task: search result | query: {text}"

    def encode_text(self, text: str) -> list[float]:
        """Encode a query using the asymmetric search-query instruction."""

        return self._request(
            [self._query_text(text)],
            model=self.query_model,
            single_text=True,
        )[0]

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode document text using the configured synchronous model."""

        if not texts:
            return []
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.media_batch_size):
            vectors.extend(
                self._request(
                    [self._document_text(text) for text in texts[offset : offset + self.media_batch_size]],
                    model=self.batch_model,
                )
            )
        return vectors

    encode_documents = encode_texts

    def _media_input(self, path: str | Path, context_text: str | None = None) -> tuple[dict[str, Any], str, int]:
        media_path = Path(path)
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        size = media_path.stat().st_size
        if size > self.max_inline_media_bytes:
            raise ValueError(
                f"media {media_path} exceeds {self.max_inline_media_bytes} inline bytes"
            )
        part = OpenRouterJSONClient._media_part(self, media_path)
        content: list[dict[str, Any]] = []
        if context_text:
            content.append({"type": "text", "text": context_text})
        content.append(part)
        mime_type = self._mime(media_path)
        kind = mime_type.split("/", 1)[0]
        return {"content": content}, kind, size

    def encode_media(
        self,
        path: str | Path,
        *,
        context_text: str | None = None,
    ) -> list[float]:
        return self.encode_media_many([path], context_texts=[context_text] if context_text else None)[0]

    def encode_media_many(
        self,
        paths: list[str | Path],
        *,
        context_texts: list[str] | None = None,
    ) -> list[list[float]]:
        """Encode local image/audio/video files, preserving input order.

        Images are grouped up to ``media_batch_size``. Audio and video are
        sent one item per request because their provider-side token budgets
        are duration-based and substantially larger than image requests.
        """

        if not paths:
            return []
        if context_texts is not None and len(context_texts) != len(paths):
            raise ValueError("context_texts must have one entry per media path")
        items = [
            self._media_input(path, context_texts[index] if context_texts else None)
            for index, path in enumerate(paths)
        ]
        outputs: list[list[float]] = []
        batch: list[dict[str, Any]] = []
        batch_kind: str | None = None
        batch_bytes = 0
        for item, kind, size in items:
            max_items = self.media_batch_size if kind == "image" else 1
            incompatible = batch and (batch_kind != kind or len(batch) >= max_items)
            too_large = batch and batch_bytes + size > self.max_inline_media_bytes
            if incompatible or too_large:
                outputs.extend(self._request(batch, model=self.batch_model))
                batch = []
                batch_kind = None
                batch_bytes = 0
            batch.append(item)
            batch_kind = kind
            batch_bytes += size
        if batch:
            outputs.extend(self._request(batch, model=self.batch_model))
        return outputs

    def encode_image(self, path: str | Path) -> list[float]:
        return self.encode_media(path)

    def encode_images(self, paths: list[str | Path]) -> list[list[float]]:
        return self.encode_media_many(paths)

    def encode_audio(self, path: str | Path) -> list[float]:
        return self.encode_media(path)

    def encode_video(self, path: str | Path) -> list[float]:
        return self.encode_media(path)
