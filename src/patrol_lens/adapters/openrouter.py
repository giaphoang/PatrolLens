from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import (
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_MODEL,
)
from ..history import response_usage

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
    Query inputs use OpenRouter's synchronous embeddings endpoint. Bulk
    document/media inputs can either use that endpoint or OpenRouter's
    asynchronous Batch API. Batch jobs are checkpointed so an interrupted
    ingestion can reconnect without submitting the same paid job again.

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
        batch_api: bool = False,
        batch_poll_interval_s: float = 10.0,
        batch_timeout_s: float = 86_400.0,
        batch_checkpoint_dir: str | Path | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        if dimensions <= 0 or dimensions > 3072:
            raise ValueError("Gemini Embedding 2 dimensions must be between 1 and 3072")
        if media_batch_size <= 0:
            raise ValueError("embedding media batch size must be positive")
        if batch_poll_interval_s < 0:
            raise ValueError("embedding batch poll interval cannot be negative")
        if batch_timeout_s <= 0:
            raise ValueError("embedding batch timeout must be positive")
        self.model_name = model.removesuffix(":batch")
        self.query_model = (query_model or self.model_name).removesuffix(":batch")
        self.batch_api_enabled = batch_api
        self.batch_model = (batch_model or self.model_name) if batch_api else self.model_name
        self.dimensions = dimensions
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("PATROLLENS_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        self.http_referer = http_referer or os.getenv("PATROLLENS_OPENROUTER_HTTP_REFERER")
        self.title = title or os.getenv("PATROLLENS_OPENROUTER_TITLE")
        self.timeout_s = timeout_s
        self.max_inline_media_bytes = max_inline_media_bytes
        self.media_batch_size = media_batch_size
        self.batch_poll_interval_s = batch_poll_interval_s
        self.batch_timeout_s = batch_timeout_s
        self.batch_checkpoint_dir = (
            Path(batch_checkpoint_dir).expanduser().resolve()
            if batch_checkpoint_dir is not None
            else None
        )
        self.recorder = recorder
        self._client: Any = None
        self.last_runtime_info: dict[str, Any] = {}
        self.reset_runtime_info()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for Gemini Embedding 2")

    def reset_runtime_info(self) -> None:
        """Start a fresh usage/latency ledger for the next ingestion unit."""

        self.last_runtime_info = {
            "provider": "openrouter",
            "model": self.model_name,
            "models": [],
            "api_calls": 0,
            "batch_jobs": 0,
            "batch_poll_requests": 0,
            "batch_checkpoint_hits": 0,
            "input_items": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "reported_cost_usd": 0.0,
            "cost_available": True,
            "cost_source": "not_called",
            "usage_source": "openrouter_response.usage",
        }

    def _record_runtime(self, response: Any, *, model: str, input_count: int, started: float) -> None:
        tokens, provider_cost = response_usage(response)
        runtime = self.last_runtime_info
        runtime["api_calls"] = int(runtime.get("api_calls", 0)) + 1
        runtime["input_items"] = int(runtime.get("input_items", 0)) + input_count
        for key in ("input", "output", "total"):
            runtime_key = f"{key}_tokens"
            runtime[runtime_key] = int(runtime.get(runtime_key, 0)) + tokens[key]
        runtime["latency_ms"] = round(
            float(runtime.get("latency_ms", 0.0))
            + (time.monotonic() - started) * 1000,
            3,
        )
        models = runtime.setdefault("models", [])
        if model not in models:
            models.append(model)
        if provider_cost is None:
            runtime["cost_available"] = False
            runtime["reported_cost_usd"] = None
            runtime["cost_source"] = "unavailable"
        elif runtime.get("cost_available", True):
            runtime["reported_cost_usd"] = round(
                float(runtime.get("reported_cost_usd", 0.0) or 0.0)
                + provider_cost,
                8,
            )
            runtime["cost_source"] = "provider"
        else:
            runtime["cost_source"] = "unavailable"

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

    def _batch_headers(self, *, request_hash: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._embedding_headers())
        if request_hash:
            headers["Idempotency-Key"] = f"patrol-lens-embedding-{request_hash}"
        return headers

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    _mime = staticmethod(OpenRouterJSONClient._mime)
    _audio_format = staticmethod(OpenRouterJSONClient._audio_format)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()

    def _batch_endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/beta/batches"

    @staticmethod
    def _unwrap_batch(payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        if isinstance(data, dict) and ("id" in data or "status" in data):
            return data
        return payload

    @staticmethod
    def _http_error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", errors="replace")
        except Exception:
            return str(exc)

    def _batch_http_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        request_hash: str | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=(
                json.dumps(payload, separators=(",", ":")).encode("utf-8")
                if payload is not None
                else None
            ),
            headers=self._batch_headers(request_hash=request_hash),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = self._http_error_detail(exc)
            raise RuntimeError(f"OpenRouter Batch API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter Batch API request failed: {exc.reason}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("OpenRouter Batch API returned a non-object response")
        return self._unwrap_batch(result)

    def _batch_request_payload(self, inputs: list[Any]) -> dict[str, Any]:
        # The Batch endpoint selects batch pricing. Its request bodies use the
        # canonical model slug rather than forwarding ``:batch`` to the
        # synchronous embeddings endpoint.
        canonical_model = self.batch_model.removesuffix(":batch")
        requests = []
        for index, item in enumerate(inputs):
            requests.append(
                {
                    "custom_id": f"embedding-{index:06d}",
                    "body": {
                        "model": canonical_model,
                        "input": item,
                        "dimensions": self.dimensions,
                        "encoding_format": "float",
                    },
                }
            )
        return {
            "endpoint": "/v1/embeddings",
            "model": canonical_model,
            "requests": requests,
        }

    @staticmethod
    def _batch_request_hash(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _batch_checkpoint_path(self, request_hash: str) -> Path | None:
        if self.batch_checkpoint_dir is None:
            return None
        return self.batch_checkpoint_dir / f"{request_hash}.json"

    @staticmethod
    def _read_batch_checkpoint(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write_batch_checkpoint(path: Path | None, value: dict[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}-{threading.get_ident()}")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _checkpoint_batch(
        self,
        path: Path | None,
        *,
        request_hash: str,
        batch: dict[str, Any],
        include_result: bool = False,
    ) -> None:
        value: dict[str, Any] = {
            "schema_version": 1,
            "request_hash": request_hash,
            "batch_id": batch.get("id"),
            "status": batch.get("status"),
            "model": self.batch_model,
            "dimensions": self.dimensions,
            "updated_at": self._utc_now(),
        }
        if include_result:
            value["result"] = batch
        self._write_batch_checkpoint(path, value)

    @staticmethod
    def _batch_result_body(item: dict[str, Any]) -> dict[str, Any]:
        error = item.get("error")
        if error:
            raise RuntimeError(f"OpenRouter batch item failed: {error}")

        response = item.get("response")
        if isinstance(response, dict):
            status_code = response.get("status_code")
            if status_code is not None and not 200 <= int(status_code) < 300:
                raise RuntimeError(
                    f"OpenRouter batch item returned HTTP {status_code}: {response.get('body')}"
                )
            body = response.get("body")
            if isinstance(body, dict):
                return body
            if "data" in response:
                return response

        result = item.get("result")
        if isinstance(result, dict):
            nested = result.get("response")
            if isinstance(nested, dict):
                body = nested.get("body")
                if isinstance(body, dict):
                    return body
                if "data" in nested:
                    return nested
            if "data" in result:
                return result

        if "data" in item:
            return item
        raise RuntimeError(f"OpenRouter batch item has no embedding response: {item}")

    def _ordered_batch_items(
        self,
        batch: dict[str, Any],
        *,
        expected: int,
    ) -> list[dict[str, Any]]:
        raw_results = batch.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError("completed OpenRouter batch has no inline results")
        by_id: dict[str, dict[str, Any]] = {}
        without_id: list[dict[str, Any]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                raise RuntimeError("OpenRouter batch returned an invalid result item")
            custom_id = item.get("custom_id")
            if custom_id is None:
                without_id.append(item)
            else:
                by_id[str(custom_id)] = item
        ordered_items: list[dict[str, Any]] = []
        for index in range(expected):
            custom_id = f"embedding-{index:06d}"
            item = by_id.get(custom_id)
            if item is None and len(without_id) == expected:
                item = without_id[index]
            if item is None:
                raise RuntimeError(f"OpenRouter batch result is missing {custom_id}")
            ordered_items.append(item)
        return ordered_items

    def _batch_usage(
        self,
        batch: dict[str, Any],
        *,
        expected: int,
    ) -> tuple[dict[str, int], float | None]:
        """Aggregate usage from per-request batch bodies or the batch envelope."""

        items = self._ordered_batch_items(batch, expected=expected)
        bodies = [self._batch_result_body(item) for item in items]
        has_item_usage = any(self._field(body, "usage") is not None for body in bodies)
        if not has_item_usage:
            return response_usage(batch)
        totals = {"input": 0, "output": 0, "total": 0}
        cost: float | None = 0.0
        for body in bodies:
            tokens, item_cost = response_usage(body)
            for key in totals:
                totals[key] += tokens[key]
            if item_cost is None:
                cost = None
            elif cost is not None:
                cost += item_cost
        return totals, cost

    def _record_batch_runtime(
        self,
        batch: dict[str, Any],
        *,
        input_count: int,
        expected: int,
        started: float,
        poll_requests: int,
    ) -> None:
        tokens, provider_cost = self._batch_usage(batch, expected=expected)
        runtime = self.last_runtime_info
        runtime["api_calls"] = int(runtime.get("api_calls", 0)) + 1
        runtime["batch_jobs"] = int(runtime.get("batch_jobs", 0)) + 1
        runtime["batch_poll_requests"] = int(
            runtime.get("batch_poll_requests", 0)
        ) + poll_requests
        runtime["input_items"] = int(runtime.get("input_items", 0)) + input_count
        for key in ("input", "output", "total"):
            runtime_key = f"{key}_tokens"
            runtime[runtime_key] = int(runtime.get(runtime_key, 0)) + tokens[key]
        runtime["latency_ms"] = round(
            float(runtime.get("latency_ms", 0.0))
            + (time.monotonic() - started) * 1000,
            3,
        )
        model = self.batch_model
        models = runtime.setdefault("models", [])
        if model not in models:
            models.append(model)
        if provider_cost is None:
            runtime["cost_available"] = False
            runtime["reported_cost_usd"] = None
            runtime["cost_source"] = "unavailable"
        elif runtime.get("cost_available", True):
            runtime["reported_cost_usd"] = round(
                float(runtime.get("reported_cost_usd", 0.0) or 0.0)
                + provider_cost,
                8,
            )
            runtime["cost_source"] = "provider"
        else:
            runtime["cost_source"] = "unavailable"

    def _record_batch_checkpoint_hit(self, input_count: int) -> None:
        runtime = self.last_runtime_info
        runtime["batch_checkpoint_hits"] = int(
            runtime.get("batch_checkpoint_hits", 0)
        ) + 1
        runtime["cache_hits"] = int(runtime.get("cache_hits", 0)) + input_count
        if int(runtime.get("api_calls", 0)) == 0:
            runtime["cost_source"] = "checkpoint_only"

    def _validated_vectors(self, data: Any, expected: int) -> list[list[float]]:
        if not isinstance(data, list):
            raise RuntimeError("OpenRouter returned an invalid embeddings response")
        ordered = sorted(data, key=lambda item: int(self._field(item, "index", 0)))
        if len(ordered) != expected:
            raise RuntimeError(
                f"OpenRouter returned {len(ordered)} embeddings for {expected} inputs"
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

    def _vectors_from_batch(
        self,
        batch: dict[str, Any],
        *,
        expected: int,
    ) -> list[list[float]]:
        ordered_items = self._ordered_batch_items(batch, expected=expected)

        vectors: list[list[float]] = []
        for item in ordered_items:
            body = self._batch_result_body(item)
            vectors.extend(self._validated_vectors(body.get("data"), 1))
        return vectors

    def _batch_request(
        self,
        inputs: list[Any],
        *,
        started: float | None = None,
    ) -> list[list[float]]:
        started = started or time.monotonic()
        payload = self._batch_request_payload(inputs)
        request_hash = self._batch_request_hash(payload)
        checkpoint_path = self._batch_checkpoint_path(request_hash)
        checkpoint = self._read_batch_checkpoint(checkpoint_path)

        batch: dict[str, Any] | None = None
        if checkpoint and checkpoint.get("request_hash") == request_hash:
            saved_result = checkpoint.get("result")
            if isinstance(saved_result, dict) and saved_result.get("status") == "completed":
                self._record_batch_checkpoint_hit(len(inputs))
                return self._vectors_from_batch(saved_result, expected=len(inputs))
            if checkpoint.get("status") not in {
                "failed",
                "canceled",
                "cancelled",
                "expired",
            }:
                batch_id = checkpoint.get("batch_id")
                if batch_id:
                    batch = {"id": str(batch_id), "status": checkpoint.get("status")}

        if batch is None:
            batch = self._batch_http_json(
                "POST",
                self._batch_endpoint(),
                payload=payload,
                request_hash=request_hash,
            )
            if not batch.get("id"):
                raise RuntimeError("OpenRouter Batch API did not return a batch id")
            self._checkpoint_batch(
                checkpoint_path,
                request_hash=request_hash,
                batch=batch,
                include_result=str(batch.get("status", "")).lower() == "completed",
            )

        batch_id = str(batch["id"])
        deadline = time.monotonic() + self.batch_timeout_s
        terminal = {"completed", "failed", "canceled", "cancelled", "expired"}
        poll_requests = 0
        while str(batch.get("status", "")).lower() not in terminal:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"OpenRouter embedding batch {batch_id} did not finish within "
                    f"{self.batch_timeout_s:g}s; rerun ingestion to resume polling"
                )
            if self.batch_poll_interval_s:
                time.sleep(min(self.batch_poll_interval_s, max(0.0, deadline - time.monotonic())))
            batch = self._batch_http_json("GET", f"{self._batch_endpoint()}/{batch_id}")
            poll_requests += 1
            self._checkpoint_batch(
                checkpoint_path,
                request_hash=request_hash,
                batch=batch,
                include_result=str(batch.get("status", "")).lower() == "completed",
            )

        status = str(batch.get("status", "")).lower()
        if status != "completed":
            detail = batch.get("error") or batch.get("errors") or batch.get("request_counts")
            raise RuntimeError(f"OpenRouter embedding batch {batch_id} ended as {status}: {detail}")
        self._record_batch_runtime(
            batch,
            input_count=len(inputs),
            expected=len(inputs),
            started=started,
            poll_requests=poll_requests,
        )
        return self._vectors_from_batch(batch, expected=len(inputs))

    def _request(
        self,
        inputs: list[Any],
        *,
        model: str,
        single_text: bool = False,
    ) -> list[list[float]]:
        if not inputs:
            return []
        started = time.monotonic()
        if self.batch_api_enabled and model == self.batch_model:
            return self._batch_request(inputs, started=started)
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
        self._record_runtime(
            response,
            model=model,
            input_count=len(inputs),
            started=started,
        )
        if self.recorder and request_id is not None:
            self.recorder.model_response(
                response,
                request_event_id=request_id,
                started_monotonic=started,
                model=model,
                output_summary={"embedding_count": len(self._field(response, "data", []) or [])},
            )
        data = self._field(response, "data", []) or []
        return self._validated_vectors(data, len(inputs))

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
        """Encode document text using the configured bulk model."""

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

    def canary_ingestion(self) -> list[float]:
        """Exercise the selected ingestion transport and validate its dimensions."""

        return self.encode_texts(["PatrolLens embedding dimension canary"])[0]

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
