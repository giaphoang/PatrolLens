from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from ..config import DEFAULT_GEMINI_MODEL

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_INLINE_MEDIA_BYTES = 18 * 1024 * 1024


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
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("PATROLLENS_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        self.http_referer = http_referer or os.getenv("PATROLLENS_OPENROUTER_HTTP_REFERER")
        self.title = title or os.getenv("PATROLLENS_OPENROUTER_TITLE")
        self.timeout_s = timeout_s
        self.max_inline_media_bytes = max_inline_media_bytes
        self._client: Any = None
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for Gemini reasoning through OpenRouter")

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
        try:
            response = self._create_completion(
                model=selected_model,
                messages=messages,
                schema=schema,
            )
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
            response = self._create_completion(
                model=selected_model,
                messages=fallback_messages,
                schema=None,
            )

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("OpenRouter returned no completion choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else choice.message
        text = _message_text(message)
        if not text:
            raise RuntimeError("OpenRouter returned no structured text")
        return _parse_json(text)
