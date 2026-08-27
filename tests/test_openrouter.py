from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from patrol_lens.adapters.openrouter import OpenRouterJSONClient


class FakeCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.response = response or SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"supported":true}'))]
        )
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None and len(self.calls) == 1:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def test_openrouter_builds_openai_compatible_multimodal_payload(tmp_path):
    image = tmp_path / "frame.jpg"
    video = tmp_path / "clip.mp4"
    audio = tmp_path / "speech.wav"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    completions = FakeCompletions()
    client = OpenRouterJSONClient(
        model="google/gemini-3.1-pro-preview",
        api_key="test-key",
        http_referer="https://patrol-lens.test",
        title="PatrolLens tests",
    )
    client._client = FakeClient(completions)

    result = client.generate_json(
        "Inspect the selected evidence.",
        {"type": "object", "properties": {"supported": {"type": "boolean"}}},
        media_paths=[str(image), str(video), str(audio)],
    )

    assert result == {"supported": True}
    request = completions.calls[0]
    assert request["model"] == "google/gemini-3.1-pro-preview"
    assert request["temperature"] == 0
    assert request["extra_headers"] == {
        "HTTP-Referer": "https://patrol-lens.test",
        "X-OpenRouter-Title": "PatrolLens tests",
    }
    parts = request["messages"][0]["content"]
    assert [part["type"] for part in parts] == ["text", "image_url", "video_url", "input_audio"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[2]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert parts[3]["input_audio"]["format"] == "wav"
    assert base64.b64decode(parts[3]["input_audio"]["data"]) == b"audio"
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True


def test_openrouter_retries_without_schema_when_provider_rejects_it():
    completions = FakeCompletions(error=RuntimeError("response_format json_schema is unsupported"))
    client = OpenRouterJSONClient(api_key="test-key")
    client._client = FakeClient(completions)

    result = client.generate_json(
        "Return a decision.",
        {"type": "object", "properties": {"ok": {"type": "boolean"}}},
    )

    assert result == {"supported": True}
    assert len(completions.calls) == 2
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]
    assert "JSON Schema" in completions.calls[1]["messages"][0]["content"][0]["text"]


def test_openrouter_requires_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterJSONClient()
