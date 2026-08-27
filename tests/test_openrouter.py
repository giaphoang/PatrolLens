from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from patrol_lens.adapters.openrouter import (
    EmbeddingDimensionError,
    OpenRouterEmbeddingClient,
    OpenRouterJSONClient,
)


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


class FakeEmbeddings:
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"] if isinstance(kwargs["input"], list) else [kwargs["input"]]
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index + 1)] * self.dimensions, index=index)
                for index, _input in enumerate(inputs)
            ]
        )


class FakeEmbeddingClient:
    def __init__(self, embeddings: FakeEmbeddings) -> None:
        self.embeddings = embeddings


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


def test_openrouter_embedding_client_batches_text_and_media(tmp_path):
    image = tmp_path / "frame.jpg"
    video = tmp_path / "chunk.mp4"
    audio = tmp_path / "chunk.wav"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    embeddings = FakeEmbeddings()
    client = OpenRouterEmbeddingClient(
        model="google/gemini-embedding-2",
        batch_model="google/gemini-embedding-2:batch",
        query_model="google/gemini-embedding-2",
        dimensions=3,
        api_key="test-key",
        media_batch_size=2,
    )
    client._client = FakeEmbeddingClient(embeddings)

    query_vector = client.encode_text("red jacket")
    document_vectors = client.encode_texts(["Miranda rights", "ABC 123"])
    media_vectors = client.encode_media_many([image, video, audio])

    assert len(query_vector) == 3
    assert len(document_vectors) == 2
    assert len(media_vectors) == 3
    assert embeddings.calls[0]["model"] == "google/gemini-embedding-2"
    assert embeddings.calls[0]["input"] == "task: search result | query: red jacket"
    assert embeddings.calls[0]["dimensions"] == 3
    assert embeddings.calls[0]["extra_body"] == {"output_dimensionality": 3}
    assert embeddings.calls[1]["model"] == "google/gemini-embedding-2:batch"
    assert embeddings.calls[1]["input"] == [
        "title: none | text: Miranda rights",
        "title: none | text: ABC 123",
    ]
    assert embeddings.calls[2]["input"][0]["content"][0]["type"] == "image_url"
    assert embeddings.calls[3]["input"][0]["content"][0]["type"] == "video_url"
    assert embeddings.calls[4]["input"][0]["content"][0]["type"] == "input_audio"


def test_openrouter_rejects_provider_dimension_mismatch():
    embeddings = FakeEmbeddings(dimensions=4)
    client = OpenRouterEmbeddingClient(dimensions=3, api_key="test-key")
    client._client = FakeEmbeddingClient(embeddings)

    with pytest.raises(EmbeddingDimensionError, match=r"Expected 3, got 4"):
        client.encode_text("dimension mismatch")
