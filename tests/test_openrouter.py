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
    def __init__(self, dimensions: int = 3, usage_cost: float | None = None) -> None:
        self.dimensions = dimensions
        self.usage_cost = usage_cost
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"] if isinstance(kwargs["input"], list) else [kwargs["input"]]
        response = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index + 1)] * self.dimensions, index=index)
                for index, _input in enumerate(inputs)
            ]
        )
        if self.usage_cost is not None:
            response.usage = SimpleNamespace(
                prompt_tokens=11,
                total_tokens=11,
                cost=self.usage_cost,
            )
        return response


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


def test_openrouter_embedding_client_sync_batches_text_and_media(tmp_path):
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
    assert embeddings.calls[1]["model"] == "google/gemini-embedding-2"
    assert embeddings.calls[1]["input"] == [
        "title: none | text: Miranda rights",
        "title: none | text: ABC 123",
    ]
    assert embeddings.calls[2]["input"][0]["content"][0]["type"] == "image_url"
    assert embeddings.calls[3]["input"][0]["content"][0]["type"] == "video_url"
    assert embeddings.calls[4]["input"][0]["content"][0]["type"] == "input_audio"


def test_openrouter_embedding_client_uses_checkpointed_batch_api(tmp_path):
    calls: list[tuple[str, str, dict | None]] = []
    client = OpenRouterEmbeddingClient(
        model="google/gemini-embedding-2",
        batch_model="google/gemini-embedding-2:batch",
        dimensions=3,
        api_key="test-key",
        batch_api=True,
        batch_poll_interval_s=0,
        batch_timeout_s=30,
        batch_checkpoint_dir=tmp_path / "checkpoints",
    )

    def fake_http(method, url, *, payload=None, request_hash=None):
        calls.append((method, url, payload))
        if method == "POST":
            return {"id": "batch-123", "status": "validating"}
        return {
            "id": "batch-123",
            "status": "completed",
            "results": [
                {
                    "custom_id": "embedding-000001",
                    "response": {
                        "status_code": 200,
                        "body": {"data": [{"index": 0, "embedding": [2.0, 2.0, 2.0]}]},
                    },
                },
                {
                    "custom_id": "embedding-000000",
                    "response": {
                        "status_code": 200,
                        "body": {"data": [{"index": 0, "embedding": [1.0, 1.0, 1.0]}]},
                    },
                },
            ],
        }

    client._batch_http_json = fake_http
    vectors = client.encode_texts(["first", "second"])

    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    assert [call[0] for call in calls] == ["POST", "GET"]
    assert calls[0][1] == "https://openrouter.ai/api/beta/batches"
    payload = calls[0][2]
    assert payload is not None
    assert payload["endpoint"] == "/v1/embeddings"
    assert payload["model"] == "google/gemini-embedding-2"
    assert payload["requests"][0]["body"]["model"] == "google/gemini-embedding-2"
    assert payload["requests"][0]["body"]["dimensions"] == 3
    assert "output_dimensionality" not in payload["requests"][0]["body"]

    client._batch_http_json = lambda *_args, **_kwargs: pytest.fail(
        "completed batch should be read from its checkpoint"
    )
    assert client.encode_texts(["first", "second"]) == vectors
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 1


def test_openrouter_batch_embedding_records_result_usage(tmp_path):
    client = OpenRouterEmbeddingClient(
        model="google/gemini-embedding-2",
        batch_model="google/gemini-embedding-2:batch",
        dimensions=3,
        api_key="test-key",
        batch_api=True,
        batch_poll_interval_s=0,
        batch_checkpoint_dir=tmp_path / "checkpoints",
    )

    def fake_http(method, _url, *, payload=None, request_hash=None):
        if method == "POST":
            return {"id": "batch-with-usage", "status": "validating"}
        return {
            "id": "batch-with-usage",
            "status": "completed",
            "results": [
                {
                    "custom_id": "embedding-000000",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "data": [{"index": 0, "embedding": [1.0, 1.0, 1.0]}],
                            "usage": {"prompt_tokens": 7, "total_tokens": 7, "cost": 0.001},
                        },
                    },
                }
            ],
        }

    client._batch_http_json = fake_http

    assert client.encode_texts(["first"]) == [[1.0, 1.0, 1.0]]
    assert client.last_runtime_info["batch_jobs"] == 1
    assert client.last_runtime_info["batch_poll_requests"] == 1
    assert client.last_runtime_info["total_tokens"] == 7
    assert client.last_runtime_info["reported_cost_usd"] == 0.001
    assert client.last_runtime_info["cost_source"] == "provider"
    assert client.last_runtime_info["latency_ms"] >= 0


def test_openrouter_rejects_provider_dimension_mismatch():
    embeddings = FakeEmbeddings(dimensions=4)
    client = OpenRouterEmbeddingClient(dimensions=3, api_key="test-key")
    client._client = FakeEmbeddingClient(embeddings)

    with pytest.raises(EmbeddingDimensionError, match=r"Expected 3, got 4"):
        client.encode_text("dimension mismatch")


def test_openrouter_embedding_records_provider_usage_and_latency():
    embeddings = FakeEmbeddings(usage_cost=0.0042)
    client = OpenRouterEmbeddingClient(
        dimensions=3,
        api_key="test-key",
    )
    client._client = FakeEmbeddingClient(embeddings)

    client.encode_texts(["one", "two"])

    runtime = client.last_runtime_info
    assert runtime["api_calls"] == 1
    assert runtime["input_items"] == 2
    assert runtime["input_tokens"] == 11
    assert runtime["total_tokens"] == 11
    assert runtime["reported_cost_usd"] == 0.0042
    assert runtime["cost_source"] == "provider"
    assert runtime["latency_ms"] >= 0
