from __future__ import annotations

import shutil
import subprocess

import pytest

from patrol_lens.adapters.asr import WordSpan
from patrol_lens.adapters.audio import AudioAnalysis
from patrol_lens.config import IngestionConfig, RetrievalConfig
from patrol_lens.index import IndexStore, SQLiteVectorIndex
from patrol_lens.ingestion import IngestionBackends, IngestionPipeline
from patrol_lens.retrieval import CoarseRetriever, HeuristicQueryPlanner
from patrol_lens.text import HashEmbeddingEncoder


class FakeVisual(HashEmbeddingEncoder):
    model_name = "fake-siglip"

    def __init__(self):
        super().__init__(model_name=self.model_name)

    def encode_image(self, _image_path):
        return self.encode_text("person wearing a red shirt")

    def encode_images(self, paths):
        return [self.encode_image(path) for path in paths]


class FakeASR:
    model_name = "fake-whisper"

    def transcribe(self, _audio_path):
        return [WordSpan(500, 1600, "right to remain silent", 0.9)]


class FakeOCR:
    model_name = "fake-ocr"

    def detect(self, _image_path):
        return [{"text": "ABC 123", "confidence": 0.95, "box": [[0, 0], [10, 10]]}]


class FakeAudio:
    model_name = "fake-audio"

    def analyze(self, _audio_path, _start_ms, _end_ms):
        return AudioAnalysis(0.9, ["speech"], 0.88)


class FakeMultimodalEmbedding:
    model_name = "fake-gemini-embedding-2"
    batch_model = "fake-gemini-embedding-2:batch"

    def __init__(self):
        self.encoder = HashEmbeddingEncoder(dimensions=4, model_name=self.model_name)
        self.text_inputs = []
        self.media_inputs = []

    def encode_text(self, text):
        return self.encoder.encode_text(text)

    def encode_texts(self, texts):
        self.text_inputs.extend(texts)
        return [self.encode_text(text) for text in texts]

    def encode_media_many(self, paths, *, context_texts=None):
        self.media_inputs.extend(str(path) for path in paths)
        return [self.encoder.encode_text(f"media {path}") for path in paths]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_synthetic_core_ingestion_to_multimodal_retrieval(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=160x120:d=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest",
            "-c:v", "libx264", "-c:a", "aac", str(source),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    store = IndexStore(tmp_path / "index")
    visual = FakeVisual()
    stats = IngestionPipeline(
        store,
        backends=IngestionBackends(
            visual=visual,
            asr=FakeASR(),
        ),
        config=IngestionConfig(
            window_ms=2_000,
            stride_ms=1_000,
            frame_step_ms=1_000,
            audio_window_ms=2_000,
            audio_stride_ms=1_000,
        ),
    ).ingest_path(source)
    retriever = CoarseRetriever(
        store,
        planner=HeuristicQueryPlanner(),
        visual_encoder=visual,
        vector_index=SQLiteVectorIndex(store),
        config=RetrievalConfig(top_k=5),
    )

    _plan, candidates = retriever.retrieve(
        "Find the person in the red shirt while saying right to remain silent"
    )

    assert stats["visual"] == 1
    assert stats["transcript"] == 1
    assert stats["ocr"] == 0
    assert stats["audio"] == 0
    assert candidates
    assert set(candidates[0].covered_modalities) >= {"visual", "transcript"}
    store.close()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_gemini_embedding_ingestion_keeps_exact_text_and_indexes_all_modalities(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest",
            "-c:v", "libx264", "-c:a", "aac", str(source),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    store = IndexStore(tmp_path / "index")
    embedding = FakeMultimodalEmbedding()
    stats = IngestionPipeline(
        store,
        backends=IngestionBackends(
            asr=FakeASR(),
            ocr=FakeOCR(),
            audio=FakeAudio(),
            embedding=embedding,
        ),
        config=IngestionConfig(
            window_ms=2_000,
            stride_ms=1_000,
            frame_step_ms=1_000,
            audio_window_ms=2_000,
            audio_stride_ms=1_000,
            embedding_dimensions=4,
            embedding_batch_size=2,
        ),
        vector_index=SQLiteVectorIndex(store),
    ).ingest_path(source)

    assert stats["video_embeddings"] == 0
    assert stats["image_embeddings"] == 1
    assert stats["sampled_frames"] >= 2
    assert stats["deduplicated_frames"] == stats["sampled_frames"] - 1
    assert stats["transcript"] == 1
    assert stats["ocr"] >= 2
    assert stats["audio"] >= 2
    assert stats["embedding_vectors"] == (
        stats["video_embeddings"]
        + stats["image_embeddings"]
        + stats["transcript"]
        + stats["ocr"]
    )
    assert store.ingestion_status(
        stats["video_id"], stats["fingerprint"]
    )["status"] == "complete"
    assert store.search_text("ABC 123", modalities=["ocr"])
    assert len(store.embedding_records("visual", embedding.model_name)) == (
        stats["video_embeddings"] + stats["image_embeddings"]
    )
    assert len(store.embedding_records("transcript", embedding.model_name)) == stats["transcript"]
    assert len(store.embedding_records("ocr", embedding.model_name)) == stats["ocr"]
    assert len(store.embedding_records("audio_event", embedding.model_name)) == 0
    assert all(path.endswith(".jpg") for path in embedding.media_inputs)
    assert stats["embedding_cache_hits"] >= 1

    calls_before_retry = (len(embedding.text_inputs), len(embedding.media_inputs))
    retried = IngestionPipeline(
        store,
        backends=IngestionBackends(
            asr=FakeASR(),
            ocr=FakeOCR(),
            audio=FakeAudio(),
            embedding=embedding,
        ),
        config=IngestionConfig(
            window_ms=2_000,
            stride_ms=1_000,
            frame_step_ms=1_000,
            audio_window_ms=2_000,
            audio_stride_ms=1_000,
            embedding_dimensions=4,
            embedding_batch_size=2,
        ),
        vector_index=SQLiteVectorIndex(store),
    ).ingest_path(source, force=True)
    assert (len(embedding.text_inputs), len(embedding.media_inputs)) == calls_before_retry
    assert retried["embedding_cache_misses"] == 0
    assert retried["embedding_cache_hits"] == retried["embedding_vectors"]
    store.close()
