from __future__ import annotations

import shutil
import subprocess

import pytest

from patrol_lens.adapters.clap import (
    CLAP_DIMENSIONS,
    ClapCoreMLBackend,
    ClapEmbeddingDimensionError,
    _normalized_vector,
    clap_intervals,
)
from patrol_lens.config import IngestionConfig, RetrievalConfig
from patrol_lens.domain import EmbeddingRecord, Evidence, QueryPlan, VideoAsset
from patrol_lens.index import IndexStore, SQLiteVectorIndex
from patrol_lens.ingestion import IngestionBackends, IngestionPipeline
from patrol_lens.retrieval import CoarseRetriever


class FakeClap:
    model_name = "fake/larger_clap_general_coreml"
    dimensions = CLAP_DIMENSIONS

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.audio_calls: list[tuple[int, int]] = []
        self.text_calls: list[str] = []

    def encode_audio_windows(self, _media_path, intervals):
        for interval in intervals:
            if self.fail_after is not None and len(self.audio_calls) >= self.fail_after:
                raise RuntimeError("synthetic CLAP interruption")
            self.audio_calls.append(interval)
            yield [1.0, *([0.0] * (CLAP_DIMENSIONS - 1))]

    def encode_text(self, text):
        self.text_calls.append(text)
        return [1.0, *([0.0] * (CLAP_DIMENSIONS - 1))]


def test_clap_two_hour_window_count_and_final_boundary():
    intervals = clap_intervals(2 * 60 * 60 * 1000)

    assert len(intervals) == 1_439
    assert intervals[0] == (0, 10_000)
    assert intervals[-1] == (7_190_000, 7_200_000)


def test_clap_vector_guard_normalizes_and_rejects_wrong_dimensions():
    vector = _normalized_vector([2.0, *([0.0] * (CLAP_DIMENSIONS - 1))])

    assert vector[0] == 1.0
    assert sum(value * value for value in vector) == pytest.approx(1.0)
    with pytest.raises(ClapEmbeddingDimensionError):
        _normalized_vector([1.0, 0.0])


def test_completed_video_gets_additive_clap_backfill_without_reindexing(tmp_path):
    store = IndexStore(tmp_path / "index")
    asset = VideoAsset("video-1", "/not/read-by-fake.mp4", "sha", 20_000)
    original = IngestionPipeline(store).ingest_asset(asset)
    store.add_evidence(
        Evidence("keep", asset.id, 0, 1_000, "transcript", "keep me", 1.0, "asr")
    )
    clap = FakeClap()

    stats = IngestionPipeline(
        store,
        backends=IngestionBackends(audio_embedding=clap),
    ).ingest_asset(asset)

    assert original["skipped"] is False
    assert stats["additive_clap_backfill"] is True
    assert stats["audio_embeddings"] == 3
    assert len(clap.audio_calls) == 3
    assert store.get_evidence("keep") is not None
    assert len(store.embedding_records("audio_event", clap.model_name)) == 3
    store.close()


def test_clap_retry_reuses_each_checkpointed_window(tmp_path):
    store = IndexStore(tmp_path / "index")
    asset = VideoAsset("video-1", "/not/read-by-fake.mp4", "sha", 20_000)
    interrupted = FakeClap(fail_after=1)
    pipeline = IngestionPipeline(
        store,
        backends=IngestionBackends(audio_embedding=interrupted),
    )

    with pytest.raises(RuntimeError, match="synthetic CLAP interruption"):
        pipeline.ingest_asset(asset)

    resumed = FakeClap()
    stats = IngestionPipeline(
        store,
        backends=IngestionBackends(audio_embedding=resumed),
    ).ingest_asset(asset)

    assert stats["clap_cache_hits"] == 1
    assert stats["clap_cache_misses"] == 2
    assert resumed.audio_calls == [(5_000, 15_000), (10_000, 20_000)]
    assert len(store.embedding_records("audio_event", resumed.model_name)) == 3
    store.close()


def test_audio_query_uses_clap_text_vector_branch(tmp_path):
    store = IndexStore(tmp_path / "index")
    asset = VideoAsset("video-1", "/tmp/bodycam.mp4", "sha", 60_000)
    evidence = Evidence(
        "audio-1",
        asset.id,
        20_000,
        30_000,
        "audio_event",
        "CLAP acoustic window",
        1.0,
        FakeClap.model_name,
    )
    store.upsert_asset(asset)
    store.add_evidence_and_embeddings(
        [evidence],
        [
            EmbeddingRecord(
                "audio-embedding-1",
                evidence.id,
                "audio_event",
                FakeClap.model_name,
                [1.0, *([0.0] * (CLAP_DIMENSIONS - 1))],
            )
        ],
    )
    clap = FakeClap()
    retriever = CoarseRetriever(
        store,
        audio_encoder=clap,
        vector_index=SQLiteVectorIndex(store),
        config=RetrievalConfig(top_k=3),
    )
    plan = QueryPlan(
        "siren",
        audio_queries=["an emergency vehicle siren"],
        required_modalities=["audio_event"],
    )

    candidates = retriever.retrieve_plan(plan)

    assert candidates
    assert "audio_event:0:clap" in candidates[0].branch_scores
    assert clap.text_calls == ["an emergency vehicle siren"]
    store.close()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
def test_coreml_adapter_streams_48khz_fixed_windows(tmp_path):
    source = tmp_path / "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=12",
            str(source),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    audio_model = tmp_path / "clap_audio_encoder.mlpackage"
    audio_model.mkdir()
    backend = ClapCoreMLBackend(audio_model, tmp_path / "unused.onnx", tmp_path / "unused")
    shapes = []

    class FakeModel:
        def predict(self, inputs):
            shapes.append(inputs["audio"].shape)
            return {
                "embedding": [1.0, *([0.0] * (CLAP_DIMENSIONS - 1))]
            }

    backend._audio_model = FakeModel()
    vectors = list(backend.encode_audio_windows(source, [(0, 10_000), (5_000, 12_000)]))

    assert len(vectors) == 2
    assert shapes == [(1, 480_000), (1, 480_000)]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
def test_clap_decode_failure_never_reaches_model(tmp_path):
    audio_model = tmp_path / "clap_audio_encoder.mlpackage"
    audio_model.mkdir()
    backend = ClapCoreMLBackend(audio_model, tmp_path / "unused.onnx", tmp_path / "unused")

    class FailIfCalled:
        def predict(self, _inputs):
            raise AssertionError("invalid decoded audio must not be embedded")

    backend._audio_model = FailIfCalled()

    with pytest.raises(RuntimeError, match="CLAP audio decoding failed"):
        list(backend.encode_audio_windows(tmp_path / "missing.wav", [(0, 10_000)]))
