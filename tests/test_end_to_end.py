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
    model_name = "fake-paddleocr"

    def detect(self, _image_path):
        return [{"text": "ABC 123", "confidence": 0.95, "box": [[0, 0], [10, 10]]}]


class FakeAudio:
    model_name = "fake-yamnet-prosody"

    def analyze(self, _audio_path, _start_ms, _end_ms):
        return AudioAnalysis(-15.0, 0.9, 180.0, ["elevated vocal intensity"], 0.88)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_synthetic_ingestion_to_multimodal_retrieval(tmp_path):
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
            ocr=FakeOCR(),
            audio=FakeAudio(),
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

    _plan, candidates = retriever.retrieve("Find when the person in the red shirt started shouting")

    assert stats["visual"] >= 2
    assert stats["transcript"] == 1
    assert stats["ocr"] >= 2
    assert stats["audio"] >= 2
    assert candidates
    assert set(candidates[0].covered_modalities) >= {"visual", "audio_event"}
    store.close()
