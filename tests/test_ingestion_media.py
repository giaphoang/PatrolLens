from __future__ import annotations

import shutil
import subprocess

import pytest

from patrol_lens.adapters.media import (
    compress_video_480p,
    deduplicate_keyframes,
    extract_audio_segment,
    extract_clip,
    extract_frame,
    iter_segments,
    probe_video,
)
from PIL import Image
from patrol_lens.config import IngestionConfig
from patrol_lens.domain import Evidence, VideoAsset
from patrol_lens.index import IndexStore
from patrol_lens.ingestion import IngestionBackends, IngestionPipeline


def test_ninety_minute_video_uses_overlapping_coarse_windows():
    asset = VideoAsset("video-1", "bodycam.mp4", "hash", 90 * 60 * 1000)
    segments = list(iter_segments(asset, window_ms=16_000, stride_ms=8_000))

    assert len(segments) == 674
    assert segments[0].start_ms == 0
    assert segments[-1].end_ms == asset.duration_ms


def test_metadata_ingestion_is_restartable(tmp_path):
    store = IndexStore(tmp_path / "index")
    pipeline = IngestionPipeline(store, config=IngestionConfig())
    asset = VideoAsset("v1", "/not/read/without/backends.mp4", "hash", 33_000, has_audio=False)

    first = pipeline.ingest_asset(asset)
    second = pipeline.ingest_asset(asset)

    assert first["segments"] == 4
    assert second["skipped"] is True
    store.close()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_separate_480p_compression_is_resumable(tmp_path):
    source = tmp_path / "source-high-resolution.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=1440x1080:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest",
            "-c:v", "libx264", "-c:a", "aac", str(source),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    destination = tmp_path / "compressed_video_corpus" / "source.mp4"

    first = compress_video_480p(
        source,
        destination,
        preset="ultrafast",
    )
    modified_ns = first.stat().st_mtime_ns
    compressed_asset = probe_video(first)
    second = compress_video_480p(source, destination, preset="ultrafast")

    assert first == destination.resolve()
    assert compressed_asset.width <= 854
    assert compressed_asset.height <= 480
    assert compressed_asset.has_audio
    assert second.stat().st_mtime_ns == modified_ns


def test_completed_asset_is_preserved_across_backend_fingerprint_change(tmp_path):
    store = IndexStore(tmp_path / "index")
    asset = VideoAsset("v1", "/not/read/without/backends.mp4", "hash", 33_000, has_audio=False)
    first = IngestionPipeline(store, config=IngestionConfig())
    first_stats = first.ingest_asset(asset)
    store.add_evidence(Evidence("stale", "v1", 0, 1000, "ocr", "OLD", 1.0, "old-model"))

    second = IngestionPipeline(store, config=IngestionConfig(window_ms=8_000, stride_ms=4_000))
    second_stats = second.ingest_asset(asset)

    assert second_stats["skipped"] is True
    assert second_stats["preserved_completed"] is True
    assert store.get_evidence("stale") is not None
    assert store.ingestion_status("v1", first_stats["fingerprint"])["status"] == "complete"

    rebuilt = second.ingest_asset(asset, force=True)

    assert rebuilt["skipped"] is False
    assert store.get_evidence("stale") is None
    assert store.ingestion_status("v1", first_stats["fingerprint"])["status"] == "superseded"
    store.close()


def test_incomplete_asset_reuses_existing_transcript(tmp_path):
    class CountingASR:
        model_name = "openai/whisper-large-v3-turbo"

        def __init__(self):
            self.calls = 0

        def transcribe(self, _audio_path):
            self.calls += 1
            return []

    store = IndexStore(tmp_path / "index")
    asset = VideoAsset("v1", "/not/read.mp4", "hash", 10_000, has_audio=True)
    store.upsert_asset(asset)
    store.add_evidence(
        Evidence("existing-transcript", "v1", 0, 2_000, "transcript", "keep me", 0.9, "faster-whisper")
    )
    audio_path = store.root / "media" / "audio" / "v1.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"existing audio")
    backend = CountingASR()

    stats = IngestionPipeline(
        store,
        backends=IngestionBackends(asr=backend),
        config=IngestionConfig(),
    ).ingest_asset(asset)

    assert stats["transcript"] == 1
    assert stats["transcript_reused"] is True
    assert backend.calls == 0
    assert store.get_evidence("existing-transcript") is not None
    store.close()


def test_adjacent_equivalent_frames_extend_one_keyframe_interval(tmp_path):
    frames = []
    for ordinal, color in enumerate(("red", "red", "red", "blue")):
        path = tmp_path / f"frame-{ordinal}.png"
        Image.new("RGB", (32, 32), color).save(path)
        frames.append((ordinal * 1_000, path))

    keyframes = deduplicate_keyframes(
        frames,
        frame_step_ms=1_000,
        duration_ms=4_000,
    )

    assert len(keyframes) == 2
    assert (keyframes[0].start_ms, keyframes[0].end_ms, keyframes[0].frame_count) == (
        0,
        3_000,
        3,
    )
    assert (keyframes[1].start_ms, keyframes[1].end_ms) == (3_000, 4_000)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_active_media_tools_extract_bounded_artifacts(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=160x120:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest",
            "-c:v", "libx264", "-c:a", "aac", str(source),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    asset = probe_video(source)
    frame = extract_frame(source, 500, tmp_path / "frame.jpg")
    audio = extract_audio_segment(source, 250, 1_250, tmp_path / "audio.wav")
    clip = extract_clip(source, 500, 1_500, tmp_path / "clip.mp4", fps=4)

    assert asset.has_audio
    assert frame.stat().st_size > 0
    assert audio.stat().st_size > 0
    assert clip.stat().st_size > 0
