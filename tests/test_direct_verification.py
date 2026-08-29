from __future__ import annotations

import json

import patrol_lens.verification.direct_media as direct_media_module
from patrol_lens.domain import CandidateInterval, Evidence, QueryPlan, VideoAsset
from patrol_lens.verification import DirectCandidateMedia, DirectMediaConfig


def test_direct_candidate_media_extracts_one_bounded_clip(monkeypatch, tmp_path):
    calls = []

    def fake_extract(video_path, start_ms, end_ms, output_path, fps=None, max_width=None):
        calls.append((video_path, start_ms, end_ms, output_path, fps, max_width))
        output_path.write_bytes(b"clip")
        return output_path

    monkeypatch.setattr(direct_media_module, "extract_clip", fake_extract)
    candidate = CandidateInterval(
        "candidate-1",
        "video-1",
        10_000,
        80_000,
        evidence=[
            Evidence(
                "evidence-1",
                "video-1",
                58_000,
                60_000,
                "transcript",
                "laughter begins",
                0.9,
                "asr",
            )
        ],
    )
    provider = DirectCandidateMedia(
        tmp_path / "runs",
        config=DirectMediaConfig(max_clip_ms=20_000, fps=3.0, max_width=720),
    )

    context = provider.prepare(
        "find laughter onset",
        QueryPlan("find laughter onset", required_modalities=["visual", "audio_event"]),
        candidate,
        VideoAsset("video-1", "/videos/source.mp4", "hash", 90_000, has_audio=True),
    )

    assert len(calls) == 1
    assert (context.start_ms, context.end_ms) == (49_000, 69_000)
    assert context.direct_modalities == frozenset({"visual", "audio", "audiovisual"})
    assert calls[0][4:] == (3.0, 720)
    manifest = json.loads((context.workspace / "context.json").read_text())
    assert manifest["verification_interval_ms"] == [49_000, 69_000]
    assert manifest["media_paths"] == list(context.media_paths)
