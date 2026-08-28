from __future__ import annotations

import json
import os

import pytest

from patrol_lens.config import IngestionConfig
from patrol_lens.domain import VideoAsset
from patrol_lens.ingestion.planning import (
    IngestionCostRates,
    build_cost_report,
    estimate_video_cost,
    prioritize_oldest,
    select_pending_batch,
)


def test_video_priority_uses_oldest_modification_time(tmp_path):
    newer = tmp_path / "a.mp4"
    older = tmp_path / "z.mp4"
    newer.write_bytes(b"new")
    older.write_bytes(b"old")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    assert prioritize_oldest([newer, older]) == [older.resolve(), newer.resolve()]


def test_cost_estimate_separates_asr_text_images_and_local_clap(tmp_path):
    asset = VideoAsset(
        "video-1",
        str(tmp_path / "video.mp4"),
        "hash",
        3_600_000,
        width=854,
        height=480,
        has_audio=True,
    )
    estimate = estimate_video_cost(
        asset,
        artifact_root=tmp_path / "index",
        config=IngestionConfig(frame_step_ms=1_000),
        scheduling_state="pending",
        asr_enabled=True,
        remote_asr_enabled=True,
        embedding_enabled=True,
        image_embedding_enabled=True,
        clap_enabled=True,
        rates=IngestionCostRates(),
    )

    assert estimate["asr"]["estimated_cost_usd"] == 0.04
    assert estimate["text_embeddings"]["estimated_tokens"] == 14_400
    assert estimate["image_embeddings"]["estimated_images"] == 3_600
    assert estimate["image_embeddings"]["estimated_tiles_per_image"] == 6
    assert estimate["local_clap"]["estimated_remote_cost_usd"] == 0
    assert estimate["incremental_estimated_cost_usd"] == pytest.approx(2.55064)


def test_existing_keyframe_manifest_tightens_image_estimate(tmp_path):
    asset = VideoAsset(
        "video-1",
        str(tmp_path / "video.mp4"),
        "hash",
        60_000,
        width=320,
        height=180,
        has_audio=False,
    )
    config = IngestionConfig(frame_step_ms=1_000)
    manifest_path = tmp_path / "media" / "keyframes" / asset.id / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "video_sha256": asset.sha256,
                "frame_step_ms": config.frame_step_ms,
                "duplicate_distance": config.visual_duplicate_distance,
                "scene_change_distance": config.visual_scene_change_distance,
                "keyframes": [{"path": "one.jpg"}, {"path": "two.jpg"}],
            }
        )
    )

    estimate = estimate_video_cost(
        asset,
        artifact_root=tmp_path,
        config=config,
        scheduling_state="pending",
        asr_enabled=False,
        remote_asr_enabled=False,
        embedding_enabled=True,
        image_embedding_enabled=True,
        clap_enabled=False,
        rates=IngestionCostRates(),
    )

    assert estimate["image_embeddings"]["estimated_images"] == 2
    assert estimate["image_embeddings"]["estimate_basis"] == "existing_keyframe_manifest"


def test_batch_skips_completed_entries_and_report_totals_selected_cost(tmp_path):
    entries = [
        {
            "video_id": "complete",
            "scheduling_state": "complete",
            "cost_estimate": {
                "gross_estimated_cost_usd": 2.0,
                "incremental_estimated_cost_usd": 0.0,
            },
        },
        {
            "video_id": "oldest-pending",
            "scheduling_state": "retry_failed",
            "cost_estimate": {
                "gross_estimated_cost_usd": 1.0,
                "incremental_estimated_cost_usd": 0.5,
            },
        },
        {
            "video_id": "later-pending",
            "scheduling_state": "pending",
            "cost_estimate": {
                "gross_estimated_cost_usd": 1.0,
                "incremental_estimated_cost_usd": 0.75,
            },
        },
    ]
    selected = select_pending_batch(entries, 1)
    selected[0]["selected_for_batch"] = True
    report = build_cost_report(
        entries,
        input_root=tmp_path,
        artifact_root=tmp_path / "index",
        video_batch_size=1,
        rates=IngestionCostRates(),
    )

    assert [entry["video_id"] for entry in selected] == ["oldest-pending"]
    assert report["summary"]["pending_video_count"] == 2
    assert report["summary"]["selected_batch_estimated_cost_usd"] == 0.5
    assert report["summary"]["pending_corpus_estimated_cost_usd"] == 1.25
    assert report["summary"]["remaining_after_selected_batch_estimated_cost_usd"] == 0.75
