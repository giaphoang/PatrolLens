from __future__ import annotations

from patrol_lens.adapters.media import iter_segments
from patrol_lens.domain import VideoAsset


def test_ninety_minute_video_uses_overlapping_coarse_windows():
    asset = VideoAsset("video-1", "bodycam.mp4", "hash", 90 * 60 * 1000)
    segments = list(iter_segments(asset, window_ms=16_000, stride_ms=8_000))

    assert len(segments) == 674
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 16_000
    assert segments[-1].end_ms == asset.duration_ms
