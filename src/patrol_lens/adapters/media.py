from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from ..domain import Segment, VideoAsset

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def _tool(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"{name} is required; install FFmpeg and ensure it is on PATH")
    return resolved


def _run(command: list[str], *, label: str) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        raise RuntimeError(f"{label} failed: {detail}")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: str | Path) -> VideoAsset:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    command = [
        _tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)
    ]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not inspect video {source}: {exc}") from exc
    streams = payload.get("streams", [])
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), {})
    duration = float(video_stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    rate = str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1")
    try:
        numerator, denominator = rate.split("/")
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = None
    digest = sha256_file(source)
    return VideoAsset(
        id=f"video-{digest[:16]}",
        path=str(source),
        sha256=digest,
        duration_ms=max(1, round(duration * 1000)),
        fps=fps,
        width=int(video_stream["width"]) if video_stream.get("width") else None,
        height=int(video_stream["height"]) if video_stream.get("height") else None,
        has_audio=any(item.get("codec_type") == "audio" for item in streams),
        metadata={"container": payload.get("format", {}).get("format_name")},
    )


def iter_video_files(path: str | Path) -> Iterator[Path]:
    root = Path(path).expanduser()
    if root.is_file():
        if root.suffix.lower() in VIDEO_EXTENSIONS:
            yield root.resolve()
        return
    if not root.exists():
        raise FileNotFoundError(root)
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            yield candidate.resolve()


def iter_segments(
    asset: VideoAsset,
    *,
    window_ms: int = 16_000,
    stride_ms: int = 8_000,
    kind: str = "coarse",
) -> Iterator[Segment]:
    if window_ms <= 0 or stride_ms <= 0:
        raise ValueError("window_ms and stride_ms must be positive")
    start_ms = 0
    ordinal = 0
    while start_ms < asset.duration_ms:
        end_ms = min(asset.duration_ms, start_ms + window_ms)
        yield Segment(
            id=f"{asset.id}-{kind}-{ordinal:06d}",
            video_id=asset.id,
            start_ms=start_ms,
            end_ms=end_ms,
            kind=kind,
            metadata={"ordinal": ordinal},
        )
        ordinal += 1
        if end_ms >= asset.duration_ms:
            break
        start_ms += stride_ms


def extract_audio(video_path: str | Path, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _tool("ffmpeg"), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(destination),
        ],
        label="audio extraction",
    )
    return destination


def extract_audio_segment(
    video_path: str | Path,
    start_ms: int,
    end_ms: int,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration_s = max(0.1, (end_ms - start_ms) / 1000)
    _run(
        [
            _tool("ffmpeg"), "-y", "-ss", f"{max(0, start_ms) / 1000:.3f}", "-i", str(video_path),
            "-t", f"{duration_s:.3f}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(destination),
        ],
        label="audio segment extraction",
    )
    return destination


def extract_frame(video_path: str | Path, timestamp_ms: int, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _tool("ffmpeg"), "-y", "-ss", f"{max(0, timestamp_ms) / 1000:.3f}",
            "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(destination),
        ],
        label="frame extraction",
    )
    return destination


def extract_frame_sequence(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    step_ms: int = 1_000,
    start_ms: int = 0,
    end_ms: int | None = None,
) -> list[tuple[int, Path]]:
    """Decode a long video once and return `(timestamp_ms, frame_path)` records."""

    if step_ms <= 0:
        raise ValueError("step_ms must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pattern = destination / "frame-%08d.jpg"
    command = [
        _tool("ffmpeg"), "-y", "-ss", f"{max(0, start_ms) / 1000:.3f}", "-i", str(video_path),
    ]
    if end_ms is not None:
        command += ["-t", f"{max(0.1, (end_ms - start_ms) / 1000):.3f}"]
    command += ["-vf", f"fps={1000 / step_ms:.8f}", "-q:v", "3", str(pattern)]
    _run(command, label="frame sequence extraction")
    paths = sorted(destination.glob("frame-*.jpg"))
    return [(start_ms + ordinal * step_ms, path) for ordinal, path in enumerate(paths)]


def extract_clip(
    video_path: str | Path,
    start_ms: int,
    end_ms: int,
    output_path: str | Path,
    fps: float | None = None,
    max_width: int | None = 960,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration_s = max(0.1, (end_ms - start_ms) / 1000)
    command = [
        _tool("ffmpeg"), "-y", "-ss", f"{max(0, start_ms) / 1000:.3f}", "-i", str(video_path),
        "-t", f"{duration_s:.3f}",
    ]
    filters: list[str] = []
    if fps:
        filters.append(f"fps={fps}")
    if max_width:
        filters.append(f"scale={max_width}:-2:force_original_aspect_ratio=decrease")
    if filters:
        command += ["-vf", ",".join(filters)]
    command += [
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart", str(destination),
    ]
    _run(command, label="clip extraction")
    return destination
