from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from ..domain import Segment, VideoAsset


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"{name} is required; install FFmpeg and ensure it is on PATH")
    return resolved


def probe_video(path: str | Path) -> VideoAsset:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    command = [
        _tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)
    ]
    try:
        raw = subprocess.check_output(command, text=True)
        payload = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not inspect video {source}: {exc}") from exc
    video_stream = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), {})
    duration = float(video_stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    rate = video_stream.get("r_frame_rate", "0/1")
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
        duration_ms=max(1, int(duration * 1000)),
        fps=fps,
        width=int(video_stream["width"]) if video_stream.get("width") else None,
        height=int(video_stream["height"]) if video_stream.get("height") else None,
    )


def iter_video_files(path: str | Path) -> Iterator[Path]:
    root = Path(path).expanduser()
    if root.is_file():
        if root.suffix.lower() in VIDEO_EXTENSIONS:
            yield root
        return
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            yield candidate


def iter_segments(asset: VideoAsset, *, window_ms: int = 16_000, stride_ms: int = 8_000, kind: str = "coarse") -> Iterator[Segment]:
    if asset.duration_ms <= 0:
        return
    start = 0
    ordinal = 0
    while start < asset.duration_ms:
        end = min(asset.duration_ms, start + window_ms)
        yield Segment(f"{asset.id}-{kind}-{ordinal:06d}", asset.id, start, end, kind, {"ordinal": ordinal})
        ordinal += 1
        if end >= asset.duration_ms:
            break
        start += stride_ms


def extract_audio(video_path: str | Path, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _tool("ffmpeg"), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return destination


def extract_frame(video_path: str | Path, timestamp_ms: int, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    seconds = max(0, timestamp_ms) / 1000
    command = [
        _tool("ffmpeg"), "-y", "-ss", f"{seconds:.3f}", "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(destination)
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return destination


def extract_clip(video_path: str | Path, start_ms: int, end_ms: int, output_path: str | Path, fps: float | None = None) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, (end_ms - start_ms) / 1000)
    seconds = max(0, start_ms) / 1000
    command = [_tool("ffmpeg"), "-y", "-ss", f"{seconds:.3f}", "-i", str(video_path), "-t", f"{duration:.3f}"]
    if fps:
        command += ["-vf", f"fps={fps}"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-c:a", "aac", "-movflags", "+faststart", str(destination)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return destination
