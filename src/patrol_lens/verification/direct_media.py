from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

from ..adapters.media import extract_clip
from ..domain import CandidateInterval, QueryPlan, VideoAsset

if TYPE_CHECKING:
    from ..history import TrajectoryRecorder


@dataclass(frozen=True)
class DirectVerificationContext:
    """One bounded media observation and its replayable workspace."""

    workspace: Path
    media_paths: tuple[str, ...]
    start_ms: int
    end_ms: int
    direct_modalities: frozenset[str]


@dataclass(frozen=True)
class DirectMediaConfig:
    max_clip_ms: int = 45_000
    fps: float = 4.0
    max_width: int = 854

    def __post_init__(self) -> None:
        if self.max_clip_ms <= 0:
            raise ValueError("direct verification clip duration must be positive")
        if self.fps <= 0:
            raise ValueError("direct verification clip FPS must be positive")
        if self.max_width <= 0:
            raise ValueError("direct verification clip width must be positive")


class DirectCandidateMedia:
    """Prepare exactly one candidate clip for direct multimodal verification."""

    def __init__(
        self,
        run_root: str | Path,
        *,
        config: DirectMediaConfig | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.run_root = Path(run_root).expanduser().resolve()
        self.config = config or DirectMediaConfig()
        self.recorder = recorder

    def _interval(self, candidate: CandidateInterval) -> tuple[int, int]:
        if candidate.duration_ms <= self.config.max_clip_ms:
            return candidate.start_ms, candidate.end_ms
        centers = [
            (item.start_ms + item.end_ms) // 2
            for item in candidate.evidence
            if candidate.start_ms <= item.start_ms <= candidate.end_ms
        ]
        center = int(median(centers)) if centers else (candidate.start_ms + candidate.end_ms) // 2
        start = max(candidate.start_ms, center - self.config.max_clip_ms // 2)
        end = min(candidate.end_ms, start + self.config.max_clip_ms)
        start = max(candidate.start_ms, end - self.config.max_clip_ms)
        return start, end

    def prepare(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        *,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> DirectVerificationContext:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("candidate verification cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("candidate verification deadline reached")

        start_ms, end_ms = self._interval(candidate)
        workspace = self.run_root / f"direct-{candidate.id}-{uuid.uuid4().hex[:10]}"
        workspace.mkdir(parents=True, exist_ok=False)
        clip_path = workspace / f"candidate-{start_ms}-{end_ms}.mp4"
        event_id: str | None = None
        if self.recorder:
            event_id = self.recorder.emit(
                "verification_media_started",
                stage="direct_verification",
                status="started",
                input_summary={
                    "video_id": asset.id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "fps": self.config.fps,
                },
            )
        try:
            extract_clip(
                asset.path,
                start_ms,
                end_ms,
                clip_path,
                fps=self.config.fps,
                max_width=self.config.max_width,
            )
        except Exception as exc:
            if self.recorder:
                self.recorder.emit(
                    "verification_media_failed",
                    stage="direct_verification",
                    parent_id=event_id,
                    status="failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            raise

        direct_modalities = {"visual"}
        if asset.has_audio:
            direct_modalities.update(("audio", "audiovisual"))
        context = DirectVerificationContext(
            workspace=workspace,
            media_paths=(str(clip_path),),
            start_ms=start_ms,
            end_ms=end_ms,
            direct_modalities=frozenset(direct_modalities),
        )
        manifest = {
            "schema_version": 1,
            "query": query,
            "query_plan": plan.to_dict(),
            "candidate": candidate.to_dict(),
            "video_path": asset.path,
            "media_paths": list(context.media_paths),
            "verification_interval_ms": [start_ms, end_ms],
            "direct_modalities": sorted(context.direct_modalities),
        }
        temporary = workspace / "context.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(workspace / "context.json")
        if self.recorder:
            self.recorder.emit(
                "verification_media_completed",
                stage="direct_verification",
                parent_id=event_id,
                status="completed",
                output_summary={
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "direct_modalities": sorted(context.direct_modalities),
                },
                media_references=list(context.media_paths),
            )
        return context
