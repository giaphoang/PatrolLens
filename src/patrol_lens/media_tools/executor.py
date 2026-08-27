from __future__ import annotations

from pathlib import Path

from ..adapters.media import extract_audio_segment, extract_clip, extract_frame
from ..config import AgentConfig
from ..domain import AgentAction, CandidateInterval, ToolObservation, VideoAsset


class ActionValidationError(ValueError):
    pass


class MediaToolExecutor:
    """Bounded FFmpeg implementation of OmniAgent-style perception actions."""

    def __init__(
        self,
        asset: VideoAsset,
        candidate: CandidateInterval,
        run_dir: str | Path,
        *,
        config: AgentConfig | None = None,
    ) -> None:
        self.asset = asset
        self.candidate = candidate
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or AgentConfig()
        self.counter = 0

    def _range(self, action: AgentAction, max_duration_ms: int) -> tuple[int, int]:
        start = self.candidate.start_ms if action.start_ms is None else int(action.start_ms)
        end = self.candidate.end_ms if action.end_ms is None else int(action.end_ms)
        if start < self.candidate.start_ms or end > self.candidate.end_ms:
            raise ActionValidationError(
                f"requested [{start}, {end}] is outside candidate "
                f"[{self.candidate.start_ms}, {self.candidate.end_ms}]"
            )
        if start < 0 or end > self.asset.duration_ms or end <= start:
            raise ActionValidationError("invalid or out-of-bounds media interval")
        if end - start > max_duration_ms:
            raise ActionValidationError(f"requested interval exceeds {max_duration_ms} ms tool budget")
        return start, end

    def execute(self, action: AgentAction) -> ToolObservation:
        self.counter += 1
        prefix = f"turn-{self.counter:02d}"
        if action.type == "answer":
            return ToolObservation(prefix, action, 0, 0, [], action.answer or "")
        if action.type == "get_frames":
            start, end = self._range(action, self.candidate.duration_ms)
            count = action.num_frames
            if count is None:
                fps = action.fps or 2.0
                count = max(1, round((end - start) / 1000 * fps))
            if count < 1 or count > self.config.max_frames_per_action:
                raise ActionValidationError(
                    f"get_frames num_frames must be 1..{self.config.max_frames_per_action}"
                )
            timestamps = [
                round(start + (end - start) * index / max(1, count - 1))
                for index in range(count)
            ]
            paths = []
            for index, timestamp in enumerate(timestamps):
                path = self.run_dir / f"{prefix}-frame-{index:03d}-{timestamp}.jpg"
                extract_frame(self.asset.path, timestamp, path)
                paths.append(str(path))
            return ToolObservation(
                prefix,
                action,
                start,
                end,
                paths,
                metadata={"timestamps_ms": timestamps, "modality": "visual"},
            )
        if action.type == "get_audio":
            if not self.asset.has_audio:
                raise ActionValidationError("video has no audio stream")
            start, end = self._range(action, self.config.max_audio_ms)
            path = self.run_dir / f"{prefix}-audio-{start}-{end}.wav"
            extract_audio_segment(self.asset.path, start, end, path)
            return ToolObservation(
                prefix, action, start, end, [str(path)], metadata={"modality": "audio"}
            )
        if action.type == "get_clip":
            start, end = self._range(action, self.config.max_clip_ms)
            path = self.run_dir / f"{prefix}-clip-{start}-{end}.mp4"
            extract_clip(self.asset.path, start, end, path, fps=action.fps)
            return ToolObservation(
                prefix,
                action,
                start,
                end,
                [str(path)],
                metadata={"modality": "audiovisual" if self.asset.has_audio else "visual"},
            )
        raise ActionValidationError(f"unsupported action: {action.type}")
