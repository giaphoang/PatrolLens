from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from statistics import median
from typing import Any, Protocol

from ..adapters.media import extract_audio_segment, extract_clip
from ..config import RefinementConfig
from ..domain import CandidateInterval, QueryPlan, VerificationResult, VideoAsset


class JSONGenerator(Protocol):
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        media_paths: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]: ...


REFINEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start_ms": {"type": "integer"},
        "end_ms": {"type": "integer"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "boundary_basis": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["start_ms", "end_ms", "confidence", "boundary_basis"],
    "additionalProperties": False,
}


def _window_db(samples: tuple[int, ...]) -> float:
    if not samples:
        return -80.0
    rms = math.sqrt(sum(item * item for item in samples) / len(samples)) / 32768
    return 20 * math.log10(max(rms, 1e-6))


def detect_audio_onset(wav_path: str | Path, absolute_start_ms: int) -> int | None:
    """Find a sustained relative loudness rise; this is only a boundary cue."""

    try:
        with wave.open(str(wav_path), "rb") as handle:
            if handle.getsampwidth() != 2:
                return None
            rate = handle.getframerate()
            channels = handle.getnchannels()
            raw = handle.readframes(handle.getnframes())
    except (OSError, wave.Error):
        return None
    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if channels > 1:
        values = values[::channels]
    size = max(1, rate // 5)  # 200 ms
    levels = [_window_db(values[offset : offset + size]) for offset in range(0, len(values), size)]
    if len(levels) < 4:
        return None
    baseline = median(levels[: max(3, len(levels) // 4)])
    threshold = max(-26.0, baseline + 7.0)
    for index in range(1, len(levels) - 1):
        if levels[index] >= threshold and levels[index + 1] >= threshold:
            return absolute_start_ms + index * 200
    return None


class LightweightTimestampRefiner:
    def __init__(
        self,
        client: JSONGenerator | None,
        *,
        model: str | None = None,
        config: RefinementConfig | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.config = config or RefinementConfig()

    def refine(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        verification: VerificationResult,
        workspace: str | Path,
    ) -> VerificationResult:
        if verification.status != "supported":
            return verification
        start = max(candidate.start_ms, verification.start_ms - self.config.context_ms)
        end = min(candidate.end_ms, verification.end_ms + self.config.context_ms)
        if end - start > 20_000:
            center = (verification.start_ms + verification.end_ms) // 2
            start = max(candidate.start_ms, center - 10_000)
            end = min(candidate.end_ms, start + 20_000)
        run_dir = Path(workspace).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        clip_path = run_dir / f"refine-{start}-{end}.mp4"
        extract_clip(asset.path, start, end, clip_path, fps=self.config.frame_fps)
        media = [str(clip_path)]
        onset: int | None = None
        if asset.has_audio:
            audio_path = run_dir / f"refine-{start}-{end}.wav"
            extract_audio_segment(asset.path, start, end, audio_path)
            media.append(str(audio_path))
            if plan.target == "onset" and "audio_event" in plan.required_modalities:
                onset = detect_audio_onset(audio_path, start)
        if self.client is None:
            if onset is None:
                return verification
            return VerificationResult(
                **{
                    **verification.to_dict(),
                    "start_ms": onset,
                    "warnings": [*verification.warnings, "start_refined_from_audio_onset"],
                }
            )
        prompt = f"""Refine only the temporal boundaries of an already-supported body-camera event.
Return absolute milliseconds inside [{start}, {end}]. The start is the first frame/audio instant
that satisfies the query; the end is the last supporting instant. Do not broaden to surrounding
context. Initial interval: [{verification.start_ms}, {verification.end_ms}].
Query: {query}
Deterministic relative-loudness onset cue (not semantic proof): {onset}"""
        data = self.client.generate_json(
            prompt,
            REFINEMENT_SCHEMA,
            media_paths=media,
            model=self.model,
        )
        refined_start = max(start, min(end, int(data.get("start_ms", verification.start_ms))))
        refined_end = max(refined_start, min(end, int(data.get("end_ms", verification.end_ms))))
        warnings = list(verification.warnings)
        warnings.extend(str(item) for item in data.get("boundary_basis", []))
        return VerificationResult(
            status=verification.status,
            event_description=verification.event_description,
            start_ms=refined_start,
            end_ms=refined_end,
            confidence=min(
                verification.confidence,
                max(0.0, min(1.0, float(data.get("confidence", verification.confidence)))),
            ),
            evidence=verification.evidence,
            missing_evidence=verification.missing_evidence,
            warnings=warnings,
        )
