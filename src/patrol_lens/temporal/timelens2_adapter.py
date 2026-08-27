from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import RefinementConfig
from ..domain import VerificationResult


@dataclass(frozen=True)
class GroundedInterval:
    start_ms: int
    end_ms: int
    score: float | None = None


class TimeLens2Adapter:
    """Opt-in subprocess boundary to an independently installed TimeLens2 wrapper.

    The configured command receives `--video`, `--query`, `--start-ms`, and
    `--end-ms`, and must print `{\"intervals_ms\": [[start, end], ...]}`.
    """

    def __init__(
        self,
        command: list[str],
        *,
        acknowledge_restricted_license: bool = False,
        timeout_s: int = 300,
    ) -> None:
        if not command:
            raise ValueError("TimeLens2 command cannot be empty")
        if not acknowledge_restricted_license:
            raise RuntimeError(
                "TimeLens2 is opt-in: acknowledge its top-level academic-only/non-EU license first"
            )
        self.command = command
        self.timeout_s = timeout_s

    def ground(
        self,
        video_path: str | Path,
        query: str,
        start_ms: int,
        end_ms: int,
    ) -> list[GroundedInterval]:
        command = [
            *self.command,
            "--video", str(video_path),
            "--query", query,
            "--start-ms", str(start_ms),
            "--end-ms", str(end_ms),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"TimeLens2 wrapper failed: {result.stderr.strip()[-500:]}")
        try:
            payload = json.loads(result.stdout)
            intervals = payload["intervals_ms"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("TimeLens2 wrapper did not return intervals_ms JSON") from exc
        grounded: list[GroundedInterval] = []
        for item in intervals:
            if not isinstance(item, list) or len(item) < 2:
                continue
            left = max(start_ms, min(end_ms, int(item[0])))
            right = max(left, min(end_ms, int(item[1])))
            score = float(item[2]) if len(item) > 2 else None
            grounded.append(GroundedInterval(left, right, score))
        return grounded


def should_use_timelens2(
    result: VerificationResult,
    config: RefinementConfig | None = None,
) -> bool:
    cfg = config or RefinementConfig()
    return (
        result.status == "supported"
        and (
            result.end_ms - result.start_ms > cfg.max_interval_ms_without_specialist
            or result.confidence < cfg.timelens_confidence_threshold
        )
    )
