from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .retrieval.search import CoarseRetriever


def temporal_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def evaluate_file(path: str | Path, retriever: CoarseRetriever, top_k: int = 10) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    details: list[dict[str, Any]] = []
    hits = 0
    best_ious: list[float] = []
    for row in rows:
        _plan, candidates = retriever.retrieve(row["query"])
        candidates = candidates[:top_k]
        expected = row.get("relevant", [])
        best = 0.0
        for video_id, start_ms, end_ms in expected:
            for candidate in candidates:
                if candidate.video_id == video_id:
                    best = max(best, temporal_iou((start_ms, end_ms), (candidate.start_ms, candidate.end_ms)))
        matched = best > 0
        hits += int(matched)
        best_ious.append(best)
        details.append({"query": row["query"], "matched": matched, "best_temporal_iou": best})
    return {
        "queries": len(rows),
        "recall_at_k": hits / len(rows) if rows else 0.0,
        "mean_best_temporal_iou": sum(best_ious) / len(best_ious) if best_ious else 0.0,
        "details": details,
    }
