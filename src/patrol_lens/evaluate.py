from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .retrieval import Retriever


def evaluate_file(path: str | Path, retriever: Retriever, top_k: int = 10) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    query_results = []
    hits = 0
    retrieved = 0
    for row in rows:
        _plan, candidates, _status = retriever.search(row["query"], top_k=top_k, max_rerank=0)
        predicted = {(candidate.segment.video_id, candidate.segment.start_ms, candidate.segment.end_ms) for candidate in candidates}
        expected = {tuple(item) for item in row.get("relevant", [])}
        match = False
        for video_id, start_ms, end_ms in expected:
            if any(candidate.segment.video_id == video_id and candidate.segment.end_ms > start_ms and candidate.segment.start_ms < end_ms for candidate in candidates):
                match = True
                break
        hits += int(match)
        retrieved += len(predicted)
        query_results.append({"query": row["query"], "matched": match, "result_count": len(predicted)})
    return {"queries": len(rows), "queries_with_hit": hits, "hit_rate": hits / len(rows) if rows else 0.0, "results_returned": retrieved, "details": query_results}
