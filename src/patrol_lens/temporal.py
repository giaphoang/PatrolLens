from __future__ import annotations

from .domain import Candidate, Segment


def expand_segment(segment: Segment, padding_ms: int, duration_ms: int | None = None) -> Segment:
    start = max(0, segment.start_ms - padding_ms)
    end = segment.end_ms + padding_ms
    if duration_ms is not None:
        end = min(duration_ms, end)
    return Segment(segment.id, segment.video_id, start, end, segment.kind, dict(segment.metadata))


def merge_candidates(candidates: list[Candidate], *, gap_ms: int = 2000) -> list[Candidate]:
    """Merge adjacent evidence windows within the same video."""
    ordered = sorted(candidates, key=lambda item: (item.segment.video_id, item.segment.start_ms))
    merged: list[Candidate] = []
    for current in ordered:
        if not merged:
            merged.append(current)
            continue
        previous = merged[-1]
        same_video = previous.segment.video_id == current.segment.video_id
        close = current.segment.start_ms <= previous.segment.end_ms + gap_ms
        if not (same_video and close):
            merged.append(current)
            continue
        previous.segment = Segment(
            previous.segment.id,
            previous.segment.video_id,
            min(previous.segment.start_ms, current.segment.start_ms),
            max(previous.segment.end_ms, current.segment.end_ms),
            previous.segment.kind,
            {**previous.segment.metadata, "merged_segment_ids": [previous.segment.id, current.segment.id]},
        )
        previous.score = max(previous.score, current.score)
        if previous.rerank_score is None or (current.rerank_score is not None and current.rerank_score > previous.rerank_score):
            previous.rerank_score = current.rerank_score
        if previous.confidence is None or (current.confidence is not None and current.confidence > previous.confidence):
            previous.confidence = current.confidence
        previous.modality_scores.update(current.modality_scores)
        seen = {observation.id for observation in previous.evidence}
        previous.evidence.extend(observation for observation in current.evidence if observation.id not in seen)
        previous.warnings.extend(warning for warning in current.warnings if warning not in previous.warnings)
    return merged
