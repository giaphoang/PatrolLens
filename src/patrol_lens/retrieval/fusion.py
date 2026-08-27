from __future__ import annotations

import hashlib
from collections import defaultdict

from ..config import RetrievalConfig
from ..domain import CandidateInterval, QueryPlan, RetrievalHit


def _near(left: RetrievalHit, right: RetrievalHit, tolerance_ms: int) -> bool:
    a, b = left.evidence, right.evidence
    return a.start_ms <= b.end_ms + tolerance_ms and b.start_ms <= a.end_ms + tolerance_ms


def _candidate_id(video_id: str, start_ms: int, end_ms: int) -> str:
    digest = hashlib.sha1(f"{video_id}:{start_ms}:{end_ms}".encode()).hexdigest()[:12]
    return f"candidate-{digest}"


def fuse_hits(
    branches: dict[str, list[RetrievalHit]],
    plan: QueryPlan,
    *,
    config: RetrievalConfig | None = None,
    video_durations: dict[str, int] | None = None,
) -> list[CandidateInterval]:
    """Temporal join followed by weighted reciprocal-rank fusion."""

    cfg = config or RetrievalConfig()
    durations = video_durations or {}
    all_hits = [item for values in branches.values() for item in values]
    by_video: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in all_hits:
        by_video[hit.evidence.video_id].append(hit)

    candidates: list[CandidateInterval] = []
    required = set(plan.required_modalities)
    tolerance = plan.relation_tolerance_ms or cfg.temporal_tolerance_ms
    for video_id, video_hits in by_video.items():
        for seed in video_hits:
            selected: dict[str, RetrievalHit] = {seed.branch: seed}
            for branch, hits in branches.items():
                choices = [item for item in hits if item.evidence.video_id == video_id and _near(seed, item, tolerance)]
                if choices:
                    selected[branch] = min(choices, key=lambda item: item.rank)
            evidence = list({item.evidence.id: item.evidence for item in selected.values()}.values())
            covered = {item.modality for item in evidence}
            missing = sorted(required - covered)
            if missing and cfg.require_conjunctive_modalities:
                continue
            raw_start = min(item.start_ms for item in evidence)
            raw_end = max(item.end_ms for item in evidence)
            start_ms = max(0, raw_start - cfg.candidate_padding_ms)
            end_ms = raw_end + cfg.candidate_padding_ms
            duration = durations.get(video_id)
            if duration is not None:
                end_ms = min(duration, end_ms)
            if end_ms - start_ms > cfg.max_candidate_ms:
                center = (raw_start + raw_end) // 2
                start_ms = max(0, center - cfg.max_candidate_ms // 2)
                end_ms = start_ms + cfg.max_candidate_ms
                if duration is not None and end_ms > duration:
                    end_ms = duration
                    start_ms = max(0, end_ms - cfg.max_candidate_ms)

            branch_scores: dict[str, float] = {}
            for branch, hit in selected.items():
                weight = plan.modality_weights.get(hit.evidence.modality, 1.0)
                branch_scores[branch] = weight / (cfg.rrf_constant + hit.rank)
            score = sum(branch_scores.values())
            if required and not missing:
                score *= 1.15
            candidates.append(
                CandidateInterval(
                    id=_candidate_id(video_id, start_ms, end_ms),
                    video_id=video_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    score=score,
                    evidence=sorted(evidence, key=lambda item: (item.start_ms, item.end_ms)),
                    branch_scores=branch_scores,
                    covered_modalities=sorted(covered),
                    missing_modalities=missing,
                    metadata={"relation": plan.relation, "seed_evidence_id": seed.evidence.id},
                )
            )
    return merge_candidates(candidates, gap_ms=cfg.merge_gap_ms)


def _overlap_ratio(left: CandidateInterval, right: CandidateInterval) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    shorter = max(1, min(left.duration_ms, right.duration_ms))
    return intersection / shorter


def merge_candidates(candidates: list[CandidateInterval], *, gap_ms: int = 3_000) -> list[CandidateInterval]:
    """Deduplicate joined seeds without merging distinct repeated events."""

    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    kept: list[CandidateInterval] = []
    for current in ordered:
        duplicate = next(
            (
                item
                for item in kept
                if item.video_id == current.video_id
                and (_overlap_ratio(item, current) >= 0.7 or abs(item.start_ms - current.start_ms) <= gap_ms)
            ),
            None,
        )
        if duplicate is None:
            kept.append(current)
            continue
        duplicate.start_ms = min(duplicate.start_ms, current.start_ms)
        duplicate.end_ms = max(duplicate.end_ms, current.end_ms)
        duplicate.score = max(duplicate.score, current.score)
        duplicate.evidence = sorted(
            {item.id: item for item in [*duplicate.evidence, *current.evidence]}.values(),
            key=lambda item: (item.start_ms, item.end_ms),
        )
        duplicate.branch_scores.update(
            {
                key: max(value, duplicate.branch_scores.get(key, 0.0))
                for key, value in current.branch_scores.items()
            }
        )
        duplicate.covered_modalities = sorted(set(duplicate.covered_modalities) | set(current.covered_modalities))
        duplicate.missing_modalities = sorted(set(duplicate.missing_modalities) & set(current.missing_modalities))
    return sorted(kept, key=lambda item: item.score, reverse=True)
