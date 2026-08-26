from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

from .domain import Candidate, Observation, QueryPlan, RerankDecision, Segment, result_dict
from .query import plan_query
from .storage import IndexStore
from .temporal import merge_candidates


class TextEncoder(Protocol):
    model_name: str

    def encode_text(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidate: Candidate, asset_path: str | None = None) -> RerankDecision: ...


def _rrf(rank: int, weight: float, constant: int = 60) -> float:
    return weight / (constant + rank)


class Retriever:
    def __init__(
        self,
        store: IndexStore,
        *,
        text_encoder: TextEncoder | None = None,
        visual_encoder: TextEncoder | None = None,
        clip_encoder: TextEncoder | None = None,
        audio_encoder: TextEncoder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder or text_encoder
        self.clip_encoder = clip_encoder
        self.audio_encoder = audio_encoder
        self.reranker = reranker

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        retrieve_k: int = 100,
        max_rerank: int = 20,
        merge_gap_ms: int = 2000,
    ) -> tuple[QueryPlan, list[Candidate], str]:
        plan = plan_query(query)
        candidates: dict[str, Candidate] = {}
        branches: list[tuple[str, list[tuple[str, float, Any]], float]] = []

        branches.append(("text", self.store.search_text(query, retrieve_k, modality="text"), plan.modality_weights.get("text", 1.0)))
        branches.append(("clip_text", self.store.search_text(query, retrieve_k, modality="clip"), plan.modality_weights.get("clip", 1.0)))
        if plan.ocr_terms:
            branches.append(("ocr", self.store.search_text(query, retrieve_k, modality="ocr"), plan.modality_weights.get("ocr", 0.5)))

        if self.text_encoder:
            vector = self.text_encoder.encode_text(query)
            text_hits = [(segment_id, score, None) for segment_id, score in self.store.search_vectors(vector, modality="text", model=self.text_encoder.model_name, limit=retrieve_k)]
            branches.append(("text_vector", text_hits, plan.modality_weights.get("text", 1.0)))
            if plan.ocr_terms:
                ocr_hits = [(segment_id, score, None) for segment_id, score in self.store.search_vectors(vector, modality="ocr", model=self.text_encoder.model_name, limit=retrieve_k)]
                branches.append(("ocr_vector", ocr_hits, plan.modality_weights.get("ocr", 0.5)))

        if self.visual_encoder:
            vector = self.visual_encoder.encode_text(query)
            visual_hits = [(segment_id, score, None) for segment_id, score in self.store.search_vectors(vector, modality="visual", model=self.visual_encoder.model_name, limit=retrieve_k)]
            branches.append(("visual", visual_hits, plan.modality_weights.get("visual", 1.0)))

        if self.clip_encoder:
            vector = self.clip_encoder.encode_text(query)
            clip_hits = [(segment_id, score, None) for segment_id, score in self.store.search_vectors(vector, modality="clip", model=self.clip_encoder.model_name, limit=retrieve_k)]
            branches.append(("clip", clip_hits, plan.modality_weights.get("clip", 1.0)))

        if plan.audio_intent:
            audio_hits = self.store.search_label("audio", plan.audio_intent, retrieve_k)
            branches.append(("audio", audio_hits, plan.modality_weights.get("audio", 0.5)))
        if self.audio_encoder:
            vector = self.audio_encoder.encode_text(query)
            audio_vector_hits = [(segment_id, score, None) for segment_id, score in self.store.search_vectors(vector, modality="audio", model=self.audio_encoder.model_name, limit=retrieve_k)]
            branches.append(("audio_vector", audio_vector_hits, plan.modality_weights.get("audio", 0.5)))

        branch_ranks: dict[str, dict[str, float]] = defaultdict(dict)
        for branch_name, hits, weight in branches:
            for rank, (segment_id, _score, evidence) in enumerate(hits, start=1):
                segment = self.store.get_segment(segment_id)
                if segment is None:
                    continue
                candidate = candidates.setdefault(segment_id, Candidate(segment=segment))
                contribution = _rrf(rank, weight)
                candidate.score += contribution
                candidate.modality_scores[branch_name] = max(candidate.modality_scores.get(branch_name, 0.0), contribution)
                branch_ranks[branch_name][segment_id] = contribution
                if evidence is not None and evidence.id not in {item.id for item in candidate.evidence}:
                    candidate.evidence.append(evidence)

        for segment_id, candidate in candidates.items():
            for observation in self.store.get_observations(segment_id):
                if observation.id not in {item.id for item in candidate.evidence} and observation.modality in {"audio", "ocr", "text"}:
                    candidate.evidence.append(observation)

        ordered = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        rerank_status = "not_requested"
        if self.reranker and ordered:
            rerank_status = "complete"
            for candidate in ordered[:max_rerank]:
                asset = self.store.get_asset(candidate.segment.video_id)
                try:
                    decision = self.reranker.rerank(query, candidate, asset.path if asset else None)
                except Exception as exc:  # provider failures must not erase local results
                    candidate.warnings.append(f"reranker_unavailable: {exc}")
                    rerank_status = "partial"
                    continue
                if decision.match == "no":
                    candidate.warnings.append("reranker_rejected")
                    candidate.rerank_score = 0.0
                    continue
                candidate.rerank_score = decision.confidence
                candidate.confidence = decision.confidence
                if decision.warning:
                    candidate.warnings.append(decision.warning)
                if decision.evidence:
                    candidate.warnings.append("reranker_evidence_validated")
                    for index, item in enumerate(decision.evidence):
                        modality = item.get("modality")
                        offset = item.get("offset_ms")
                        claim = item.get("claim")
                        if modality not in {"visual", "clip", "audio", "text", "ocr"} or not isinstance(offset, int) or not isinstance(claim, str):
                            continue
                        if 0 <= offset <= candidate.segment.end_ms - candidate.segment.start_ms:
                            evidence = Observation(
                                id=f"{candidate.segment.id}-rerank-{index}",
                                segment_id=candidate.segment.id,
                                video_id=candidate.segment.video_id,
                                modality=modality,
                                start_ms=candidate.segment.start_ms + offset,
                                end_ms=min(candidate.segment.end_ms, candidate.segment.start_ms + offset + 1000),
                                text=claim if modality in {"text", "ocr"} else None,
                                label=claim if modality in {"visual", "clip", "audio"} else None,
                                confidence=decision.confidence,
                                metadata={"source": "openrouter", "relative_offset_ms": offset},
                            )
                            if evidence.id not in {item.id for item in candidate.evidence}:
                                candidate.evidence.append(evidence)
                if decision.event_end_offset_ms > decision.event_start_offset_ms:
                    start = candidate.segment.start_ms + decision.event_start_offset_ms
                    end = candidate.segment.start_ms + decision.event_end_offset_ms
                    candidate.segment = Segment(candidate.segment.id, candidate.segment.video_id, max(candidate.segment.start_ms, start), min(candidate.segment.end_ms, end), candidate.segment.kind, candidate.segment.metadata)
            ordered = sorted(
                [item for item in ordered if item.rerank_score != 0.0],
                key=lambda item: (item.rerank_score if item.rerank_score is not None else 0.0, item.score),
                reverse=True,
            )

        ordered = merge_candidates(ordered, gap_ms=merge_gap_ms)
        return plan, ordered[:top_k], rerank_status

    def search_json(self, query: str, **kwargs: Any) -> dict[str, Any]:
        plan, candidates, status = self.search(query, **kwargs)
        return result_dict(query, plan, candidates, index_version=self.store.get_metadata("index_version", "0.1"), rerank_status=status)
