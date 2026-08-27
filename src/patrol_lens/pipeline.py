from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from .agent.gemini_agent import ActivePerceptionAgent, AgentRunResult
from .domain import (
    CandidateInterval,
    EvidenceResult,
    QueryPlan,
    SearchResponse,
    VerificationResult,
    VideoAsset,
)
from .index.postgres_store import PostgresIndexStore
from .index.sqlite_store import IndexStore
from .retrieval.search import CoarseRetriever
from .temporal.refine import LightweightTimestampRefiner
from .temporal.timelens2_adapter import TimeLens2Adapter, should_use_timelens2


class EventVerifier(Protocol):
    def verify(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        run: AgentRunResult,
    ) -> VerificationResult: ...


class SearchPipeline:
    """Retrieve cheaply, inspect actively, verify semantically, then ground precisely."""

    def __init__(
        self,
        store: IndexStore | PostgresIndexStore,
        retriever: CoarseRetriever,
        agent: ActivePerceptionAgent,
        verifier: EventVerifier,
        refiner: LightweightTimestampRefiner,
        *,
        timelens2: TimeLens2Adapter | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.agent = agent
        self.verifier = verifier
        self.refiner = refiner
        self.timelens2 = timelens2

    def search(self, query: str, *, max_candidates: int = 10) -> SearchResponse:
        plan, candidates = self.retriever.retrieve(query)
        results: list[EvidenceResult] = []
        warnings: list[str] = []
        examined = 0
        for candidate in candidates[:max_candidates]:
            asset = self.store.get_asset(candidate.video_id)
            if asset is None:
                warnings.append(f"missing_asset:{candidate.video_id}")
                continue
            examined += 1
            try:
                run = self.agent.inspect(query, plan, candidate, asset)
                verification = self.verifier.verify(query, plan, candidate, asset, run)
            except Exception as exc:  # noqa: BLE001 - isolate one failed candidate/provider call
                warnings.append(f"candidate_failed:{candidate.id}:{exc}")
                continue
            if verification.status != "supported":
                continue
            try:
                grounded = self.refiner.refine(
                    query, plan, candidate, asset, verification, run.memory
                )
            except Exception as exc:  # noqa: BLE001 - verified interval remains a safe fallback
                grounded = replace(
                    verification,
                    warnings=[*verification.warnings, f"lightweight_refinement_failed:{exc}"],
                )
            method = "gemini_lightweight"
            intervals = [(grounded.start_ms, grounded.end_ms, grounded.confidence)]
            if self.timelens2 and should_use_timelens2(grounded):
                try:
                    specialist = self.timelens2.ground(
                        asset.path,
                        query,
                        candidate.start_ms,
                        candidate.end_ms,
                    )
                    if specialist:
                        intervals = [
                            (
                                item.start_ms,
                                item.end_ms,
                                min(grounded.confidence, item.score)
                                if item.score is not None
                                else grounded.confidence,
                            )
                            for item in specialist
                        ]
                        method = "timelens2"
                except Exception as exc:  # noqa: BLE001 - optional specialist must not erase a result
                    grounded = replace(
                        grounded,
                        warnings=[*grounded.warnings, f"timelens2_failed:{exc}"],
                    )
            for start_ms, end_ms, confidence in intervals:
                results.append(
                    EvidenceResult(
                        video_id=asset.id,
                        video_path=asset.path,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        confidence=max(0.0, min(1.0, confidence)),
                        description=grounded.event_description,
                        evidence=grounded.evidence,
                        retrieval_score=candidate.score,
                        grounding_method=method,
                        warnings=grounded.warnings,
                    )
                )
        results.sort(key=lambda item: (item.confidence, item.retrieval_score), reverse=True)
        return SearchResponse(
            query=query,
            plan=plan,
            results=results,
            candidates_examined=examined,
            index_version=self.store.get_metadata("index_version", "unknown"),
            warnings=warnings,
        )
