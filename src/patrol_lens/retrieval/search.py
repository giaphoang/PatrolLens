from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ..config import RetrievalConfig
from ..domain import CandidateInterval, Evidence, QueryPlan, RetrievalHit
from ..index.faiss_store import AutoVectorIndex, PostgresVectorIndex, VectorIndex
from ..index.postgres_store import PostgresIndexStore
from ..index.sqlite_store import IndexStore
from .fusion import fuse_hits
from .planner import HeuristicQueryPlanner, QueryPlanner

if TYPE_CHECKING:
    from ..history import TrajectoryRecorder


class TextEncoder(Protocol):
    model_name: str

    def encode_text(self, text: str) -> list[float]: ...


class CoarseRetriever:
    def __init__(
        self,
        store: IndexStore | PostgresIndexStore,
        *,
        planner: QueryPlanner | None = None,
        visual_encoder: TextEncoder | None = None,
        semantic_encoder: TextEncoder | None = None,
        audio_encoder: TextEncoder | None = None,
        vector_index: VectorIndex | PostgresVectorIndex | None = None,
        config: RetrievalConfig | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.store = store
        self.planner = planner or HeuristicQueryPlanner()
        self.visual_encoder = visual_encoder
        self.semantic_encoder = semantic_encoder
        self.audio_encoder = audio_encoder
        self.vector_index = vector_index or AutoVectorIndex(store)
        self.config = config or RetrievalConfig()
        self.recorder = recorder

    @staticmethod
    def _hits(branch: str, pairs: list[tuple[Evidence, float]]) -> list[RetrievalHit]:
        return [
            RetrievalHit(branch=branch, rank=rank, score=score, evidence=evidence)
            for rank, (evidence, score) in enumerate(pairs, start=1)
        ]

    def retrieve_plan(self, plan: QueryPlan) -> list[CandidateInterval]:
        branches: dict[str, list[RetrievalHit]] = {}
        limit = self.config.branch_k
        for index, query in enumerate(plan.transcript_queries):
            branch = f"transcript:{index}"
            branches[branch] = self._hits(
                branch,
                self.store.search_text(query, modalities=["transcript"], limit=limit),
            )
            if self.semantic_encoder:
                semantic_branch = f"{branch}:semantic"
                branches[semantic_branch] = self._hits(
                    semantic_branch,
                    self.vector_index.search(
                        self.semantic_encoder.encode_text(query),
                        modality="transcript",
                        model=self.semantic_encoder.model_name,
                        limit=limit,
                    ),
                )
        for index, query in enumerate(plan.ocr_queries):
            branch = f"ocr:{index}"
            pairs = self.store.top_evidence("ocr", limit=limit) if query == "*" else self.store.search_text(
                query, modalities=["ocr"], limit=limit
            )
            branches[branch] = self._hits(branch, pairs)
            if self.semantic_encoder and query != "*":
                semantic_branch = f"{branch}:semantic"
                branches[semantic_branch] = self._hits(
                    semantic_branch,
                    self.vector_index.search(
                        self.semantic_encoder.encode_text(query),
                        modality="ocr",
                        model=self.semantic_encoder.model_name,
                        limit=limit,
                    ),
                )
        for index, query in enumerate(plan.audio_queries):
            branch = f"audio_event:{index}"
            branches[branch] = self._hits(
                branch,
                self.store.search_text(query, modalities=["audio_event"], limit=limit),
            )
            if self.audio_encoder:
                clap_branch = f"{branch}:clap"
                branches[clap_branch] = self._hits(
                    clap_branch,
                    self.vector_index.search(
                        self.audio_encoder.encode_text(query),
                        modality="audio_event",
                        model=self.audio_encoder.model_name,
                        limit=limit,
                    ),
                )
        if self.visual_encoder:
            for index, query in enumerate(plan.visual_queries):
                branch = f"visual:{index}"
                pairs = self.vector_index.search(
                    self.visual_encoder.encode_text(query),
                    modality="visual",
                    model=self.visual_encoder.model_name,
                    limit=limit,
                )
                branches[branch] = self._hits(branch, pairs)

        durations = {asset.id: asset.duration_ms for asset in self.store.all_assets()}
        candidates = fuse_hits(
            branches,
            plan,
            config=self.config,
            video_durations=durations,
        )
        return candidates[: self.config.top_k]

    def retrieve(self, query: str) -> tuple[QueryPlan, list[CandidateInterval]]:
        planner_event: str | None = None
        if self.recorder:
            planner_event = self.recorder.emit(
                "planner_started",
                stage="planner",
                status="started",
                input_summary={"query": query, "planner": type(self.planner).__name__},
            )
        try:
            if self.recorder:
                with self.recorder.scope(stage="planner", parent_id=planner_event):
                    plan = self.planner.plan(query)
            else:
                plan = self.planner.plan(query)
        except Exception as exc:
            if self.recorder:
                self.recorder.emit(
                    "provider_error",
                    stage="planner",
                    parent_id=planner_event,
                    status="failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            raise
        if self.recorder:
            self.recorder.emit(
                "planner_completed",
                stage="planner",
                parent_id=planner_event,
                status="completed",
                output_summary=plan.to_dict(),
            )
            retrieval_event = self.recorder.emit(
                "retrieval_started",
                stage="retrieval",
                status="started",
                input_summary=plan.to_dict(),
            )
            with self.recorder.scope(stage="retrieval", parent_id=retrieval_event):
                candidates = self.retrieve_plan(plan)
            self.recorder.update_summary(candidates_retrieved=len(candidates))
            self.recorder.emit(
                "retrieval_completed",
                stage="retrieval",
                parent_id=retrieval_event,
                status="completed",
                output_summary={
                    "candidate_count": len(candidates),
                    "candidates": [
                        {
                            "candidate_id": item.id,
                            "video_id": item.video_id,
                            "start_ms": item.start_ms,
                            "end_ms": item.end_ms,
                            "score": item.score,
                            "rank": rank,
                        }
                        for rank, item in enumerate(candidates, start=1)
                    ],
                },
            )
        else:
            candidates = self.retrieve_plan(plan)
        return plan, candidates

    def search(self, query: str, **_kwargs: Any) -> tuple[QueryPlan, list[CandidateInterval], str]:
        plan, candidates = self.retrieve(query)
        return plan, candidates, "coarse_only"

    def search_json(self, query: str, **_kwargs: Any) -> dict[str, Any]:
        plan, candidates = self.retrieve(query)
        return {
            "query": query,
            "query_plan": plan.to_dict(),
            "index_version": self.store.get_metadata("index_version", "unknown"),
            "stage": "coarse_retrieval",
            "results": [item.to_dict() for item in candidates],
        }
