from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from .config import SearchConfig
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
from .verification import DirectVerificationContext

if TYPE_CHECKING:
    from .history import TrajectoryRecorder


class CandidateMediaProvider(Protocol):
    def prepare(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        *,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> DirectVerificationContext: ...


class EventVerifier(Protocol):
    def verify(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        context: DirectVerificationContext,
    ) -> VerificationResult: ...


@dataclass(frozen=True)
class _CandidateProgress:
    candidate_id: str
    result: EvidenceResult


@dataclass(frozen=True)
class _CandidateOutcome:
    candidate_id: str
    results: tuple[EvidenceResult, ...] = ()
    warning: str | None = None
    cancelled: bool = False
    status: str | None = None
    confidence: float | None = None


class _SearchControl:
    """Thread-safe cancellation and single-winner early-stop state."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._winner_id: str | None = None
        self._winner_confidence: float | None = None

    def claim_winner(self, candidate_id: str, confidence: float) -> bool:
        with self._lock:
            if self.cancel_event.is_set():
                return self._winner_id == candidate_id
            self._winner_id = candidate_id
            self._winner_confidence = confidence
            self.cancel_event.set()
            return True

    def cancel(self) -> None:
        with self._lock:
            self.cancel_event.set()

    def take_task(
        self,
        work: queue.Queue[tuple[CandidateInterval, VideoAsset, int]],
    ) -> tuple[CandidateInterval, VideoAsset, int] | None:
        """Claim queued work atomically with respect to stop/winner changes."""

        with self._lock:
            if self.cancel_event.is_set():
                return None
            try:
                return work.get_nowait()
            except queue.Empty:
                return None

    def is_winner(self, candidate_id: str) -> bool:
        with self._lock:
            return self._winner_id == candidate_id

    def winner(self) -> tuple[str | None, float | None]:
        with self._lock:
            return self._winner_id, self._winner_confidence


class SearchPipeline:
    """Retrieve broadly, rerank cheaply, verify directly, then ground precisely."""

    def __init__(
        self,
        store: IndexStore | PostgresIndexStore,
        retriever: CoarseRetriever,
        media_provider: CandidateMediaProvider,
        verifier: EventVerifier,
        refiner: LightweightTimestampRefiner,
        *,
        timelens2: TimeLens2Adapter | None = None,
        config: SearchConfig | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.media_provider = media_provider
        self.verifier = verifier
        self.refiner = refiner
        self.timelens2 = timelens2
        self.config = config or SearchConfig()
        self.recorder = recorder

    @staticmethod
    def _required_direct_modalities(plan: QueryPlan) -> set[str]:
        required: set[str] = set()
        if "visual" in plan.required_modalities:
            required.add("visual")
        if "audio_event" in plan.required_modalities:
            required.add("audio")
        if {"visual", "audio_event"}.issubset(set(plan.required_modalities)):
            required.add("audiovisual")
        return required

    @staticmethod
    def _deadline_reached(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _rerank_candidates(
        self,
        plan: QueryPlan,
        candidates: list[CandidateInterval],
    ) -> list[CandidateInterval]:
        """Prefer required-modality coverage, then preserve retrieval scoring."""

        required = set(plan.required_modalities)

        def key(candidate: CandidateInterval) -> tuple[int, int, int, float, float]:
            covered = set(candidate.covered_modalities)
            covered.update(item.modality for item in candidate.evidence)
            matched = len(required & covered)
            missing = len(required - covered)
            diversity = len(candidate.branch_scores)
            evidence_confidence = max(
                (item.confidence for item in candidate.evidence),
                default=0.0,
            )
            return (-missing, matched, diversity, candidate.score, evidence_confidence)

        ranked = sorted(candidates, key=key, reverse=True)
        if self.recorder:
            self.recorder.emit(
                "candidate_reranking_completed",
                stage="candidate_reranking",
                status="completed",
                output_summary={
                    "method": "required_modality_coverage_then_retrieval_score",
                    "candidate_count": len(ranked),
                    "candidates": [
                        {
                            "candidate_id": item.id,
                            "rank": rank,
                            "retrieval_score": item.score,
                            "covered_modalities": item.covered_modalities,
                            "missing_modalities": item.missing_modalities,
                        }
                        for rank, item in enumerate(ranked, start=1)
                    ],
                },
            )
        return ranked

    @staticmethod
    def _as_result(
        candidate: CandidateInterval,
        asset: VideoAsset,
        grounded: VerificationResult,
        *,
        method: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        confidence: float | None = None,
    ) -> EvidenceResult:
        return EvidenceResult(
            video_id=asset.id,
            video_path=asset.path,
            start_ms=grounded.start_ms if start_ms is None else start_ms,
            end_ms=grounded.end_ms if end_ms is None else end_ms,
            confidence=max(
                0.0,
                min(1.0, grounded.confidence if confidence is None else confidence),
            ),
            description=grounded.event_description,
            evidence=grounded.evidence,
            retrieval_score=candidate.score,
            grounding_method=method,
            warnings=grounded.warnings,
        )

    def _ground_supported(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        verification: VerificationResult,
        context: DirectVerificationContext,
        *,
        control: _SearchControl | None = None,
        deadline: float | None = None,
    ) -> tuple[EvidenceResult, ...]:
        if control and control.cancel_event.is_set() and not control.is_winner(candidate.id):
            return ()
        if self._deadline_reached(deadline):
            return ()
        refinement_event: str | None = None
        if self.recorder:
            refinement_event = self.recorder.emit(
                "refinement_started",
                stage="refinement",
                status="started",
                input_summary={
                    "start_ms": verification.start_ms,
                    "end_ms": verification.end_ms,
                    "confidence": verification.confidence,
                },
            )
        try:
            if self.recorder:
                with self.recorder.scope(stage="refinement", parent_id=refinement_event):
                    grounded = self.refiner.refine(
                        query, plan, candidate, asset, verification, context.workspace
                    )
            else:
                grounded = self.refiner.refine(
                    query, plan, candidate, asset, verification, context.workspace
                )
        except Exception as exc:  # noqa: BLE001 - verified interval remains a safe fallback
            grounded = replace(
                verification,
                warnings=[*verification.warnings, f"lightweight_refinement_failed:{exc}"],
            )
        method = "gemini_lightweight"
        intervals = [(grounded.start_ms, grounded.end_ms, grounded.confidence)]
        if self.timelens2 and should_use_timelens2(grounded):
            if control and control.cancel_event.is_set() and not control.is_winner(candidate.id):
                return ()
            if self._deadline_reached(deadline):
                return ()
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
        results = tuple(
            self._as_result(
                candidate,
                asset,
                grounded,
                method=method,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=confidence,
            )
            for start_ms, end_ms, confidence in intervals
        )
        if self.recorder:
            self.recorder.emit(
                "refinement_completed",
                stage="refinement",
                parent_id=refinement_event,
                status="completed",
                confidence=max((item.confidence for item in results), default=grounded.confidence),
                output_summary={"method": method, "results": [item.to_dict() for item in results]},
            )
        return results

    def _evaluate_candidate_inner(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        *,
        control: _SearchControl | None = None,
        deadline: float | None = None,
        progress: queue.Queue[object] | None = None,
    ) -> _CandidateOutcome:
        if control and control.cancel_event.is_set():
            return _CandidateOutcome(candidate.id, cancelled=True)
        try:
            context = self.media_provider.prepare(
                query,
                plan,
                candidate,
                asset,
                cancel_event=control.cancel_event if control else None,
                deadline=deadline,
            )
            if self._deadline_reached(deadline):
                if control:
                    control.cancel()
                return _CandidateOutcome(candidate.id, cancelled=True)
            if control and control.cancel_event.is_set() and not control.is_winner(candidate.id):
                return _CandidateOutcome(candidate.id, cancelled=True)
            verification = self.verifier.verify(query, plan, candidate, asset, context)
        except Exception as exc:  # noqa: BLE001 - isolate one failed candidate/provider call
            if control and control.cancel_event.is_set():
                return _CandidateOutcome(candidate.id, cancelled=True)
            return _CandidateOutcome(
                candidate.id,
                warning=f"candidate_failed:{candidate.id}:{exc}",
                status="failed",
            )
        if verification.status != "supported":
            return _CandidateOutcome(
                candidate.id,
                status=verification.status,
                confidence=verification.confidence,
            )

        direct_satisfied = self._required_direct_modalities(plan).issubset(
            context.direct_modalities
        )
        qualifies = bool(
            control
            and self.config.early_stop_confidence is not None
            and verification.confidence >= self.config.early_stop_confidence
            and direct_satisfied
            and not verification.missing_evidence
        )
        won_early_stop = bool(
            qualifies and control and control.claim_winner(candidate.id, verification.confidence)
        )
        if won_early_stop and self.recorder:
            self.recorder.emit(
                "early_stop",
                stage="candidate_verification",
                status="triggered",
                confidence=verification.confidence,
                output_summary={
                    "threshold": self.config.early_stop_confidence,
                    "direct_modalities": sorted(context.direct_modalities),
                },
            )
        provisional = self._as_result(
            candidate,
            asset,
            verification,
            method="gemini_verifier",
        )
        if self.recorder:
            self.recorder.record_partial_result(
                provisional.to_dict(), provisional.confidence
            )
        if progress is not None:
            progress.put(_CandidateProgress(candidate.id, provisional))
        if control and control.cancel_event.is_set() and not (
            won_early_stop or control.is_winner(candidate.id)
        ):
            return _CandidateOutcome(
                candidate.id,
                (provisional,),
                cancelled=True,
                status="supported",
                confidence=verification.confidence,
            )
        grounded = self._ground_supported(
            query,
            plan,
            candidate,
            asset,
            verification,
            context,
            control=control,
            deadline=deadline,
        )
        return _CandidateOutcome(
            candidate.id,
            grounded or (provisional,),
            status="supported",
            confidence=verification.confidence,
        )

    def _evaluate_candidate(
        self,
        query: str,
        plan: QueryPlan,
        candidate: CandidateInterval,
        asset: VideoAsset,
        *,
        candidate_rank: int | None = None,
        control: _SearchControl | None = None,
        deadline: float | None = None,
        progress: queue.Queue[object] | None = None,
    ) -> _CandidateOutcome:
        candidate_event: str | None = None
        if self.recorder:
            candidate_event = self.recorder.emit(
                "candidate_started",
                stage="candidate",
                candidate_id=candidate.id,
                candidate_rank=candidate_rank,
                status="started",
                input_summary={
                    "video_id": candidate.video_id,
                    "start_ms": candidate.start_ms,
                    "end_ms": candidate.end_ms,
                    "retrieval_score": candidate.score,
                },
            )
            with self.recorder.scope(
                stage="candidate",
                candidate_id=candidate.id,
                candidate_rank=candidate_rank,
                parent_id=candidate_event,
            ):
                outcome = self._evaluate_candidate_inner(
                    query,
                    plan,
                    candidate,
                    asset,
                    control=control,
                    deadline=deadline,
                    progress=progress,
                )
        else:
            outcome = self._evaluate_candidate_inner(
                query,
                plan,
                candidate,
                asset,
                control=control,
                deadline=deadline,
                progress=progress,
            )
        if self.recorder:
            if outcome.cancelled:
                event_type = "candidate_cancelled"
                status = "cancelled"
            elif outcome.status == "supported":
                event_type = "candidate_supported"
                status = "supported"
            elif outcome.status in {"rejected", "uncertain"}:
                event_type = "candidate_rejected"
                status = outcome.status
            elif outcome.warning:
                event_type = "provider_error"
                status = "failed"
            else:
                event_type = "candidate_verified"
                status = outcome.status or "completed"
            self.recorder.emit(
                "candidate_verified",
                stage="candidate_verification",
                parent_id=candidate_event,
                candidate_id=candidate.id,
                candidate_rank=candidate_rank,
                status=outcome.status or status,
                confidence=outcome.confidence,
                output_summary={"result_count": len(outcome.results), "warning": outcome.warning},
            )
            if event_type != "candidate_verified":
                self.recorder.emit(
                    event_type,
                    stage="candidate",
                    parent_id=candidate_event,
                    candidate_id=candidate.id,
                    candidate_rank=candidate_rank,
                    status=status,
                    confidence=outcome.confidence,
                    error=outcome.warning,
                    output_summary={"results": [item.to_dict() for item in outcome.results]},
                )
        return outcome

    def _response(
        self,
        query: str,
        plan: QueryPlan,
        results: list[EvidenceResult],
        examined: int,
        warnings: list[str],
        *,
        index_version: str | None = None,
    ) -> SearchResponse:
        results.sort(key=lambda item: (item.confidence, item.retrieval_score), reverse=True)
        response = SearchResponse(
            query=query,
            plan=plan,
            results=results,
            candidates_examined=examined,
            index_version=(
                str(self.store.get_metadata("index_version", "unknown"))
                if index_version is None
                else index_version
            ),
            warnings=warnings,
        )
        if self.recorder:
            best = results[0].to_dict() if results else None
            self.recorder.update_summary(
                candidates_examined=examined,
                best_partial_result=best,
                result_count=len(results),
                best_confidence=results[0].confidence if results else None,
            )
        return response

    def _preflight_budget(self, candidate_count: int) -> bool:
        if not self.recorder:
            return False
        estimate = self.recorder.estimate_cost(
            candidate_count=candidate_count,
            calls_per_candidate=2,
        )
        if (
            self.config.max_run_cost_usd is not None
            and estimate > self.config.max_run_cost_usd
        ):
            self.recorder.deny_estimated_cost(estimate)
            return True
        return False

    def _search_sequential(self, query: str, *, max_candidates: int) -> SearchResponse:
        plan, candidates = self.retriever.retrieve(query)
        results: list[EvidenceResult] = []
        warnings: list[str] = []
        examined = 0
        selected = self._rerank_candidates(plan, candidates)[:max_candidates]
        if self._preflight_budget(len(selected)):
            return self._response(
                query,
                plan,
                [],
                0,
                ["run_cost_denied:estimated_upper_bound_exceeds_limit"],
            )
        for rank, candidate in enumerate(selected, start=1):
            asset = self.store.get_asset(candidate.video_id)
            if asset is None:
                warnings.append(f"missing_asset:{candidate.video_id}")
                continue
            examined += 1
            outcome = self._evaluate_candidate(
                query, plan, candidate, asset, candidate_rank=rank
            )
            if outcome.warning:
                warnings.append(outcome.warning)
            results.extend(outcome.results)
        return self._response(query, plan, results, examined, warnings)

    def _retrieve_with_deadline(
        self,
        query: str,
        deadline: float,
    ) -> tuple[QueryPlan, list[CandidateInterval]] | None:
        output: queue.Queue[object] = queue.Queue(maxsize=1)

        def retrieve() -> None:
            try:
                output.put(self.retriever.retrieve(query))
            except BaseException as exc:  # noqa: BLE001 - forwarded to the caller thread
                output.put(exc)

        threading.Thread(target=retrieve, name="patrol-lens-retrieval", daemon=True).start()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            item = output.get(timeout=remaining)
        except queue.Empty:
            return None
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

    def _search_bounded(self, query: str, *, max_candidates: int) -> SearchResponse:
        started = time.monotonic()
        deadline = started + self.config.timeout_s if self.config.timeout_s is not None else None
        index_version = str(self.store.get_metadata("index_version", "unknown"))
        if self._deadline_reached(deadline):
            if self.recorder:
                self.recorder.emit(
                    "timeout",
                    stage="metadata",
                    status="timeout",
                    output_summary={"timeout_s": self.config.timeout_s},
                )
            return self._response(
                query,
                QueryPlan(original_text=query),
                [],
                0,
                ["search_timeout:metadata", "search_timeout_no_supported_result"],
                index_version=index_version,
            )
        if deadline is None:
            plan, candidates = self.retriever.retrieve(query)
        else:
            retrieved = self._retrieve_with_deadline(query, deadline)
            if retrieved is None:
                if self.recorder:
                    self.recorder.emit(
                        "timeout",
                        stage="retrieval",
                        status="timeout",
                        output_summary={"timeout_s": self.config.timeout_s},
                    )
                return self._response(
                    query,
                    QueryPlan(original_text=query),
                    [],
                    0,
                    ["search_timeout:retrieval", "search_timeout_no_supported_result"],
                    index_version=index_version,
                )
            plan, candidates = retrieved

        warnings: list[str] = []
        tasks: list[tuple[CandidateInterval, VideoAsset, int]] = []
        ranked = self._rerank_candidates(plan, candidates)
        for rank, candidate in enumerate(ranked[:max_candidates], start=1):
            asset = self.store.get_asset(candidate.video_id)
            if asset is None:
                warnings.append(f"missing_asset:{candidate.video_id}")
            else:
                tasks.append((candidate, asset, rank))
        if not tasks:
            return self._response(
                query, plan, [], 0, warnings, index_version=index_version
            )

        if self._preflight_budget(len(tasks)):
            return self._response(
                query,
                plan,
                [],
                0,
                ["run_cost_denied:estimated_upper_bound_exceeds_limit"],
                index_version=index_version,
            )

        work: queue.Queue[tuple[CandidateInterval, VideoAsset, int]] = queue.Queue()
        events: queue.Queue[object] = queue.Queue()
        for task in tasks:
            work.put(task)
        control = _SearchControl()
        if self.recorder:
            self.recorder.register_budget_listener(control.cancel)
        examined = 0
        examined_lock = threading.Lock()
        worker_count = min(self.config.candidate_parallelism, len(tasks))

        def worker() -> None:
            nonlocal examined
            try:
                while True:
                    if self._deadline_reached(deadline):
                        control.cancel()
                        break
                    task = control.take_task(work)
                    if task is None:
                        break
                    candidate, asset, candidate_rank = task
                    with examined_lock:
                        examined += 1
                    outcome = self._evaluate_candidate(
                        query,
                        plan,
                        candidate,
                        asset,
                        candidate_rank=candidate_rank,
                        control=control,
                        deadline=deadline,
                        progress=events,
                    )
                    events.put(outcome)
            finally:
                events.put(None)

        for ordinal in range(worker_count):
            threading.Thread(
                target=worker,
                name=f"patrol-lens-candidate-{ordinal + 1}",
                daemon=True,
            ).start()

        results_by_candidate: dict[str, tuple[EvidenceResult, ...]] = {}
        stopped_workers = 0
        timed_out = False
        winner_finished = False

        def record(item: object) -> None:
            nonlocal stopped_workers, winner_finished
            if item is None:
                stopped_workers += 1
            elif isinstance(item, _CandidateProgress):
                results_by_candidate[item.candidate_id] = (item.result,)
            elif isinstance(item, _CandidateOutcome):
                if item.warning:
                    warnings.append(item.warning)
                if item.results:
                    results_by_candidate[item.candidate_id] = item.results
                winner_id, _confidence = control.winner()
                if winner_id == item.candidate_id:
                    winner_finished = True

        while stopped_workers < worker_count and not winner_finished:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                timed_out = True
                control.cancel()
                break
            try:
                record(events.get(timeout=remaining))
            except queue.Empty:
                timed_out = True
                control.cancel()
                break

        while True:
            try:
                record(events.get_nowait())
            except queue.Empty:
                break

        winner_id, winner_confidence = control.winner()
        if deadline is not None and time.monotonic() >= deadline and not winner_finished:
            timed_out = True
            control.cancel()
        if winner_id is not None:
            warnings.append(
                f"early_stop:{winner_id}:confidence={winner_confidence:.3f}"
            )
        if timed_out:
            warnings.append(f"search_timeout:{self.config.timeout_s:g}s")
            warnings.append(
                "search_timeout_returning_best_supported"
                if results_by_candidate
                else "search_timeout_no_supported_result"
            )
            if self.recorder:
                self.recorder.emit(
                    "timeout",
                    stage="search",
                    status="timeout",
                    output_summary={
                        "timeout_s": self.config.timeout_s,
                        "candidates_examined": examined,
                        "supported_candidates": len(results_by_candidate),
                    },
                )
        if self.recorder and self.recorder.budget_exceeded:
            warnings.append(
                "run_cost_budget_reached_returning_best_supported"
                if results_by_candidate
                else "run_cost_budget_reached_no_supported_result"
            )
        results = [
            result
            for candidate_results in results_by_candidate.values()
            for result in candidate_results
        ]
        return self._response(
            query,
            plan,
            results,
            examined,
            warnings,
            index_version=index_version,
        )

    def search(self, query: str, *, max_candidates: int = 10) -> SearchResponse:
        if self.config == SearchConfig():
            return self._search_sequential(query, max_candidates=max_candidates)
        return self._search_bounded(query, max_candidates=max_candidates)
