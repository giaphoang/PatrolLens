from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

TRAJECTORY_SCHEMA_VERSION = "1.0"
SUMMARY_SCHEMA_VERSION = "1.0"
_SUMMARY_TEXT_LIMIT = 4_000


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    """Keep history JSON-safe, bounded, and free of embedded media bytes."""

    if depth > 6:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            return "<inline-media-omitted>"
        return value[:_SUMMARY_TEXT_LIMIT]
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in list(value.items())[:100]:
            lowered = str(key).lower()
            if lowered in {"api_key", "authorization", "openrouter_api_key"}:
                output[str(key)] = "<redacted>"
            else:
                output[str(key)] = _safe_value(item, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item, depth=depth + 1) for item in list(value)[:100]]
    return str(value)[:_SUMMARY_TEXT_LIMIT]


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        item = getattr(value, name, None)
        if item is not None:
            return item
        extra = getattr(value, "model_extra", None)
        if isinstance(extra, dict) and name in extra:
            return extra[name]
    return default


def response_usage(response: Any) -> tuple[dict[str, int], float | None]:
    """Normalize OpenAI/OpenRouter usage and provider-reported cost fields."""

    usage = _field(response, "usage", default={}) or {}
    prompt = int(_field(usage, "prompt_tokens", "input_tokens", default=0) or 0)
    completion = int(_field(usage, "completion_tokens", "output_tokens", default=0) or 0)
    total = int(_field(usage, "total_tokens", default=prompt + completion) or 0)
    tokens = {
        "input": prompt,
        "output": completion,
        "total": total,
    }
    cost = _field(usage, "cost", "total_cost")
    if cost is None:
        details = _field(usage, "cost_details", default={}) or {}
        cost = _field(details, "total_cost", "cost")
    if cost is None:
        cost = _field(response, "cost")
    try:
        normalized_cost = None if cost is None else max(0.0, float(cost))
    except (TypeError, ValueError):
        normalized_cost = None
    return tokens, normalized_cost


class TrajectoryRecorder:
    """Thread-safe, append-only execution history for one CLI run."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        query: str,
        command: str,
        parameters: dict[str, Any] | None = None,
        run_id: str | None = None,
        max_cost_usd: float | None = None,
        estimated_model_call_cost_usd: float = 0.02,
    ) -> None:
        if max_cost_usd is not None and max_cost_usd <= 0:
            raise ValueError("maximum run cost must be positive")
        if estimated_model_call_cost_usd < 0:
            raise ValueError("estimated model call cost cannot be negative")
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.history_root = self.artifact_root / "history"
        self.run_id = run_id or self.new_run_id()
        self.run_root = self.history_root / self.run_id
        self.trajectory_path = self.run_root / "trajectory.jsonl"
        self.summary_path = self.run_root / "summary.json"
        self.index_path = self.history_root / "index.jsonl"
        self.max_cost_usd = max_cost_usd
        self.estimated_model_call_cost_usd = estimated_model_call_cost_usd
        self._started_monotonic = time.monotonic()
        self._lock = threading.RLock()
        self._sequence = 0
        self._context: ContextVar[dict[str, Any]] = ContextVar(
            f"patrol_lens_history_{self.run_id}", default={}
        )
        self._budget_event = threading.Event()
        self._budget_listeners: list[Callable[[], None]] = []
        self._budget_emitted = False
        self.run_root.mkdir(parents=True, exist_ok=False)
        self.trajectory_path.touch(mode=0o600)
        started_at = _utc_now()
        self.summary: dict[str, Any] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "command": command,
            "query": query,
            "title": query[:160],
            "parameters": _safe_value(parameters or {}),
            "started_at": started_at,
            "finished_at": None,
            "status": "running",
            "elapsed_seconds": 0.0,
            "last_completed_stage": None,
            "candidates_retrieved": 0,
            "candidates_examined": 0,
            "final_result": None,
            "best_partial_result": None,
            "result_count": 0,
            "best_confidence": None,
            "openrouter_calls": 0,
            "token_usage": {"input": 0, "output": 0, "total": 0},
            "total_cost_usd": 0.0,
            "max_run_cost_usd": max_cost_usd,
            "estimated_upper_bound_cost_usd": None,
            "termination_reason": None,
            "error": None,
            "trajectory_path": str(self.trajectory_path),
        }
        self.emit(
            "run_started",
            stage="run",
            status="running",
            input_summary={"query": query, "command": command},
        )

    @staticmethod
    def new_run_id() -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{stamp}-{uuid.uuid4().hex[:10]}"

    @property
    def budget_exceeded(self) -> bool:
        return self._budget_event.is_set()

    @contextmanager
    def scope(self, **metadata: Any) -> Iterator[None]:
        context = {**self._context.get(), **{key: value for key, value in metadata.items() if value is not None}}
        token = self._context.set(context)
        try:
            yield
        finally:
            self._context.reset(token)

    def register_budget_listener(self, listener: Callable[[], None]) -> None:
        with self._lock:
            self._budget_listeners.append(listener)
            exceeded = self._budget_event.is_set()
        if exceeded:
            listener()

    def _append(self, path: Path, value: dict[str, Any]) -> None:
        encoded = (json.dumps(value, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _persist_summary(self, *, update_index: bool = False) -> None:
        self.summary["elapsed_seconds"] = round(time.monotonic() - self._started_monotonic, 3)
        temporary = self.summary_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_safe_value(self.summary), handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.summary_path)
        if update_index:
            self._append(
                self.index_path,
                {
                    "schema_version": SUMMARY_SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "query": self.summary["query"],
                    "title": self.summary["title"],
                    "created_at": self.summary["started_at"],
                    "status": self.summary["status"],
                    "duration": self.summary["elapsed_seconds"],
                    "cost": self.summary["total_cost_usd"],
                    "best_confidence": self.summary["best_confidence"],
                    "result_count": self.summary["result_count"],
                    "trajectory_path": self.summary["trajectory_path"],
                },
            )

    def emit(self, event_type: str, **metadata: Any) -> str:
        with self._lock:
            self._sequence += 1
            event_id = f"evt-{self._sequence:08d}-{uuid.uuid4().hex[:8]}"
            context = {**self._context.get(), **metadata}
            event = {
                "schema_version": TRAJECTORY_SCHEMA_VERSION,
                "timestamp": _utc_now(),
                "run_id": self.run_id,
                "event_id": event_id,
                "sequence": self._sequence,
                "event_type": event_type,
                "parent_id": context.pop("parent_id", None),
                "candidate_id": context.pop("candidate_id", None),
                "stage": context.pop("stage", None),
                "candidate_rank": context.pop("candidate_rank", None),
                "agent_turn": context.pop("agent_turn", None),
                "model": context.pop("model", None),
                "latency_ms": context.pop("latency_ms", None),
                "token_usage": context.pop("token_usage", None),
                "request_cost_usd": context.pop("request_cost_usd", None),
                "cumulative_cost_usd": round(float(self.summary["total_cost_usd"]), 8),
                "input_summary": context.pop("input_summary", None),
                "output_summary": context.pop("output_summary", None),
                "media_references": context.pop("media_references", []),
                "status": context.pop("status", None),
                "confidence": context.pop("confidence", None),
                "error": context.pop("error", None),
                "metadata": context,
            }
            self._append(self.trajectory_path, _safe_value(event))
            if event_type == "candidate_started":
                self.summary["candidates_examined"] = (
                    int(self.summary["candidates_examined"]) + 1
                )
            if event_type.endswith("_completed"):
                self.summary["last_completed_stage"] = event["stage"] or event_type
            self._persist_summary(
                update_index=event_type
                in {
                    "run_started",
                    "run_completed",
                    "run_failed",
                    "run_cancelled",
                    "budget_exceeded",
                    "timeout",
                    "model_response",
                    "candidate_supported",
                }
            )
            return event_id

    def update_summary(self, **values: Any) -> None:
        with self._lock:
            self.summary.update(_safe_value(values))
            self._persist_summary()

    def record_partial_result(self, result: dict[str, Any], confidence: float) -> None:
        with self._lock:
            current = self.summary.get("best_confidence")
            if current is None or confidence > float(current):
                self.summary["best_confidence"] = confidence
                self.summary["best_partial_result"] = _safe_value(result)
            self._persist_summary(update_index=True)

    def model_request(
        self,
        *,
        model: str,
        input_summary: Any,
        media_references: list[str] | None = None,
    ) -> tuple[str, float]:
        event_id = self.emit(
            "model_request",
            model=model,
            input_summary=input_summary,
            media_references=media_references or [],
            status="started",
        )
        return event_id, time.monotonic()

    def model_response(
        self,
        response: Any,
        *,
        request_event_id: str,
        started_monotonic: float,
        model: str,
        output_summary: Any = None,
    ) -> None:
        tokens, provider_cost = response_usage(response)
        request_cost = (
            provider_cost
            if provider_cost is not None
            else self.estimated_model_call_cost_usd
        )
        listeners: list[Callable[[], None]] = []
        with self._lock:
            aggregate = self.summary["token_usage"]
            for key in ("input", "output", "total"):
                aggregate[key] = int(aggregate.get(key, 0)) + tokens[key]
            self.summary["openrouter_calls"] = int(self.summary["openrouter_calls"]) + 1
            self.summary["total_cost_usd"] = round(
                float(self.summary["total_cost_usd"]) + request_cost, 8
            )
            crossed = bool(
                self.max_cost_usd is not None
                and float(self.summary["total_cost_usd"]) >= self.max_cost_usd
                and not self._budget_emitted
            )
            if crossed:
                self._budget_emitted = True
                self._budget_event.set()
                listeners = list(self._budget_listeners)
        self.emit(
            "model_response",
            parent_id=request_event_id,
            model=model,
            latency_ms=round((time.monotonic() - started_monotonic) * 1000, 3),
            token_usage=tokens,
            request_cost_usd=round(request_cost, 8),
            cost_source="provider" if provider_cost is not None else "estimated_fallback",
            output_summary=output_summary,
            status="completed",
        )
        if crossed:
            self.emit(
                "budget_exceeded",
                stage="budget",
                status="exceeded",
                output_summary={
                    "limit_usd": self.max_cost_usd,
                    "cumulative_cost_usd": self.summary["total_cost_usd"],
                },
            )
            for listener in listeners:
                listener()

    def provider_error(
        self,
        error: BaseException,
        *,
        model: str | None = None,
        request_event_id: str | None = None,
        started_monotonic: float | None = None,
    ) -> None:
        self.emit(
            "provider_error",
            parent_id=request_event_id,
            model=model,
            latency_ms=(
                round((time.monotonic() - started_monotonic) * 1000, 3)
                if started_monotonic is not None
                else None
            ),
            status="failed",
            error={"type": type(error).__name__, "message": str(error)},
        )

    def estimate_cost(self, *, candidate_count: int, calls_per_candidate: int) -> float:
        estimate = round(
            float(self.summary["total_cost_usd"])
            + candidate_count * calls_per_candidate * self.estimated_model_call_cost_usd,
            8,
        )
        self.update_summary(estimated_upper_bound_cost_usd=estimate)
        self.emit(
            "budget_estimated",
            stage="budget",
            output_summary={
                "candidate_count": candidate_count,
                "calls_per_candidate": calls_per_candidate,
                "estimated_upper_bound_cost_usd": estimate,
                "limit_usd": self.max_cost_usd,
            },
        )
        return estimate

    def deny_estimated_cost(self, estimate: float) -> None:
        listeners: list[Callable[[], None]] = []
        with self._lock:
            if not self._budget_emitted:
                self._budget_emitted = True
                self._budget_event.set()
                listeners = list(self._budget_listeners)
                should_emit = True
            else:
                should_emit = False
        if should_emit:
            self.emit(
                "budget_exceeded",
                stage="budget",
                status="denied",
                output_summary={
                    "reason": "estimated_upper_bound_exceeds_limit",
                    "estimated_upper_bound_cost_usd": estimate,
                    "limit_usd": self.max_cost_usd,
                },
            )
        for listener in listeners:
            listener()

    def finish(
        self,
        *,
        status: str,
        termination_reason: str,
        result: Any = None,
        best_partial_result: Any = None,
        candidates_retrieved: int | None = None,
        candidates_examined: int | None = None,
        result_count: int | None = None,
        best_confidence: float | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        event_type = {
            "completed": "run_completed",
            "failed": "run_failed",
            "cancelled": "run_cancelled",
            "timeout": "run_completed",
            "budget_exceeded": "run_completed",
            "denied": "run_completed",
        }.get(status, "run_failed")
        with self._lock:
            self.summary.update(
                {
                    "finished_at": _utc_now(),
                    "status": status,
                    "termination_reason": termination_reason,
                    "final_result": _safe_value(result),
                    "best_partial_result": _safe_value(best_partial_result),
                    "error": (
                        None
                        if error is None
                        else {
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    ),
                }
            )
            if candidates_retrieved is not None:
                self.summary["candidates_retrieved"] = candidates_retrieved
            if candidates_examined is not None:
                self.summary["candidates_examined"] = candidates_examined
            if result_count is not None:
                self.summary["result_count"] = result_count
            if best_confidence is not None:
                self.summary["best_confidence"] = best_confidence
        self.emit(
            event_type,
            stage="run",
            status=status,
            output_summary={
                "termination_reason": termination_reason,
                "candidates_examined": self.summary["candidates_examined"],
                "result_count": self.summary["result_count"],
                "best_confidence": self.summary["best_confidence"],
            },
            error=self.summary["error"],
        )
        with self._lock:
            self._persist_summary(update_index=True)


def list_history(artifact_root: str | Path) -> list[dict[str, Any]]:
    history_root = Path(artifact_root).expanduser().resolve() / "history"
    latest: dict[str, dict[str, Any]] = {}
    index_path = history_root / "index.jsonl"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                run_id = item.get("run_id")
                if run_id:
                    latest[str(run_id)] = item
    if history_root.exists():
        for summary_path in history_root.glob("*/summary.json"):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            run_id = summary.get("run_id")
            if run_id and run_id not in latest:
                latest[str(run_id)] = {
                    "schema_version": summary.get("schema_version", "0"),
                    "run_id": run_id,
                    "query": summary.get("query"),
                    "title": summary.get("title") or summary.get("query"),
                    "created_at": summary.get("started_at"),
                    "status": summary.get("status"),
                    "duration": summary.get("elapsed_seconds"),
                    "cost": summary.get("total_cost_usd"),
                    "best_confidence": summary.get("best_confidence"),
                    "result_count": summary.get("result_count", 0),
                    "trajectory_path": summary.get("trajectory_path"),
                }
    return sorted(latest.values(), key=lambda item: str(item.get("created_at", "")), reverse=True)


def show_history(artifact_root: str | Path, run_id: str) -> dict[str, Any]:
    run_root = Path(artifact_root).expanduser().resolve() / "history" / run_id
    summary_path = run_root / "summary.json"
    trajectory_path = run_root / "trajectory.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"history run not found: {run_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    if trajectory_path.is_file():
        with trajectory_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    return {"summary": summary, "trajectory": events}
