from __future__ import annotations

import json
from types import SimpleNamespace

from patrol_lens.history import TrajectoryRecorder, list_history, show_history


def test_trajectory_is_durable_versioned_and_branch_linked(tmp_path):
    recorder = TrajectoryRecorder(
        tmp_path,
        query="find the shouting onset",
        command="search",
        parameters={"candidate_parallelism": 4},
    )

    with recorder.scope(candidate_id="candidate-2", candidate_rank=2):
        parent = recorder.emit("candidate_started", stage="candidate")
        recorder.emit(
            "agent_turn",
            parent_id=parent,
            stage="active_perception",
            agent_turn=1,
        )

    lines = recorder.trajectory_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert events[0]["event_type"] == "run_started"
    assert events[-1]["candidate_id"] == "candidate-2"
    assert events[-1]["candidate_rank"] == 2
    assert events[-1]["parent_id"] == parent
    assert events[-1]["schema_version"] == "1.0"
    assert recorder.summary_path.is_file()


def test_provider_cost_updates_summary_and_triggers_budget(tmp_path):
    recorder = TrajectoryRecorder(
        tmp_path,
        query="query",
        command="search",
        max_cost_usd=0.05,
    )
    cancelled = []
    recorder.register_budget_listener(lambda: cancelled.append(True))
    request_id, started = recorder.model_request(
        model="provider/model",
        input_summary={"prompt": "short"},
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            cost=0.06,
        )
    )
    recorder.model_response(
        response,
        request_event_id=request_id,
        started_monotonic=started,
        model="provider/model",
    )

    summary = json.loads(recorder.summary_path.read_text(encoding="utf-8"))
    assert summary["openrouter_calls"] == 1
    assert summary["token_usage"] == {"input": 10, "output": 4, "total": 14}
    assert summary["total_cost_usd"] == 0.06
    assert recorder.budget_exceeded
    assert cancelled == [True]
    assert any(
        json.loads(line)["event_type"] == "budget_exceeded"
        for line in recorder.trajectory_path.read_text(encoding="utf-8").splitlines()
    )


def test_history_index_and_show_use_summary_without_replaying_other_runs(tmp_path):
    recorder = TrajectoryRecorder(
        tmp_path,
        query="red jacket",
        command="retrieve",
    )
    recorder.finish(
        status="completed",
        termination_reason="retrieval_completed",
        result={"results": []},
        result_count=0,
    )

    listed = list_history(tmp_path)
    assert listed[0]["run_id"] == recorder.run_id
    assert listed[0]["status"] == "completed"
    shown = show_history(tmp_path, recorder.run_id)
    assert shown["summary"]["termination_reason"] == "retrieval_completed"
    assert shown["trajectory"][-1]["event_type"] == "run_completed"
