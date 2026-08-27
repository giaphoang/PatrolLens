"""Compatibility entry point for deterministic query planning."""

from .retrieval.planner import HeuristicQueryPlanner


def plan_query(query: str):
    return HeuristicQueryPlanner().plan(query)
