from __future__ import annotations

from patrol_lens.cli import build_parser


def test_cli_exposes_index_search_and_evaluate():
    parser = build_parser()
    assert parser.parse_args(["search", "red shirt"]).command == "search"
    assert parser.parse_args(["index", "videos"]).command == "index"
    assert parser.parse_args(["evaluate", "queries.jsonl"]).command == "evaluate"
