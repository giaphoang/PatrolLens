from __future__ import annotations

from inspect import getsource

import pytest

from patrol_lens.cli import build_parser
from patrol_lens.index import POSTGRES_SCHEMA, PostgresIndexStore, PostgresVectorIndex
from patrol_lens.index.postgres_store import _hash_vector, _vector_literal


def test_postgres_schema_contains_traceability_columns_and_guards():
    for column in (
        "evidence_id",
        "video_id",
        "start_ms",
        "end_ms",
        "modality",
        "source_uri",
        "evidence_text",
        "evidence_metadata",
        "model_version",
        "confidence",
        "evidence_hash",
        "source_sha256",
        "embedding_hash",
        "embedding vector",
        "embedding_768 vector(768)",
        "pl_embedding_cache",
        "dimensions INTEGER NOT NULL CHECK (dimensions = 768)",
    ):
        assert column in POSTGRES_SCHEMA
    assert "REFERENCES pl_evidence(id) ON DELETE CASCADE" in POSTGRES_SCHEMA
    assert "pl_validate_embedding_provenance" in POSTGRES_SCHEMA
    assert "USING hnsw" in getsource(PostgresIndexStore._ensure_vector_index)


def test_vector_literals_are_finite_and_hashable():
    assert _vector_literal([1, 0.25, -2]) == "[1,0.25,-2]"
    assert len(_hash_vector([1, 0.25, -2])) == 64
    with pytest.raises(ValueError, match="finite"):
        _vector_literal([float("nan")])


def test_postgres_backend_requires_database_url(monkeypatch, tmp_path):
    monkeypatch.delenv("PATROLLENS_DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="PATROLLENS_DATABASE_URL"):
        PostgresIndexStore(root=tmp_path / "artifacts")


def test_cli_can_select_postgres_backend():
    args = build_parser().parse_args(
        [
            "retrieve",
            "license plate",
            "--backend",
            "postgres",
            "--database-url",
            "postgresql://localhost/patrol_lens",
        ]
    )

    assert args.backend == "postgres"
    assert args.database_url == "postgresql://localhost/patrol_lens"


def test_cli_exposes_768_embedding_migration():
    args = build_parser().parse_args(
        [
            "migrate-embeddings",
            "--backend",
            "postgres",
            "--database-url",
            "postgresql://localhost/patrol_lens",
        ]
    )

    assert args.command == "migrate-embeddings"
    assert args.embedding_dimensions == 768


def test_postgres_ingestion_requires_768_dimensions():
    args = build_parser().parse_args(
        [
            "ingest",
            "videos",
            "--backend",
            "postgres",
            "--embedding-dimensions",
            "3072",
        ]
    )

    with pytest.raises(ValueError, match="requires --embedding-dimensions 768"):
        from patrol_lens.cli import cmd_ingest

        cmd_ingest(args)


def test_postgres_vector_index_delegates_to_store():
    class FakeStore:
        def embedding_records(self, modality, model):
            assert (modality, model) == ("visual", "siglip2")
            return [("embedding-1", [1.0, 0.0])]

        def search_vectors(self, vector, *, modality, model, limit):
            return [(vector, modality, model, limit)]

    index = PostgresVectorIndex(FakeStore())
    assert index.rebuild(modality="visual", model="siglip2") == 1
    assert index.search([1.0, 0.0], modality="visual", model="siglip2", limit=3) == [
        ([1.0, 0.0], "visual", "siglip2", 3)
    ]
