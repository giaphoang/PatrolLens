from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

from ..domain import EmbeddingRecord, Evidence, Segment, VideoAsset
from ..text import cosine, fts_query, search_tokens
from .schema import SCHEMA


class IndexStore:
    """SQLite metadata, FTS5 text evidence, and canonical vector records."""

    backend = "sqlite"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite"
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self._migrate_legacy_assets()
        self.fts_enabled = self._ensure_fts()
        self.db.commit()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def _migrate_legacy_assets(self) -> None:
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(assets)")}
        if "has_audio" not in columns:
            self.db.execute("ALTER TABLE assets ADD COLUMN has_audio INTEGER NOT NULL DEFAULT 1")
        if "metadata_json" not in columns:
            self.db.execute("ALTER TABLE assets ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")

    def _ensure_fts(self) -> bool:
        try:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts "
                "USING fts5(evidence_id UNINDEXED, modality UNINDEXED, content)"
            )
            return True
        except sqlite3.OperationalError:
            return False

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            self.db.execute("BEGIN")
            yield
        except Exception:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def set_metadata(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.db.commit()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def upsert_asset(self, asset: VideoAsset) -> None:
        self.db.execute(
            """INSERT INTO assets
               (id, path, sha256, duration_ms, fps, width, height, has_audio, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 path=excluded.path, sha256=excluded.sha256, duration_ms=excluded.duration_ms,
                 fps=excluded.fps, width=excluded.width, height=excluded.height,
                 has_audio=excluded.has_audio, metadata_json=excluded.metadata_json""",
            (
                asset.id,
                asset.path,
                asset.sha256,
                asset.duration_ms,
                asset.fps,
                asset.width,
                asset.height,
                int(asset.has_audio),
                json.dumps(asset.metadata),
            ),
        )
        self.db.commit()

    def get_asset(self, asset_id: str) -> VideoAsset | None:
        row = self.db.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
        return None if row is None else self._asset(row)

    def all_assets(self) -> list[VideoAsset]:
        return [self._asset(row) for row in self.db.execute("SELECT * FROM assets ORDER BY id")]

    @staticmethod
    def _asset(row: sqlite3.Row) -> VideoAsset:
        return VideoAsset(
            id=row["id"],
            path=row["path"],
            sha256=row["sha256"],
            duration_ms=int(row["duration_ms"]),
            fps=row["fps"],
            width=row["width"],
            height=row["height"],
            has_audio=bool(row["has_audio"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def upsert_segments(self, segments: Iterable[Segment]) -> None:
        rows = [
            (item.id, item.video_id, item.start_ms, item.end_ms, item.kind, json.dumps(item.metadata))
            for item in segments
        ]
        self.db.executemany(
            """INSERT INTO segments
               (id, video_id, start_ms, end_ms, kind, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 video_id=excluded.video_id, start_ms=excluded.start_ms, end_ms=excluded.end_ms,
                 kind=excluded.kind, metadata_json=excluded.metadata_json""",
            rows,
        )
        self.db.commit()

    def iter_segments(self, video_id: str | None = None) -> Iterator[Segment]:
        if video_id:
            rows = self.db.execute(
                "SELECT * FROM segments WHERE video_id = ? ORDER BY start_ms", (video_id,)
            )
        else:
            rows = self.db.execute("SELECT * FROM segments ORDER BY video_id, start_ms")
        for row in rows:
            yield Segment(
                row["id"], row["video_id"], row["start_ms"], row["end_ms"],
                row["kind"], json.loads(row["metadata_json"]),
            )

    def add_evidence(self, evidence: Evidence) -> None:
        self.add_evidence_many([evidence])

    def _insert_evidence_many(self, evidence: list[Evidence]) -> None:
        if not evidence:
            return
        rows = [
            (
                item.id, item.video_id, item.segment_id, item.start_ms, item.end_ms,
                item.modality, item.content, item.confidence, item.source,
                json.dumps(item.metadata),
            )
            for item in evidence
        ]
        self.db.executemany(
            """INSERT INTO evidence
               (id, video_id, segment_id, start_ms, end_ms, modality, content,
                confidence, source, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 video_id=excluded.video_id, segment_id=excluded.segment_id,
                 start_ms=excluded.start_ms, end_ms=excluded.end_ms,
                 modality=excluded.modality, content=excluded.content,
                 confidence=excluded.confidence, source=excluded.source,
                 metadata_json=excluded.metadata_json""",
            rows,
        )
        if self.fts_enabled:
            self.db.executemany(
                "DELETE FROM evidence_fts WHERE evidence_id = ?",
                [(item.id,) for item in evidence],
            )
            searchable = [
                (item.id, item.modality, item.content)
                for item in evidence
                if item.content.strip()
            ]
            self.db.executemany(
                "INSERT INTO evidence_fts(evidence_id, modality, content) VALUES (?, ?, ?)",
                searchable,
            )

    def add_evidence_many(self, items: Iterable[Evidence]) -> None:
        evidence = list(items)
        if not evidence:
            return
        with self.transaction():
            self._insert_evidence_many(evidence)

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        row = self.db.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
        return None if row is None else self._evidence(row)

    def evidence_between(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
        modalities: Sequence[str] | None = None,
    ) -> list[Evidence]:
        sql = """SELECT * FROM evidence
                 WHERE video_id = ? AND end_ms > ? AND start_ms < ?"""
        params: list[Any] = [video_id, start_ms, end_ms]
        if modalities:
            placeholders = ",".join("?" for _ in modalities)
            sql += f" AND modality IN ({placeholders})"
            params.extend(modalities)
        sql += " ORDER BY start_ms, end_ms"
        return [self._evidence(row) for row in self.db.execute(sql, params)]

    def iter_evidence(self, modality: str | None = None) -> Iterator[Evidence]:
        if modality:
            rows = self.db.execute(
                "SELECT * FROM evidence WHERE modality = ? ORDER BY video_id, start_ms", (modality,)
            )
        else:
            rows = self.db.execute("SELECT * FROM evidence ORDER BY video_id, start_ms")
        for row in rows:
            yield self._evidence(row)

    def evidence_count(self, video_id: str, *, modality: str | None = None) -> int:
        if modality:
            row = self.db.execute(
                "SELECT COUNT(*) AS count FROM evidence WHERE video_id = ? AND modality = ?",
                (video_id, modality),
            ).fetchone()
        else:
            row = self.db.execute(
                "SELECT COUNT(*) AS count FROM evidence WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return int(row["count"])

    def top_evidence(self, modality: str, *, limit: int = 60) -> list[tuple[Evidence, float]]:
        rows = self.db.execute(
            """SELECT * FROM evidence WHERE modality = ?
               ORDER BY confidence DESC, start_ms ASC LIMIT ?""",
            (modality, limit),
        ).fetchall()
        return [(self._evidence(row), float(row["confidence"])) for row in rows]

    @staticmethod
    def _evidence(row: sqlite3.Row) -> Evidence:
        return Evidence(
            id=row["id"],
            video_id=row["video_id"],
            segment_id=row["segment_id"],
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            modality=row["modality"],
            content=row["content"],
            confidence=float(row["confidence"]),
            source=row["source"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def search_text(
        self,
        query: str,
        *,
        modalities: Sequence[str] | None = None,
        limit: int = 60,
    ) -> list[tuple[Evidence, float]]:
        if not query.strip():
            return []
        if self.fts_enabled:
            try:
                sql = """SELECT evidence_id, bm25(evidence_fts) AS rank
                         FROM evidence_fts WHERE evidence_fts MATCH ?"""
                params: list[Any] = [fts_query(query)]
                if modalities:
                    placeholders = ",".join("?" for _ in modalities)
                    sql += f" AND modality IN ({placeholders})"
                    params.extend(modalities)
                sql += " ORDER BY rank LIMIT ?"
                params.append(limit)
                found: list[tuple[Evidence, float]] = []
                for ordinal, row in enumerate(self.db.execute(sql, params), start=1):
                    item = self.get_evidence(row["evidence_id"])
                    if item:
                        found.append((item, 1.0 / ordinal))
                if found:
                    return found
            except sqlite3.OperationalError:
                pass

        query_tokens = set(search_tokens(query))
        candidates = self.iter_evidence()
        scored: list[tuple[Evidence, float]] = []
        allowed = set(modalities or [])
        for item in candidates:
            if allowed and item.modality not in allowed:
                continue
            body = set(search_tokens(item.content))
            score = len(query_tokens & body) / max(1, len(query_tokens))
            if score:
                scored.append((item, score))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:limit]

    def _insert_embeddings(self, records: list[EmbeddingRecord]) -> None:
        if not records:
            return
        rows = [
            (
                item.id, item.evidence_id, item.modality, item.model,
                json.dumps(item.vector), len(item.vector), json.dumps(item.metadata),
            )
            for item in records
        ]
        self.db.executemany(
            """INSERT INTO evidence_embeddings
               (id, evidence_id, modality, model, vector_json, dimensions, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 evidence_id=excluded.evidence_id, modality=excluded.modality,
                 model=excluded.model, vector_json=excluded.vector_json,
                 dimensions=excluded.dimensions, metadata_json=excluded.metadata_json""",
            rows,
        )
    def add_embeddings(self, records: Iterable[EmbeddingRecord]) -> None:
        items = list(records)
        if not items:
            return
        with self.transaction():
            self._insert_embeddings(items)

    def add_evidence_and_embeddings(
        self,
        evidence: Iterable[Evidence],
        records: Iterable[EmbeddingRecord],
    ) -> None:
        evidence_items = list(evidence)
        embedding_items = list(records)
        if not evidence_items and not embedding_items:
            return
        with self.transaction():
            self._insert_evidence_many(evidence_items)
            self._insert_embeddings(embedding_items)

    def get_cached_embedding(
        self,
        cache_key: str,
        *,
        expected_dimensions: int,
    ) -> list[float] | None:
        row = self.db.execute(
            "SELECT vector_json, dimensions FROM embedding_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None or int(row["dimensions"]) != expected_dimensions:
            return None
        self.db.execute(
            "UPDATE embedding_cache SET last_used_at = CURRENT_TIMESTAMP WHERE cache_key = ?",
            (cache_key,),
        )
        self.db.commit()
        return [float(value) for value in json.loads(row["vector_json"])]

    def get_cached_embeddings(
        self,
        cache_keys: list[str],
        *,
        expected_dimensions: int,
    ) -> dict[str, list[float]]:
        if not cache_keys:
            return {}
        unique = list(dict.fromkeys(cache_keys))
        placeholders = ",".join("?" for _ in unique)
        rows = self.db.execute(
            f"""SELECT cache_key, vector_json FROM embedding_cache
                WHERE dimensions = ? AND cache_key IN ({placeholders})""",
            [expected_dimensions, *unique],
        ).fetchall()
        found = {
            row["cache_key"]: [float(value) for value in json.loads(row["vector_json"])]
            for row in rows
        }
        if found:
            self.db.executemany(
                "UPDATE embedding_cache SET last_used_at = CURRENT_TIMESTAMP WHERE cache_key = ?",
                [(key,) for key in found],
            )
            self.db.commit()
        return found

    def put_cached_embedding(
        self,
        cache_key: str,
        *,
        content_hash: str,
        modality: str,
        model: str,
        dimensions: int,
        preprocessing_version: str,
        vector: list[float],
    ) -> None:
        if len(vector) != dimensions:
            raise ValueError(f"Expected {dimensions}, got {len(vector)}")
        self.db.execute(
            """INSERT INTO embedding_cache
               (cache_key, content_hash, modality, model, dimensions,
                preprocessing_version, vector_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                 vector_json=excluded.vector_json,
                 dimensions=excluded.dimensions,
                 last_used_at=CURRENT_TIMESTAMP""",
            (
                cache_key,
                content_hash,
                modality,
                model,
                dimensions,
                preprocessing_version,
                json.dumps(vector),
            ),
        )
        # A provider response becomes recoverable before evidence insertion.
        self.db.commit()

    def embedding_records(self, modality: str, model: str) -> list[tuple[str, list[float]]]:
        rows = self.db.execute(
            "SELECT evidence_id, vector_json FROM evidence_embeddings WHERE modality = ? AND model = ? ORDER BY evidence_id",
            (modality, model),
        ).fetchall()
        return [(row["evidence_id"], json.loads(row["vector_json"])) for row in rows]

    def search_vectors(
        self,
        vector: list[float],
        *,
        modality: str,
        model: str,
        limit: int = 60,
    ) -> list[tuple[Evidence, float]]:
        scored: list[tuple[Evidence, float]] = []
        for evidence_id, candidate in self.embedding_records(modality, model):
            # A model namespace can legitimately contain old and new output
            # dimensions while an index is being migrated. Only compare
            # vectors compatible with the query; PostgreSQL applies the same
            # constraint in its dimensions predicate.
            if len(candidate) != len(vector):
                continue
            item = self.get_evidence(evidence_id)
            if item:
                scored.append((item, cosine(vector, candidate)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def clear_asset_evidence(self, video_id: str) -> None:
        ids = [row["id"] for row in self.db.execute("SELECT id FROM evidence WHERE video_id = ?", (video_id,))]
        with self.transaction():
            if self.fts_enabled:
                self.db.executemany("DELETE FROM evidence_fts WHERE evidence_id = ?", [(item,) for item in ids])
            self.db.execute("DELETE FROM evidence WHERE video_id = ?", (video_id,))
            self.db.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))

    def mark_ingestion(
        self, video_id: str, fingerprint: str, status: str, stats: dict[str, Any]
    ) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO ingestion_runs
               (video_id, fingerprint, status, stats_json, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (video_id, fingerprint, status, json.dumps(stats)),
        )
        self.db.commit()

    def ingestion_status(self, video_id: str, fingerprint: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT status, stats_json FROM ingestion_runs WHERE video_id = ? AND fingerprint = ?",
            (video_id, fingerprint),
        ).fetchone()
        if row is None:
            return None
        return {"status": row["status"], "stats": json.loads(row["stats_json"])}

    def completed_ingestion_fingerprints(self, video_id: str) -> list[str]:
        return [
            row["fingerprint"]
            for row in self.db.execute(
                "SELECT fingerprint FROM ingestion_runs WHERE video_id = ? AND status = 'complete'",
                (video_id,),
            )
        ]

    def supersede_ingestions(self, video_id: str, active_fingerprint: str) -> None:
        self.db.execute(
            """UPDATE ingestion_runs SET status = 'superseded', updated_at = CURRENT_TIMESTAMP
               WHERE video_id = ? AND fingerprint <> ? AND status = 'complete'""",
            (video_id, active_fingerprint),
        )
        self.db.commit()
