from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .domain import EmbeddingRecord, Observation, Segment, VideoAsset
from .text import cosine, fts_query, normalize_text, search_tokens


SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  fps REAL,
  width INTEGER,
  height INTEGER
);
CREATE TABLE IF NOT EXISTS segments (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  kind TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY(video_id) REFERENCES assets(id)
);
CREATE INDEX IF NOT EXISTS idx_segments_video_time ON segments(video_id, start_ms, end_ms);
CREATE TABLE IF NOT EXISTS observations (
  id TEXT PRIMARY KEY,
  segment_id TEXT NOT NULL,
  video_id TEXT NOT NULL,
  modality TEXT NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  text TEXT,
  label TEXT,
  confidence REAL,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY(segment_id) REFERENCES segments(id)
);
CREATE INDEX IF NOT EXISTS idx_observations_modality ON observations(modality);
CREATE TABLE IF NOT EXISTS embeddings (
  id TEXT PRIMARY KEY,
  segment_id TEXT NOT NULL,
  modality TEXT NOT NULL,
  model TEXT NOT NULL,
  vector_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY(segment_id) REFERENCES segments(id)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_modality_model ON embeddings(modality, model);
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class IndexStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite"
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.fts_enabled = False
        try:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(document_id UNINDEXED, segment_id UNINDEXED, modality UNINDEXED, body)"
            )
            self.fts_enabled = True
        except sqlite3.OperationalError:
            self.fts_enabled = False
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def set_metadata(self, key: str, value: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def upsert_asset(self, asset: VideoAsset) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO assets(id, path, sha256, duration_ms, fps, width, height)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (asset.id, asset.path, asset.sha256, asset.duration_ms, asset.fps, asset.width, asset.height),
        )
        self.db.commit()

    def upsert_segment(self, segment: Segment) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO segments(id, video_id, start_ms, end_ms, kind, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (segment.id, segment.video_id, segment.start_ms, segment.end_ms, segment.kind, json.dumps(segment.metadata)),
        )
        self.db.commit()

    def add_observation(self, observation: Observation) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO observations
               (id, segment_id, video_id, modality, start_ms, end_ms, text, label, confidence, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                observation.id,
                observation.segment_id,
                observation.video_id,
                observation.modality,
                observation.start_ms,
                observation.end_ms,
                observation.text,
                observation.label,
                observation.confidence,
                json.dumps(observation.metadata),
            ),
        )
        if self.fts_enabled and observation.text:
            self.db.execute("DELETE FROM documents_fts WHERE document_id = ?", (observation.id,))
            self.db.execute(
                "INSERT INTO documents_fts(document_id, segment_id, modality, body) VALUES (?, ?, ?, ?)",
                (observation.id, observation.segment_id, observation.modality, normalize_text(observation.text)),
            )
        self.db.commit()

    def add_embedding(self, record: EmbeddingRecord) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO embeddings
               (id, segment_id, modality, model, vector_json, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.segment_id,
                record.modality,
                record.model,
                json.dumps(record.vector),
                json.dumps(record.metadata),
            ),
        )
        self.db.commit()

    def get_asset(self, asset_id: str) -> VideoAsset | None:
        row = self.db.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            return None
        return VideoAsset(row["id"], row["path"], row["sha256"], row["duration_ms"], row["fps"], row["width"], row["height"])

    def get_segment(self, segment_id: str) -> Segment | None:
        row = self.db.execute("SELECT * FROM segments WHERE id = ?", (segment_id,)).fetchone()
        if row is None:
            return None
        return Segment(row["id"], row["video_id"], row["start_ms"], row["end_ms"], row["kind"], json.loads(row["metadata_json"]))

    def get_observations(self, segment_id: str) -> list[Observation]:
        rows = self.db.execute("SELECT * FROM observations WHERE segment_id = ? ORDER BY start_ms", (segment_id,)).fetchall()
        return [self._observation(row) for row in rows]

    def _observation(self, row: sqlite3.Row) -> Observation:
        return Observation(
            row["id"], row["segment_id"], row["video_id"], row["modality"], row["start_ms"], row["end_ms"],
            row["text"], row["label"], row["confidence"], json.loads(row["metadata_json"]),
        )

    def search_text(self, query: str, limit: int = 50, modality: str | None = None) -> list[tuple[str, float, Observation]]:
        if not query.strip():
            return []
        if self.fts_enabled:
            try:
                sql = """SELECT document_id, segment_id, modality, bm25(documents_fts) AS rank
                         FROM documents_fts WHERE documents_fts MATCH ?"""
                params: list[Any] = [fts_query(query)]
                if modality:
                    sql += " AND modality = ?"
                    params.append(modality)
                sql += " ORDER BY rank LIMIT ?"
                params.append(limit)
                rows = self.db.execute(sql, params).fetchall()
                results = []
                for row in rows:
                    obs_row = self.db.execute("SELECT * FROM observations WHERE id = ?", (row["document_id"],)).fetchone()
                    if obs_row:
                        results.append((row["segment_id"], 1.0 / (1.0 + max(float(row["rank"]), 0.0)), self._observation(obs_row)))
                if results:
                    return results
            except sqlite3.OperationalError:
                pass
        tokens = set(search_tokens(query))
        if modality:
            rows = self.db.execute("SELECT * FROM observations WHERE text IS NOT NULL AND modality = ?", (modality,)).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM observations WHERE text IS NOT NULL").fetchall()
        scored: list[tuple[str, float, Observation]] = []
        for row in rows:
            observation = self._observation(row)
            body_tokens = set(search_tokens(observation.text or ""))
            overlap = len(tokens & body_tokens) / max(len(tokens), 1)
            if overlap:
                scored.append((observation.segment_id, overlap, observation))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def search_label(self, modality: str, label: str, limit: int = 100) -> list[tuple[str, float, Observation]]:
        rows = self.db.execute(
            "SELECT * FROM observations WHERE modality = ? AND label LIKE ? AND confidence IS NOT NULL ORDER BY confidence DESC LIMIT ?",
            (modality, f"%{label}%", limit),
        ).fetchall()
        return [(row["segment_id"], float(row["confidence"] or 0), self._observation(row)) for row in rows]

    def search_vectors(self, vector: list[float], *, modality: str, model: str | None = None, limit: int = 50) -> list[tuple[str, float]]:
        params: list[Any] = [modality]
        sql = "SELECT segment_id, vector_json FROM embeddings WHERE modality = ?"
        if model:
            sql += " AND model = ?"
            params.append(model)
        rows = self.db.execute(sql, params).fetchall()
        scored = [(row["segment_id"], cosine(vector, json.loads(row["vector_json"]))) for row in rows]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def iter_segments(self) -> Iterable[Segment]:
        rows = self.db.execute("SELECT * FROM segments ORDER BY video_id, start_ms").fetchall()
        for row in rows:
            yield Segment(row["id"], row["video_id"], row["start_ms"], row["end_ms"], row["kind"], json.loads(row["metadata_json"]))

    def all_assets(self) -> list[VideoAsset]:
        rows = self.db.execute("SELECT * FROM assets ORDER BY id").fetchall()
        return [VideoAsset(row["id"], row["path"], row["sha256"], row["duration_ms"], row["fps"], row["width"], row["height"]) for row in rows]
