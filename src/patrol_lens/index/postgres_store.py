from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

from ..domain import EmbeddingRecord, Evidence, Segment, VideoAsset, hash_evidence
from .postgres_schema import POSTGRES_SCHEMA, POSTGRES_SCHEMA_VERSION


class PostgresTraceabilityError(RuntimeError):
    """Raised when an embedding cannot be linked to canonical raw evidence."""


def _json_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _vector_literal(vector: Sequence[float]) -> str:
    values = [float(value) for value in vector]
    if not values:
        raise ValueError("embedding vector must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding vector must contain only finite values")
    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


def _parse_vector(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        return [] if not stripped else [float(item) for item in stripped.split(",")]
    if value is None:
        return []
    return [float(item) for item in value]


def _hash_vector(vector: Sequence[float]) -> str:
    return hashlib.sha256(_vector_literal(vector).encode("ascii")).hexdigest()


class PostgresIndexStore:
    """ACID evidence store with pgvector-backed, provenance-carrying rows.

    ``root`` remains a local artifact directory for extracted frames/audio and
    agent runs. The database stores canonical evidence and its traceable
    embedding; it never needs to read those local artifacts during retrieval.
    """

    backend = "postgres"

    def __init__(
        self,
        dsn: str | None = None,
        *,
        root: str | Path = ".patrol-lens",
        connection: Any | None = None,
        vector_dimensions: int | None = None,
    ) -> None:
        self.dsn = dsn or os.getenv("PATROLLENS_DATABASE_URL")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if connection is not None:
            self.db = connection
        else:
            if not self.dsn:
                raise RuntimeError(
                    "PostgreSQL backend requires --database-url or PATROLLENS_DATABASE_URL"
                )
            self.db = self._connect(self.dsn)
        self._initialize_schema()
        if vector_dimensions is not None:
            self.ensure_vector_index(vector_dimensions)

    @staticmethod
    def _connect(dsn: str) -> Any:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is not installed; install patrol-lens[postgres]"
            ) from exc
        try:
            return psycopg.connect(dsn, row_factory=dict_row)
        except Exception as exc:
            raise RuntimeError(f"could not connect to PostgreSQL: {exc}") from exc

    def _initialize_schema(self) -> None:
        try:
            with self.db.cursor() as cursor:
                cursor.execute(POSTGRES_SCHEMA)
                cursor.execute(
                    """INSERT INTO pl_metadata(key, value) VALUES (%s, %s::jsonb)
                       ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value""",
                    ("schema_version", json.dumps(POSTGRES_SCHEMA_VERSION)),
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            yield
        except Exception:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    @staticmethod
    def _asset_source_uri(asset: VideoAsset) -> str:
        source_uri = asset.metadata.get("source_uri")
        return str(source_uri) if source_uri else asset.path

    def set_metadata(self, key: str, value: Any) -> None:
        with self.transaction(), self.db.cursor() as cursor:
            cursor.execute(
                """INSERT INTO pl_metadata(key, value) VALUES (%s, %s::jsonb)
                       ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value""",
                (key, json.dumps(value, default=str)),
            )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT value FROM pl_metadata WHERE key = %s", (key,))
            row = cursor.fetchone()
        return default if row is None else _json_value(row["value"], default)

    def upsert_asset(self, asset: VideoAsset) -> None:
        with self.transaction(), self.db.cursor() as cursor:
            cursor.execute(
                """INSERT INTO pl_assets
                   (id, source_uri, sha256, duration_ms, fps, width, height, has_audio, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                   ON CONFLICT(id) DO UPDATE SET
                     source_uri = EXCLUDED.source_uri, sha256 = EXCLUDED.sha256,
                     duration_ms = EXCLUDED.duration_ms, fps = EXCLUDED.fps,
                     width = EXCLUDED.width, height = EXCLUDED.height,
                     has_audio = EXCLUDED.has_audio, metadata = EXCLUDED.metadata,
                     updated_at = CURRENT_TIMESTAMP""",
                (
                    asset.id,
                    self._asset_source_uri(asset),
                    asset.sha256,
                    asset.duration_ms,
                    asset.fps,
                    asset.width,
                    asset.height,
                    asset.has_audio,
                    json.dumps(asset.metadata, default=str),
                ),
            )

    @staticmethod
    def _asset(row: dict[str, Any]) -> VideoAsset:
        return VideoAsset(
            id=row["id"],
            path=row["source_uri"],
            sha256=row["sha256"],
            duration_ms=int(row["duration_ms"]),
            fps=row["fps"],
            width=row["width"],
            height=row["height"],
            has_audio=bool(row["has_audio"]),
            metadata=_json_value(row["metadata"], {}),
        )

    def get_asset(self, asset_id: str) -> VideoAsset | None:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM pl_assets WHERE id = %s", (asset_id,))
            row = cursor.fetchone()
        return None if row is None else self._asset(row)

    def all_assets(self) -> list[VideoAsset]:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM pl_assets ORDER BY id")
            rows = cursor.fetchall()
        return [self._asset(row) for row in rows]

    def upsert_segments(self, segments: Iterable[Segment]) -> None:
        rows = [
            (
                item.id,
                item.video_id,
                item.start_ms,
                item.end_ms,
                item.kind,
                json.dumps(item.metadata, default=str),
            )
            for item in segments
        ]
        if not rows:
            return
        with self.transaction(), self.db.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO pl_segments
                       (id, video_id, start_ms, end_ms, kind, metadata)
                       VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                       ON CONFLICT(id) DO UPDATE SET
                         video_id = EXCLUDED.video_id, start_ms = EXCLUDED.start_ms,
                         end_ms = EXCLUDED.end_ms, kind = EXCLUDED.kind,
                         metadata = EXCLUDED.metadata""",
                rows,
            )

    @staticmethod
    def _segment(row: dict[str, Any]) -> Segment:
        return Segment(
            row["id"],
            row["video_id"],
            int(row["start_ms"]),
            int(row["end_ms"]),
            row["kind"],
            _json_value(row["metadata"], {}),
        )

    def iter_segments(self, video_id: str | None = None) -> Iterator[Segment]:
        with self.db.cursor() as cursor:
            if video_id:
                cursor.execute(
                    "SELECT * FROM pl_segments WHERE video_id = %s ORDER BY start_ms",
                    (video_id,),
                )
            else:
                cursor.execute("SELECT * FROM pl_segments ORDER BY video_id, start_ms")
            rows = cursor.fetchall()
        yield from (self._segment(row) for row in rows)

    def _insert_evidence_many(self, cursor: Any, evidence: list[Evidence]) -> None:
        if not evidence:
            return
        rows = [
            (
                item.id,
                item.video_id,
                item.segment_id,
                item.start_ms,
                item.end_ms,
                item.modality,
                item.content,
                item.confidence,
                item.source,
                json.dumps(item.metadata, default=str),
                hash_evidence(item),
            )
            for item in evidence
        ]
        cursor.executemany(
            """INSERT INTO pl_evidence
               (id, video_id, segment_id, start_ms, end_ms, modality, content,
                confidence, source, metadata, evidence_hash)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
               ON CONFLICT(id) DO UPDATE SET
                 video_id = EXCLUDED.video_id, segment_id = EXCLUDED.segment_id,
                 start_ms = EXCLUDED.start_ms, end_ms = EXCLUDED.end_ms,
                 modality = EXCLUDED.modality, content = EXCLUDED.content,
                 confidence = EXCLUDED.confidence, source = EXCLUDED.source,
                 metadata = EXCLUDED.metadata, evidence_hash = EXCLUDED.evidence_hash""",
            rows,
        )

    def add_evidence(self, evidence: Evidence) -> None:
        self.add_evidence_many([evidence])

    def add_evidence_many(self, items: Iterable[Evidence]) -> None:
        evidence = list(items)
        if not evidence:
            return
        with self.transaction(), self.db.cursor() as cursor:
            self._insert_evidence_many(cursor, evidence)

    @staticmethod
    def _evidence(row: dict[str, Any]) -> Evidence:
        return Evidence(
            id=row["id"],
            video_id=row["video_id"],
            segment_id=row.get("segment_id"),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            modality=row["modality"],
            content=row["content"],
            confidence=float(row["confidence"]),
            source=row["source"],
            metadata=_json_value(row.get("metadata"), {}),
        )

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM pl_evidence WHERE id = %s", (evidence_id,))
            row = cursor.fetchone()
        return None if row is None else self._evidence(row)

    def evidence_between(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
        modalities: Sequence[str] | None = None,
    ) -> list[Evidence]:
        sql = """SELECT * FROM pl_evidence
                 WHERE video_id = %s AND end_ms > %s AND start_ms < %s"""
        params: list[Any] = [video_id, start_ms, end_ms]
        if modalities:
            placeholders = ",".join("%s" for _ in modalities)
            sql += f" AND modality IN ({placeholders})"
            params.extend(modalities)
        sql += " ORDER BY start_ms, end_ms"
        with self.db.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [self._evidence(row) for row in rows]

    def iter_evidence(self, modality: str | None = None) -> Iterator[Evidence]:
        with self.db.cursor() as cursor:
            if modality:
                cursor.execute(
                    "SELECT * FROM pl_evidence WHERE modality = %s ORDER BY video_id, start_ms",
                    (modality,),
                )
            else:
                cursor.execute("SELECT * FROM pl_evidence ORDER BY video_id, start_ms")
            rows = cursor.fetchall()
        yield from (self._evidence(row) for row in rows)

    def top_evidence(self, modality: str, *, limit: int = 60) -> list[tuple[Evidence, float]]:
        with self.db.cursor() as cursor:
            cursor.execute(
                """SELECT * FROM pl_evidence WHERE modality = %s
                   ORDER BY confidence DESC, start_ms ASC LIMIT %s""",
                (modality, limit),
            )
            rows = cursor.fetchall()
        return [(self._evidence(row), float(row["confidence"])) for row in rows]

    def search_text(
        self,
        query: str,
        *,
        modalities: Sequence[str] | None = None,
        limit: int = 60,
    ) -> list[tuple[Evidence, float]]:
        if not query.strip():
            return []
        sql = """
            WITH query AS (SELECT plainto_tsquery('simple', %s) AS q)
            SELECT e.*, ts_rank_cd(to_tsvector('simple', e.content), query.q) AS score
            FROM pl_evidence e CROSS JOIN query
            WHERE to_tsvector('simple', e.content) @@ query.q
        """
        params: list[Any] = [query]
        if modalities:
            placeholders = ",".join("%s" for _ in modalities)
            sql += f" AND e.modality IN ({placeholders})"
            params.extend(modalities)
        sql += " ORDER BY score DESC, e.start_ms, e.end_ms LIMIT %s"
        params.append(limit)
        with self.db.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        if rows:
            return [(self._evidence(row), max(float(row["score"]), 0.000001)) for row in rows]

        # Keep behavior useful on installations with unusual text
        # configuration or punctuation-heavy OCR output.
        tokens = [token for token in query.split() if token.strip()]
        if not tokens:
            return []
        fallback = "SELECT * FROM pl_evidence WHERE " + " AND ".join(
            "content ILIKE %s" for _ in tokens
        )
        fallback_params: list[Any] = [f"%{token}%" for token in tokens]
        if modalities:
            placeholders = ",".join("%s" for _ in modalities)
            fallback += f" AND modality IN ({placeholders})"
            fallback_params.extend(modalities)
        fallback += " ORDER BY confidence DESC, start_ms LIMIT %s"
        fallback_params.append(limit)
        with self.db.cursor() as cursor:
            cursor.execute(fallback, fallback_params)
            fallback_rows = cursor.fetchall()
        return [
            (self._evidence(row), float(row["confidence"]))
            for row in fallback_rows
        ]

    def _embedding_source(self, cursor: Any, evidence_id: str) -> dict[str, Any]:
        cursor.execute(
            """SELECT e.*, a.source_uri AS asset_source_uri, a.sha256 AS source_sha256
               FROM pl_evidence e JOIN pl_assets a ON a.id = e.video_id
               WHERE e.id = %s""",
            (evidence_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise PostgresTraceabilityError(
                f"embedding cannot be stored: evidence {evidence_id!r} does not exist"
            )
        return row

    def _insert_embeddings(self, cursor: Any, records: list[EmbeddingRecord]) -> None:
        if not records:
            return
        dimensions: set[int] = set()
        for item in records:
            vector = list(item.vector)
            dimension = len(vector)
            if dimension != 768:
                raise ValueError(f"Expected 768, got {dimension}")
            if not item.model:
                raise PostgresTraceabilityError(f"embedding {item.id!r} has no model version")
            source = self._embedding_source(cursor, item.evidence_id)
            if source["modality"] != item.modality:
                raise PostgresTraceabilityError(
                    f"embedding {item.id!r} modality {item.modality!r} does not match "
                    f"evidence {item.evidence_id!r} modality {source['modality']!r}"
                )
            dimensions.add(dimension)
            vector_text = _vector_literal(vector)
            embedding_sql = "NULL"
            embedding_768_sql = "%s::vector(768)"
            cursor.execute(
                f"""INSERT INTO pl_embeddings
                   (id, evidence_id, video_id, start_ms, end_ms, modality, source_uri,
                    segment_id, evidence_text, evidence_metadata, evidence_source, model_version,
                    confidence, evidence_hash, source_sha256, embedding_hash, embedding,
                    embedding_768, dimensions, metadata)
                   SELECT %s, e.id, e.video_id, e.start_ms, e.end_ms, e.modality,
                          COALESCE(NULLIF(e.metadata->>'source_uri', ''),
                                   NULLIF(e.metadata->>'media_path', ''),
                                   NULLIF(e.metadata->>'frame_path', ''), a.source_uri),
                          e.segment_id, e.content, e.metadata, e.source, %s, e.confidence,
                          e.evidence_hash, a.sha256, %s, {embedding_sql},
                          {embedding_768_sql}, %s, %s::jsonb
                   FROM pl_evidence e JOIN pl_assets a ON a.id = e.video_id
                   WHERE e.id = %s
                   ON CONFLICT(id) DO UPDATE SET
                     evidence_id = EXCLUDED.evidence_id, video_id = EXCLUDED.video_id,
                     segment_id = EXCLUDED.segment_id,
                     start_ms = EXCLUDED.start_ms, end_ms = EXCLUDED.end_ms,
                     modality = EXCLUDED.modality, source_uri = EXCLUDED.source_uri,
                     evidence_text = EXCLUDED.evidence_text,
                     evidence_metadata = EXCLUDED.evidence_metadata,
                     evidence_source = EXCLUDED.evidence_source,
                     model_version = EXCLUDED.model_version,
                     confidence = EXCLUDED.confidence, evidence_hash = EXCLUDED.evidence_hash,
                     source_sha256 = EXCLUDED.source_sha256,
                     embedding_hash = EXCLUDED.embedding_hash,
                     embedding = COALESCE(EXCLUDED.embedding, pl_embeddings.embedding),
                     embedding_768 = COALESCE(EXCLUDED.embedding_768, pl_embeddings.embedding_768),
                     dimensions = EXCLUDED.dimensions, metadata = EXCLUDED.metadata
                   RETURNING id""",
                (
                    item.id,
                    item.model,
                    _hash_vector(vector),
                    vector_text,
                    dimension,
                    json.dumps(item.metadata, default=str),
                    item.evidence_id,
                ),
            )
            if cursor.fetchone() is None:
                raise PostgresTraceabilityError(
                    f"embedding {item.id!r} was not linked to evidence {item.evidence_id!r}"
                )
        for dimension in dimensions:
            self._ensure_vector_index(cursor, dimension)

    def add_embeddings(self, records: Iterable[EmbeddingRecord]) -> None:
        items = list(records)
        if not items:
            return
        with self.transaction(), self.db.cursor() as cursor:
            self._insert_embeddings(cursor, items)

    def add_evidence_and_embeddings(
        self,
        evidence: Iterable[Evidence],
        records: Iterable[EmbeddingRecord],
    ) -> None:
        evidence_items = list(evidence)
        embedding_items = list(records)
        if not evidence_items and not embedding_items:
            return
        with self.transaction(), self.db.cursor() as cursor:
            self._insert_evidence_many(cursor, evidence_items)
            self._insert_embeddings(cursor, embedding_items)

    def get_cached_embedding(
        self,
        cache_key: str,
        *,
        expected_dimensions: int,
    ) -> list[float] | None:
        if expected_dimensions != 768:
            raise ValueError("PostgreSQL embedding cache is fixed at 768 dimensions")
        with self.db.cursor() as cursor:
            cursor.execute(
                """SELECT embedding, dimensions FROM pl_embedding_cache
                   WHERE cache_key = %s""",
                (cache_key,),
            )
            row = cursor.fetchone()
        if row is None or int(row["dimensions"]) != expected_dimensions:
            return None
        with self.transaction(), self.db.cursor() as cursor:
            cursor.execute(
                """UPDATE pl_embedding_cache SET last_used_at = CURRENT_TIMESTAMP
                   WHERE cache_key = %s""",
                (cache_key,),
            )
        return _parse_vector(row["embedding"])

    def get_cached_embeddings(
        self,
        cache_keys: list[str],
        *,
        expected_dimensions: int,
    ) -> dict[str, list[float]]:
        if expected_dimensions != 768:
            raise ValueError("PostgreSQL embedding cache is fixed at 768 dimensions")
        if not cache_keys:
            return {}
        unique = list(dict.fromkeys(cache_keys))
        with self.transaction(), self.db.cursor() as cursor:
            cursor.execute(
                """SELECT cache_key, embedding FROM pl_embedding_cache
                   WHERE dimensions = 768 AND cache_key = ANY(%s)""",
                (unique,),
            )
            rows = cursor.fetchall()
            found = {row["cache_key"]: _parse_vector(row["embedding"]) for row in rows}
            if found:
                cursor.execute(
                    """UPDATE pl_embedding_cache SET last_used_at = CURRENT_TIMESTAMP
                       WHERE cache_key = ANY(%s)""",
                    (list(found),),
                )
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
        if dimensions != 768 or len(vector) != 768:
            raise ValueError(f"Expected 768, got {len(vector)}")
        vector_text = _vector_literal(vector)
        with self.transaction(), self.db.cursor() as cursor:
            cursor.execute(
                """INSERT INTO pl_embedding_cache
                   (cache_key, content_hash, modality, model_version, dimensions,
                    preprocessing_version, embedding, embedding_hash)
                   VALUES (%s, %s, %s, %s, 768, %s, %s::vector(768), %s)
                   ON CONFLICT(cache_key) DO UPDATE SET
                     embedding = EXCLUDED.embedding,
                     embedding_hash = EXCLUDED.embedding_hash,
                     last_used_at = CURRENT_TIMESTAMP""",
                (
                    cache_key,
                    content_hash,
                    modality,
                    model,
                    preprocessing_version,
                    vector_text,
                    _hash_vector(vector),
                ),
            )

    def migrate_embeddings_768(
        self,
        embedding_backend: Any,
        *,
        batch_size: int = 6,
    ) -> dict[str, int]:
        """Re-embed canonical text and keyframe evidence into the 768-d space.

        The canonical source is ``pl_evidence``. Existing legacy vectors in
        ``pl_embeddings.embedding`` are retained while the new vector is
        written to ``embedding_768``. Media paths are read from the evidence
        metadata. Only visual image files and transcript/OCR text are eligible;
        legacy video/audio evidence remains unembedded and is regenerated as
        local evidence by the optimized ingestion pipeline.
        """

        if batch_size <= 0:
            raise ValueError("embedding migration batch size must be positive")
        if getattr(embedding_backend, "dimensions", 768) != 768:
            raise ValueError("embedding migration requires a 768-dimensional backend")
        model = str(getattr(embedding_backend, "model_name", "")).strip()
        if not model:
            raise ValueError("embedding migration backend must provide model_name")

        with self.db.cursor() as cursor:
            cursor.execute(
                """SELECT e.*,
                          p.id AS existing_embedding_id,
                          p.metadata AS existing_embedding_metadata,
                          p.embedding_768 AS existing_embedding_768,
                          p.dimensions AS existing_dimensions
                   FROM pl_evidence e
                   LEFT JOIN LATERAL (
                       SELECT id, metadata, embedding_768, dimensions, created_at
                         FROM pl_embeddings
                        WHERE evidence_id = e.id
                        ORDER BY CASE WHEN embedding_768 IS NOT NULL
                                           AND dimensions = 768 THEN 0 ELSE 1 END,
                                 created_at DESC
                        LIMIT 1
                   ) p ON TRUE
                   WHERE p.id IS NULL
                      OR p.embedding_768 IS NULL
                      OR p.dimensions <> 768
                   ORDER BY e.video_id, e.start_ms, e.id"""
            )
            candidates = cursor.fetchall()

        stats = {
            "candidates": len(candidates),
            "migrated": 0,
            "text": 0,
            "media": 0,
            "skipped_local_only": 0,
        }
        for offset in range(0, len(candidates), batch_size):
            chunk = candidates[offset : offset + batch_size]
            text_items: list[dict[str, Any]] = []
            media_items: list[tuple[dict[str, Any], Path]] = []
            for row in chunk:
                modality = str(row.get("modality", ""))
                evidence_metadata = _json_value(row.get("metadata"), {}) or {}
                media_value = evidence_metadata.get("media_path") or evidence_metadata.get("frame_path")
                if modality in {"transcript", "ocr"}:
                    text_items.append(row)
                elif modality == "visual" and media_value:
                    media_path = Path(str(media_value))
                    if not media_path.is_absolute():
                        media_path = self.root / media_path
                    if not media_path.is_file():
                        raise FileNotFoundError(
                            f"source media for evidence {row['id']!r} not found: {media_path}"
                        )
                    if media_path.suffix.lower() in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}:
                        media_items.append((row, media_path))
                    else:
                        stats["skipped_local_only"] += 1
                else:
                    stats["skipped_local_only"] += 1

            migrated: list[tuple[dict[str, Any], list[float], str]] = []
            if text_items:
                vectors = embedding_backend.encode_texts([str(row["content"]) for row in text_items])
                if len(vectors) != len(text_items):
                    raise RuntimeError(
                        f"embedding migration returned {len(vectors)} text vectors for "
                        f"{len(text_items)} evidence rows"
                    )
                migrated.extend(
                    (row, vector, "text")
                    for row, vector in zip(text_items, vectors)
                )
            if media_items:
                paths = [path for _row, path in media_items]
                encode_media_many = getattr(embedding_backend, "encode_media_many", None)
                if callable(encode_media_many):
                    vectors = encode_media_many(paths)
                else:
                    vectors = [embedding_backend.encode_media(path) for path in paths]
                if len(vectors) != len(media_items):
                    raise RuntimeError(
                        f"embedding migration returned {len(vectors)} media vectors for "
                        f"{len(media_items)} evidence rows"
                    )
                migrated.extend(
                    (row, vector, "media")
                    for (row, _path), vector in zip(media_items, vectors)
                )

            records = []
            for row, vector, input_kind in migrated:
                metadata = _json_value(row.get("existing_embedding_metadata"), {}) or {}
                metadata.update(
                    {
                        "embedding_input": input_kind,
                        "api_model": model,
                        "migration": "embedding_768",
                    }
                )
                records.append(
                    EmbeddingRecord(
                        id=row.get("existing_embedding_id") or f"{row['id']}-embedding",
                        evidence_id=row["id"],
                        modality=row["modality"],
                        model=model,
                        vector=list(vector),
                        metadata=metadata,
                    )
                )
            self.add_embeddings(records)
            stats["migrated"] += len(records)
            stats["text"] += len(text_items)
            stats["media"] += len(media_items)
        return stats

    def ensure_vector_index(self, dimensions: int) -> None:
        if dimensions != 768:
            raise ValueError("PostgreSQL HNSW indexing is fixed at 768 dimensions")
        with self.transaction(), self.db.cursor() as cursor:
            self._ensure_vector_index(cursor, dimensions)

    @staticmethod
    def _ensure_vector_index(cursor: Any, dimensions: int) -> None:
        if dimensions != 768:
            raise ValueError("PostgreSQL HNSW indexing is fixed at 768 dimensions")
        cursor.execute(
            """CREATE INDEX IF NOT EXISTS pl_embeddings_embedding_768_hnsw
               ON pl_embeddings USING hnsw
               (embedding_768 vector_cosine_ops)
               WHERE embedding_768 IS NOT NULL"""
        )

    def embedding_records(self, modality: str, model: str) -> list[tuple[str, list[float]]]:
        with self.db.cursor() as cursor:
            cursor.execute(
                """SELECT id, embedding_768 AS embedding
                   FROM pl_embeddings
                   WHERE modality = %s AND model_version = %s AND dimensions = 768
                     AND embedding_768 IS NOT NULL
                   ORDER BY id""",
                (modality, model),
            )
            rows = cursor.fetchall()
        return [(row["id"], _parse_vector(row["embedding"])) for row in rows]

    @staticmethod
    def _embedding_evidence(row: dict[str, Any]) -> Evidence:
        metadata = _json_value(row.get("evidence_metadata"), {}) or {}
        metadata.update(_json_value(row.get("metadata"), {}) or {})
        metadata.update(
            {
                "source_uri": row["source_uri"],
                "embedding_id": row["id"],
                "model_version": row["model_version"],
                "evidence_hash": row["evidence_hash"],
                "source_sha256": row["source_sha256"],
                "embedding_hash": row["embedding_hash"],
            }
        )
        return Evidence(
            id=row["evidence_id"],
            video_id=row["video_id"],
            segment_id=row.get("segment_id"),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            modality=row["modality"],
            content=row["evidence_text"],
            confidence=float(row["confidence"]),
            source=row["evidence_source"],
            metadata=metadata,
        )

    def search_vectors(
        self,
        vector: list[float],
        *,
        modality: str,
        model: str,
        limit: int = 60,
    ) -> list[tuple[Evidence, float]]:
        if limit <= 0:
            return []
        dimensions = len(vector)
        if dimensions != 768:
            raise ValueError(f"Expected 768, got {dimensions}")
        vector_text = _vector_literal(vector)
        vector_expression = "e.embedding_768"
        vector_available = "e.embedding_768 IS NOT NULL"
        with self.db.cursor() as cursor:
            cursor.execute(
                f"""SELECT e.*,
                          1 - ({vector_expression} <=> (%s::vector({dimensions}))) AS similarity
                   FROM pl_embeddings e
                   WHERE e.modality = %s AND e.model_version = %s AND e.dimensions = %s
                     AND {vector_available}
                   ORDER BY {vector_expression} <=> (%s::vector({dimensions}))
                   LIMIT %s""",
                (vector_text, modality, model, dimensions, vector_text, limit),
            )
            rows = cursor.fetchall()
        return [
            (self._embedding_evidence(row), float(row["similarity"]))
            for row in rows
        ]

    def get_embedding_trace(self, embedding_id: str) -> dict[str, Any] | None:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM pl_embeddings WHERE id = %s", (embedding_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        vector = (
            row.get("embedding_768")
            if int(row.get("dimensions", 0)) == 768 and row.get("embedding_768") is not None
            else row.get("embedding")
        )
        return {
            "embedding_id": row["id"],
            "evidence_id": row["evidence_id"],
            "video_id": row["video_id"],
            "segment_id": row["segment_id"],
            "start_ms": int(row["start_ms"]),
            "end_ms": int(row["end_ms"]),
            "modality": row["modality"],
            "source_uri": row["source_uri"],
            "evidence": row["evidence_text"],
            "evidence_metadata": _json_value(row["evidence_metadata"], {}),
            "evidence_source": row["evidence_source"],
            "model_version": row["model_version"],
            "confidence": float(row["confidence"]),
            "evidence_hash": row["evidence_hash"],
            "hash": row["evidence_hash"],
            "source_sha256": row["source_sha256"],
            "embedding_hash": row["embedding_hash"],
            "embedding": _parse_vector(vector),
            "dimensions": int(row["dimensions"]),
            "metadata": _json_value(row["metadata"], {}),
        }

    trace_embedding = get_embedding_trace

    def clear_asset_evidence(self, video_id: str) -> None:
        with self.transaction(), self.db.cursor() as cursor:
            # pl_embeddings are deleted by the evidence FK cascade.
            cursor.execute("DELETE FROM pl_evidence WHERE video_id = %s", (video_id,))
            cursor.execute("DELETE FROM pl_segments WHERE video_id = %s", (video_id,))

    def mark_ingestion(
        self,
        video_id: str,
        fingerprint: str,
        status: str,
        stats: dict[str, Any],
    ) -> None:
        with self.transaction(), self.db.cursor() as cursor:
            cursor.execute(
                """INSERT INTO pl_ingestion_runs(video_id, fingerprint, status, stats)
                       VALUES (%s, %s, %s, %s::jsonb)
                       ON CONFLICT(video_id, fingerprint) DO UPDATE SET
                         status = EXCLUDED.status, stats = EXCLUDED.stats,
                         updated_at = CURRENT_TIMESTAMP""",
                (video_id, fingerprint, status, json.dumps(stats, default=str)),
            )

    def ingestion_status(self, video_id: str, fingerprint: str) -> dict[str, Any] | None:
        with self.db.cursor() as cursor:
            cursor.execute(
                """SELECT status, stats FROM pl_ingestion_runs
                   WHERE video_id = %s AND fingerprint = %s""",
                (video_id, fingerprint),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {"status": row["status"], "stats": _json_value(row["stats"], {})}

    def completed_ingestion_fingerprints(self, video_id: str) -> list[str]:
        with self.db.cursor() as cursor:
            cursor.execute(
                """SELECT fingerprint FROM pl_ingestion_runs
                   WHERE video_id = %s AND status = 'complete'""",
                (video_id,),
            )
            rows = cursor.fetchall()
        return [row["fingerprint"] for row in rows]

    def supersede_ingestions(self, video_id: str, active_fingerprint: str) -> None:
        with self.transaction(), self.db.cursor() as cursor:
            cursor.execute(
                """UPDATE pl_ingestion_runs SET status = 'superseded', updated_at = CURRENT_TIMESTAMP
                   WHERE video_id = %s AND fingerprint <> %s AND status = 'complete'""",
                (video_id, active_fingerprint),
            )
