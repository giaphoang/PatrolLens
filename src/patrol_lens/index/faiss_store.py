from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from ..domain import Evidence
from .postgres_store import PostgresIndexStore
from .sqlite_store import IndexStore


class VectorIndex(Protocol):
    def search(
        self, vector: list[float], *, modality: str, model: str, limit: int
    ) -> list[tuple[Evidence, float]]: ...


class SQLiteVectorIndex:
    """Portable exact-search fallback used when FAISS is unavailable."""

    def __init__(self, store: IndexStore) -> None:
        self.store = store

    def search(
        self, vector: list[float], *, modality: str, model: str, limit: int = 60
    ) -> list[tuple[Evidence, float]]:
        return self.store.search_vectors(vector, modality=modality, model=model, limit=limit)


class PostgresVectorIndex:
    """pgvector search over provenance-carrying PostgreSQL embedding rows."""

    def __init__(self, store: PostgresIndexStore) -> None:
        self.store = store

    def rebuild(self, *, modality: str, model: str) -> int:
        # HNSW indexes are created per vector dimension when embeddings are
        # inserted. This method keeps the same writer interface as FAISS.
        return len(self.store.embedding_records(modality, model))

    def search(
        self, vector: list[float], *, modality: str, model: str, limit: int = 60
    ) -> list[tuple[Evidence, float]]:
        return self.store.search_vectors(vector, modality=modality, model=model, limit=limit)


class FaissVectorIndex:
    """Persistent inner-product FAISS index backed by canonical SQLite vectors."""

    def __init__(self, store: IndexStore) -> None:
        self.store = store
        self.root = store.root / "faiss"
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def available() -> bool:
        try:
            import faiss  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            return False
        return True

    def _paths(self, modality: str, model: str, dimensions: int) -> tuple[Path, Path]:
        key = hashlib.sha256(f"{modality}:{model}:{dimensions}".encode()).hexdigest()[:20]
        return self.root / f"{key}.faiss", self.root / f"{key}.json"

    def rebuild(self, *, modality: str, model: str) -> int:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("faiss-cpu and numpy are required to build the FAISS index") from exc
        # A model namespace may contain vectors at more than one configured
        # dimension over time. Remove indexes for this namespace before
        # writing the current per-dimension set so stale vectors are never
        # selected after a rebuild.
        for map_path in self.root.glob("*.json"):
            try:
                mapping = json.loads(map_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(mapping, dict) and mapping.get("modality") == modality and mapping.get("model") == model:
                map_path.unlink(missing_ok=True)
                map_path.with_suffix(".faiss").unlink(missing_ok=True)
        records = self.store.embedding_records(modality, model)
        by_dimensions: dict[int, list[tuple[str, list[float]]]] = {}
        for identifier, vector in records:
            by_dimensions.setdefault(len(vector), []).append((identifier, vector))
        for dimensions, dimension_records in by_dimensions.items():
            index_path, map_path = self._paths(modality, model, dimensions)
            identifiers = [item[0] for item in dimension_records]
            matrix = np.asarray([item[1] for item in dimension_records], dtype="float32")
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(dimensions)
            index.add(matrix)
            faiss.write_index(index, str(index_path))
            map_path.write_text(json.dumps({"modality": modality, "model": model, "dimensions": dimensions, "ids": identifiers}))
        return len(records)

    def search(
        self, vector: list[float], *, modality: str, model: str, limit: int = 60
    ) -> list[tuple[Evidence, float]]:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("faiss-cpu and numpy are required for FAISS search") from exc
        dimensions = len(vector)
        index_path, map_path = self._paths(modality, model, dimensions)
        if not index_path.exists() or not map_path.exists():
            return []
        mapping = json.loads(map_path.read_text())
        identifiers: list[str] = mapping["ids"]
        query = np.asarray([vector], dtype="float32")
        faiss.normalize_L2(query)
        index = faiss.read_index(str(index_path))
        scores, positions = index.search(query, min(limit, len(identifiers)))
        results: list[tuple[Evidence, float]] = []
        for position, score in zip(positions[0], scores[0]):
            if position < 0:
                continue
            evidence = self.store.get_evidence(identifiers[int(position)])
            if evidence:
                results.append((evidence, float(score)))
        return results


class AutoVectorIndex:
    """Use persisted FAISS when present; otherwise retain exact SQLite behavior."""

    def __init__(self, store: IndexStore) -> None:
        self.store = store
        self.faiss = FaissVectorIndex(store)
        self.sqlite = SQLiteVectorIndex(store)

    def rebuild(self, *, modality: str, model: str) -> int:
        if self.faiss.available():
            return self.faiss.rebuild(modality=modality, model=model)
        return len(self.store.embedding_records(modality, model))

    def search(
        self, vector: list[float], *, modality: str, model: str, limit: int = 60
    ) -> list[tuple[Evidence, float]]:
        index_path, map_path = self.faiss._paths(modality, model, len(vector))
        if self.faiss.available() and index_path.exists() and map_path.exists():
            return self.faiss.search(vector, modality=modality, model=model, limit=limit)
        return self.sqlite.search(vector, modality=modality, model=model, limit=limit)
