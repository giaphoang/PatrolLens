from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ..adapters.asr import ASRBackend, WordSpan
from ..adapters.clap import AudioEmbeddingBackend, clap_intervals
from ..adapters.media import (
    VisualKeyframe,
    deduplicate_keyframes,
    extract_audio,
    extract_frame_sequence,
    iter_segments,
    probe_video,
    sha256_file,
)
from ..adapters.ocr import OCRBackend
from ..config import IngestionConfig
from ..domain import EmbeddingRecord, Evidence, Segment, VideoAsset
from ..index.faiss_store import AutoVectorIndex, PostgresVectorIndex
from ..index.postgres_store import PostgresIndexStore
from ..index.sqlite_store import IndexStore
from ..runtime_metrics import peak_rss_mb


class VisualBackend(Protocol):
    model_name: str

    def encode_image(self, image_path: str) -> list[float]: ...

    def encode_images(self, image_paths: list[str]) -> list[list[float]]: ...


class EmbeddingBackend(Protocol):
    model_name: str
    dimensions: int

    def encode_texts(self, texts: list[str]) -> list[list[float]]: ...

    def encode_media_many(
        self,
        paths: list[str | Path],
        *,
        context_texts: list[str] | None = None,
    ) -> list[list[float]]: ...


@dataclass(frozen=True)
class IngestionBackends:
    visual: VisualBackend | None = None
    asr: ASRBackend | None = None
    ocr: OCRBackend | None = None
    audio_embedding: AudioEmbeddingBackend | None = None
    embedding: EmbeddingBackend | None = None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _chunks(words: list[WordSpan], max_ms: int = 16_000, silence_ms: int = 1_200) -> list[list[WordSpan]]:
    chunks: list[list[WordSpan]] = []
    current: list[WordSpan] = []
    for word in words:
        starts_new = current and (
            word.start_ms - current[-1].end_ms > silence_ms
            or word.end_ms - current[0].start_ms > max_ms
        )
        if starts_new:
            chunks.append(current)
            current = []
        current.append(word)
    if current:
        chunks.append(current)
    return chunks


class IngestionPipeline:
    """One-time, restartable extraction of timestamped multimodal evidence."""

    def __init__(
        self,
        store: IndexStore | PostgresIndexStore,
        *,
        backends: IngestionBackends | None = None,
        config: IngestionConfig | None = None,
        vector_index: AutoVectorIndex | PostgresVectorIndex | None = None,
    ) -> None:
        self.store = store
        self.backends = backends or IngestionBackends()
        self.config = config or IngestionConfig()
        if isinstance(store, PostgresIndexStore) and self.config.embedding_dimensions != 768:
            raise ValueError("PostgreSQL ingestion requires 768-dimensional embeddings")
        backend_dimensions = getattr(self.backends.embedding, "dimensions", self.config.embedding_dimensions)
        if self.backends.embedding and backend_dimensions != self.config.embedding_dimensions:
            raise ValueError(
                f"Expected {self.config.embedding_dimensions}, got backend configuration "
                f"{backend_dimensions}"
            )
        audio_dimensions = getattr(self.backends.audio_embedding, "dimensions", 512)
        if self.backends.audio_embedding and audio_dimensions != 512:
            raise ValueError(
                f"Expected 512 CLAP dimensions, got backend configuration {audio_dimensions}"
            )
        self._keyframes: dict[str, list[VisualKeyframe]] = {}
        self._cache_stats = {
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
            "embedding_api_calls": 0,
            "clap_cache_hits": 0,
            "clap_cache_misses": 0,
            "clap_model_calls": 0,
        }
        if vector_index is not None:
            self.vector_index = vector_index
        elif isinstance(store, PostgresIndexStore):
            # Keep library callers on the production vector backend even when
            # they do not explicitly inject an index implementation.
            self.vector_index = PostgresVectorIndex(store)
        else:
            self.vector_index = AutoVectorIndex(store)

    def _fingerprint(self) -> str:
        payload = {
            "config": asdict(self.config),
            "backends": {
                name: getattr(getattr(self.backends, name), "model_name", None)
                for name in (
                    "visual",
                    "embedding",
                    "asr",
                    "ocr",
                    "audio_embedding",
                )
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]

    def ingest_path(self, path: str | Path, *, force: bool = False) -> dict[str, Any]:
        return self.ingest_asset(probe_video(path), force=force)

    @staticmethod
    def _add_runtime_metrics(stats: dict[str, Any], started: float) -> dict[str, Any]:
        stats["latency_seconds"] = round(time.perf_counter() - started, 3)
        stats["peak_rss_mb"] = peak_rss_mb()
        return stats

    def ingest_asset(self, asset: VideoAsset, *, force: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        fingerprint = self._fingerprint()
        previous = self.store.ingestion_status(asset.id, fingerprint)
        if previous and previous["status"] == "complete" and not force:
            return self._add_runtime_metrics({**previous["stats"], "skipped": True}, started)

        self.store.upsert_asset(asset)
        completed = self.store.completed_ingestion_fingerprints(asset.id)
        expected_clap = (
            len(
                clap_intervals(
                    asset.duration_ms,
                    window_ms=self.config.clap_window_ms,
                    stride_ms=self.config.clap_stride_ms,
                )
            )
            if self.backends.audio_embedding and asset.has_audio
            else 0
        )
        existing_clap = (
            self.store.evidence_count(
                asset.id,
                modality="audio_event",
                source=self.backends.audio_embedding.model_name,
            )
            if expected_clap and self.backends.audio_embedding
            else 0
        )
        needs_clap_backfill = expected_clap > existing_clap
        if completed and not force and not needs_clap_backfill:
            preserved = self.store.ingestion_status(asset.id, completed[0])
            if preserved:
                return self._add_runtime_metrics(
                    {
                        **preserved["stats"],
                        "skipped": True,
                        "preserved_completed": True,
                    },
                    started,
                )
        clap_only_backfill = bool(completed and not force and needs_clap_backfill)
        if force:
            self.store.clear_asset_evidence(asset.id)
            self.store.supersede_ingestions(asset.id, fingerprint)
        segments = list(
            iter_segments(
                asset,
                window_ms=self.config.window_ms,
                stride_ms=self.config.stride_ms,
            )
        )
        self.store.upsert_segments(segments)
        stats: dict[str, Any] = {
            "video_id": asset.id,
            "duration_s": round(asset.duration_ms / 1000, 3),
            "segments": len(segments),
            "transcript": 0,
            "transcript_reused": False,
            "visual": 0,
            "ocr": 0,
            "audio_embeddings": 0,
            "visual_vectors": 0,
            "video_embeddings": 0,
            "image_embeddings": 0,
            "sampled_frames": 0,
            "visual_keyframes": 0,
            "deduplicated_frames": 0,
            "embedding_vectors": 0,
            "fingerprint": fingerprint,
            "skipped": False,
            "additive_clap_backfill": clap_only_backfill,
        }
        self._cache_stats = {key: 0 for key in self._cache_stats}
        self.store.mark_ingestion(asset.id, fingerprint, "started", stats)
        try:
            audio_path = (
                self._audio_path(asset)
                if not clap_only_backfill and self.backends.asr
                else None
            )
            if not clap_only_backfill and self.backends.asr and audio_path:
                existing_transcripts = self.store.evidence_count(
                    asset.id,
                    modality="transcript",
                )
                if self.config.reuse_existing_transcripts and existing_transcripts and not force:
                    stats["transcript"] = existing_transcripts
                    stats["transcript_reused"] = True
                else:
                    stats["transcript"], transcript_vectors = self._ingest_transcript(
                        asset,
                        segments,
                        audio_path,
                    )
                    stats["embedding_vectors"] += transcript_vectors
            if not clap_only_backfill and (self.backends.visual or self.backends.ocr):
                visual, ocr, vectors, ocr_vectors = self._ingest_frames(asset, segments)
                stats.update(visual=visual, ocr=ocr, visual_vectors=vectors)
                stats["embedding_vectors"] += ocr_vectors
            if not clap_only_backfill and self.backends.embedding:
                if self.config.embed_images:
                    image_vectors, sampled, duplicates = self._ingest_image_embeddings(asset, segments)
                    stats["image_embeddings"] = image_vectors
                    stats["sampled_frames"] = sampled
                    stats["visual_keyframes"] = image_vectors
                    stats["deduplicated_frames"] = duplicates
                    stats["embedding_vectors"] += stats["image_embeddings"]
            if self.backends.audio_embedding and asset.has_audio:
                stats["audio_embeddings"] = self._ingest_audio_embeddings(
                    asset,
                    segments,
                )
                stats["embedding_vectors"] += stats["audio_embeddings"]
            rebuild = getattr(self.vector_index, "rebuild", None)
            if callable(rebuild) and self.backends.visual:
                rebuild(modality="visual", model=self.backends.visual.model_name)
            if callable(rebuild) and self.backends.embedding:
                for modality in ("visual", "transcript", "ocr"):
                    rebuild(modality=modality, model=self.backends.embedding.model_name)
            if callable(rebuild) and self.backends.audio_embedding:
                rebuild(
                    modality="audio_event",
                    model=self.backends.audio_embedding.model_name,
                )
            self.store.set_metadata("index_version", self.config.schema_version)
            self.store.set_metadata("ingestion_fingerprint", fingerprint)
            if self.backends.embedding:
                self.store.set_metadata("embedding_model", self.backends.embedding.model_name)
                self.store.set_metadata("embedding_dimensions", self.config.embedding_dimensions)
            if self.backends.audio_embedding:
                self.store.set_metadata(
                    "audio_embedding_model",
                    self.backends.audio_embedding.model_name,
                )
                self.store.set_metadata("audio_embedding_dimensions", 512)
            stats.update(self._cache_stats)
            self._add_runtime_metrics(stats, started)
            self.store.mark_ingestion(asset.id, fingerprint, "complete", stats)
            return stats
        except Exception as exc:
            stats.update(self._cache_stats)
            self._add_runtime_metrics(stats, started)
            self.store.mark_ingestion(asset.id, fingerprint, "failed", {**stats, "error": str(exc)})
            raise

    def _audio_path(self, asset: VideoAsset) -> Path | None:
        if not asset.has_audio:
            return None
        path = self.store.root / "media" / "audio" / f"{asset.id}.wav"
        if not path.exists():
            extract_audio(asset.path, path)
        return path

    def _segment_for(self, segments: list[Segment], timestamp_ms: int) -> str | None:
        if not segments:
            return None
        ordinal = min(len(segments) - 1, max(0, timestamp_ms // self.config.stride_ms))
        return segments[ordinal].id

    def _cache_descriptor(
        self,
        content_hash: str,
        modality: str,
        input_kind: str,
    ) -> tuple[str, str]:
        assert self.backends.embedding is not None
        preprocessing_version = f"{self.config.embedding_preprocessing_version}:{input_kind}"
        payload = "\0".join(
            (
                content_hash,
                modality,
                self.backends.embedding.model_name,
                str(self.config.embedding_dimensions),
                preprocessing_version,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), preprocessing_version

    def _validate_embedding(self, vector: list[float]) -> list[float]:
        actual = len(vector)
        if actual != self.config.embedding_dimensions:
            raise RuntimeError(f"Expected {self.config.embedding_dimensions}, got {actual}")
        return [float(value) for value in vector]

    def _resolve_cached_embeddings(
        self,
        inputs: list[tuple[str, str, str, str | Path]],
        *,
        encode: Any,
    ) -> tuple[list[list[float]], list[dict[str, str]]]:
        """Resolve vectors by durable key and persist provider responses immediately."""

        assert self.backends.embedding is not None
        resolved: list[list[float] | None] = [None] * len(inputs)
        descriptors = []
        missing: dict[str, list[int]] = {}
        for content_hash, modality, input_kind, _payload in inputs:
            cache_key, preprocessing_version = self._cache_descriptor(
                content_hash, modality, input_kind
            )
            descriptors.append(
                {
                    "cache_key": cache_key,
                    "content_hash": content_hash,
                    "preprocessing_version": preprocessing_version,
                }
            )
        cache_keys = [descriptor["cache_key"] for descriptor in descriptors]
        get_many = getattr(self.store, "get_cached_embeddings", None)
        if callable(get_many):
            checked = get_many(
                cache_keys,
                expected_dimensions=self.config.embedding_dimensions,
            )
        else:
            checked = {
                key: vector
                for key in dict.fromkeys(cache_keys)
                if (
                    vector := self.store.get_cached_embedding(
                        key,
                        expected_dimensions=self.config.embedding_dimensions,
                    )
                )
                is not None
            }
        for index, descriptor in enumerate(descriptors):
            cache_key = descriptor["cache_key"]
            cached = checked.get(cache_key)
            if cached is not None:
                resolved[index] = self._validate_embedding(cached)
                self._cache_stats["embedding_cache_hits"] += 1
            else:
                missing.setdefault(cache_key, []).append(index)

        if missing:
            unique_indexes = [indexes[0] for indexes in missing.values()]
            vectors = encode([inputs[index][3] for index in unique_indexes])
            self._cache_stats["embedding_api_calls"] += 1
            if len(vectors) != len(unique_indexes):
                raise RuntimeError(
                    f"embedding backend returned {len(vectors)} vectors for "
                    f"{len(unique_indexes)} unique inputs"
                )
            for index, vector in zip(unique_indexes, vectors):
                checked_vector = self._validate_embedding(vector)
                content_hash, modality, _input_kind, _payload = inputs[index]
                descriptor = descriptors[index]
                self.store.put_cached_embedding(
                    descriptor["cache_key"],
                    content_hash=content_hash,
                    modality=modality,
                    model=self.backends.embedding.model_name,
                    dimensions=self.config.embedding_dimensions,
                    preprocessing_version=descriptor["preprocessing_version"],
                    vector=checked_vector,
                )
                indexes = missing[descriptor["cache_key"]]
                for target in indexes:
                    resolved[target] = checked_vector
                self._cache_stats["embedding_cache_misses"] += 1
                self._cache_stats["embedding_cache_hits"] += len(indexes) - 1

        if any(vector is None for vector in resolved):
            raise RuntimeError("embedding cache resolution left unresolved inputs")
        return [vector for vector in resolved if vector is not None], descriptors

    def _store_text_evidence(self, evidence: list[Evidence]) -> tuple[int, int]:
        """Persist exact text evidence and optional semantic vectors together."""

        if not evidence:
            return 0, 0
        if self.backends.embedding:
            total = 0
            for offset in range(0, len(evidence), self.config.embedding_batch_size):
                batch = evidence[offset : offset + self.config.embedding_batch_size]
                inputs = [
                    (
                        hashlib.sha256(" ".join(item.content.split()).encode("utf-8")).hexdigest(),
                        item.modality,
                        "text",
                        item.content,
                    )
                    for item in batch
                ]
                vectors, descriptors = self._resolve_cached_embeddings(
                    inputs,
                    encode=lambda values: self.backends.embedding.encode_texts(
                        [str(value) for value in values]
                    ),
                )
                records = [
                    EmbeddingRecord(
                        id=f"{item.id}-embedding",
                        evidence_id=item.id,
                        modality=item.modality,
                        model=self.backends.embedding.model_name,
                        vector=vector,
                        metadata={
                            "embedding_input": "text",
                            "api_model": self.backends.embedding.model_name,
                            "embedding_dimensions": self.config.embedding_dimensions,
                            **descriptor,
                        },
                    )
                    for item, vector, descriptor in zip(batch, vectors, descriptors)
                ]
                add_pair = getattr(self.store, "add_evidence_and_embeddings", None)
                if callable(add_pair):
                    add_pair(batch, records)
                else:
                    self.store.add_evidence_many(batch)
                    self.store.add_embeddings(records)
                total += len(records)
            return len(evidence), total
        self.store.add_evidence_many(evidence)
        return len(evidence), 0

    def _ingest_transcript(
        self,
        asset: VideoAsset,
        segments: list[Segment],
        audio_path: Path,
    ) -> tuple[int, int]:
        assert self.backends.asr is not None
        words = self.backends.asr.transcribe(str(audio_path))
        evidence: list[Evidence] = []
        for ordinal, chunk in enumerate(_chunks(words, max_ms=self.config.window_ms)):
            if not chunk:
                continue
            content = " ".join(item.text for item in chunk).strip()
            if not content:
                continue
            confidences = [item.confidence for item in chunk if item.confidence is not None]
            start_ms = min(item.start_ms for item in chunk)
            end_ms = max(item.end_ms for item in chunk)
            evidence.append(
                Evidence(
                    id=f"{asset.id}-transcript-{ordinal:07d}",
                    video_id=asset.id,
                    segment_id=self._segment_for(segments, start_ms),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    modality="transcript",
                    content=content,
                    confidence=sum(confidences) / len(confidences) if confidences else 0.75,
                    source=self.backends.asr.model_name,
                    metadata={
                        "word_count": len(chunk),
                        "source_reference": str(audio_path),
                        "source_hash": hashlib.sha256(
                            " ".join(content.split()).encode("utf-8")
                        ).hexdigest(),
                        "processing_version": self.config.schema_version,
                    },
                )
            )
        return self._store_text_evidence(evidence)

    def _encode_batch(self, paths: list[Path]) -> list[list[float]]:
        assert self.backends.visual is not None
        method = getattr(self.backends.visual, "encode_images", None)
        if callable(method):
            return method([str(path) for path in paths])
        return [self.backends.visual.encode_image(str(path)) for path in paths]

    def _ensure_frame_sequence(self, asset: VideoAsset) -> list[tuple[int, Path]]:
        frame_dir = self.store.root / "media" / "frames" / asset.id / f"{self.config.frame_step_ms}ms"
        manifest_path = frame_dir / "manifest.json"
        existing = sorted(frame_dir.glob("frame-*.jpg")) if frame_dir.exists() else []
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                manifest = {}
        complete = (
            manifest.get("video_sha256") == asset.sha256
            and manifest.get("frame_step_ms") == self.config.frame_step_ms
            and manifest.get("frame_count") == len(existing)
        )
        if existing and complete:
            frames = [(ordinal * self.config.frame_step_ms, path) for ordinal, path in enumerate(existing)]
        else:
            frames = extract_frame_sequence(
                asset.path,
                frame_dir,
                step_ms=self.config.frame_step_ms,
                end_ms=asset.duration_ms,
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "video_sha256": asset.sha256,
                        "frame_step_ms": self.config.frame_step_ms,
                        "frame_count": len(frames),
                    },
                    indent=2,
                )
            )
        return frames

    def _ensure_keyframes(self, asset: VideoAsset) -> list[VisualKeyframe]:
        cached = self._keyframes.get(asset.id)
        if cached is not None:
            return cached
        frames = self._ensure_frame_sequence(asset)
        manifest_dir = self.store.root / "media" / "keyframes" / asset.id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                manifest = {}
        config_matches = (
            manifest.get("video_sha256") == asset.sha256
            and manifest.get("frame_step_ms") == self.config.frame_step_ms
            and manifest.get("duplicate_distance") == self.config.visual_duplicate_distance
            and manifest.get("scene_change_distance") == self.config.visual_scene_change_distance
        )
        records = manifest.get("keyframes", []) if config_matches else []
        keyframes = [
            VisualKeyframe(
                timestamp_ms=int(item["timestamp_ms"]),
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                path=Path(item["path"]),
                perceptual_hash=str(item["perceptual_hash"]),
                frame_count=int(item["frame_count"]),
            )
            for item in records
            if Path(item.get("path", "")).is_file()
        ]
        if len(keyframes) != len(records) or not records:
            keyframes = deduplicate_keyframes(
                frames,
                frame_step_ms=self.config.frame_step_ms,
                duration_ms=asset.duration_ms,
                duplicate_distance=self.config.visual_duplicate_distance,
                scene_change_distance=self.config.visual_scene_change_distance,
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "video_sha256": asset.sha256,
                        "frame_step_ms": self.config.frame_step_ms,
                        "duplicate_distance": self.config.visual_duplicate_distance,
                        "scene_change_distance": self.config.visual_scene_change_distance,
                        "sampled_frame_count": len(frames),
                        "keyframes": [
                            {
                                "timestamp_ms": item.timestamp_ms,
                                "start_ms": item.start_ms,
                                "end_ms": item.end_ms,
                                "path": str(item.path),
                                "perceptual_hash": item.perceptual_hash,
                                "frame_count": item.frame_count,
                            }
                            for item in keyframes
                        ],
                    },
                    indent=2,
                )
            )
        self._keyframes[asset.id] = keyframes
        return keyframes

    def _ingest_frames(
        self,
        asset: VideoAsset,
        segments: list[Segment],
    ) -> tuple[int, int, int, int]:
        frames = self._ensure_frame_sequence(asset)
        visual_count = ocr_count = vector_count = 0
        ocr_vectors = 0
        if self.backends.visual:
            keyframes = self._ensure_keyframes(asset)
            for offset in range(0, len(keyframes), self.config.batch_size):
                batch = keyframes[offset : offset + self.config.batch_size]
                vectors = self._encode_batch([item.path for item in batch])
                evidence: list[Evidence] = []
                embeddings: list[EmbeddingRecord] = []
                for keyframe, vector in zip(batch, vectors):
                    evidence_id = f"{asset.id}-visual-{keyframe.timestamp_ms:010d}"
                    source_hash = sha256_file(keyframe.path)
                    item = Evidence(
                        id=evidence_id,
                        video_id=asset.id,
                        segment_id=self._segment_for(segments, keyframe.start_ms),
                        start_ms=keyframe.start_ms,
                        end_ms=keyframe.end_ms,
                        modality="visual",
                        content=(
                            f"canonical body-camera keyframe covering "
                            f"{keyframe.start_ms / 1000:.3f}s to {keyframe.end_ms / 1000:.3f}s"
                        ),
                        confidence=1.0,
                        source=self.backends.visual.model_name,
                        metadata={
                            "frame_path": str(keyframe.path),
                            "source_reference": str(keyframe.path),
                            "source_hash": source_hash,
                            "perceptual_hash": keyframe.perceptual_hash,
                            "sample_count": keyframe.frame_count,
                            "processing_version": self.config.schema_version,
                        },
                    )
                    evidence.append(item)
                    embeddings.append(
                        EmbeddingRecord(
                            id=f"{evidence_id}-embedding",
                            evidence_id=evidence_id,
                            modality="visual",
                            model=self.backends.visual.model_name,
                            vector=vector,
                            metadata={
                                "frame_path": str(keyframe.path),
                                "perceptual_hash": keyframe.perceptual_hash,
                            },
                        )
                    )
                add_pair = getattr(self.store, "add_evidence_and_embeddings", None)
                if callable(add_pair):
                    add_pair(evidence, embeddings)
                else:
                    self.store.add_evidence_many(evidence)
                    self.store.add_embeddings(embeddings)
                visual_count += len(evidence)
                vector_count += len(embeddings)

        if self.backends.ocr:
            evidence = []
            for timestamp_ms, frame_path in frames:
                results = list(self.backends.ocr.detect(str(frame_path)))
                frame_hash = sha256_file(frame_path) if results else None
                for ordinal, result in enumerate(results):
                    text = str(result.get("text", "")).strip()
                    confidence = float(result.get("confidence") or 0.0)
                    if not text or confidence < self.config.ocr_min_confidence:
                        continue
                    evidence.append(
                        Evidence(
                            id=f"{asset.id}-ocr-{timestamp_ms:010d}-{ordinal:03d}",
                            video_id=asset.id,
                            segment_id=self._segment_for(segments, timestamp_ms),
                            start_ms=timestamp_ms,
                            end_ms=min(asset.duration_ms, timestamp_ms + self.config.frame_step_ms),
                            modality="ocr",
                            content=text,
                            confidence=confidence,
                            source=self.backends.ocr.model_name,
                            metadata={
                                "frame_path": str(frame_path),
                                "source_reference": str(frame_path),
                                "source_hash": frame_hash,
                                "processing_version": self.config.schema_version,
                                "box": _json_safe(result.get("box")),
                            },
                        )
                    )
                    if len(evidence) >= 500:
                        count, vectors = self._store_text_evidence(evidence)
                        ocr_count += count
                        ocr_vectors += vectors
                        evidence = []
            count, vectors = self._store_text_evidence(evidence)
            ocr_count += count
            ocr_vectors += vectors
        return visual_count, ocr_count, vector_count, ocr_vectors

    def _store_media_evidence(
        self,
        evidence: list[Evidence],
        paths: list[Path],
        *,
        media_kind: str,
    ) -> tuple[int, int]:
        """Persist media evidence with one traceable vector per media item."""

        if not evidence:
            return 0, 0
        if not self.backends.embedding:
            self.store.add_evidence_many(evidence)
            return len(evidence), 0
        if media_kind != "image":
            raise ValueError("remote ingestion embeddings are limited to deduplicated images")
        if len(evidence) != len(paths):
            raise ValueError("media evidence and paths must have the same length")
        inputs = [
            (
                str(item.metadata.get("source_hash") or sha256_file(path)),
                item.modality,
                media_kind,
                path,
            )
            for item, path in zip(evidence, paths)
        ]
        vectors, descriptors = self._resolve_cached_embeddings(
            inputs,
            encode=lambda values: self.backends.embedding.encode_media_many(
                [Path(value) for value in values]
            ),
        )
        records = [
            EmbeddingRecord(
                id=f"{item.id}-embedding",
                evidence_id=item.id,
                modality=item.modality,
                model=self.backends.embedding.model_name,
                vector=vector,
                metadata={
                    "embedding_input": media_kind,
                    "api_model": self.backends.embedding.model_name,
                    "embedding_dimensions": self.config.embedding_dimensions,
                    "media_path": str(path),
                    **descriptor,
                },
            )
            for item, path, vector, descriptor in zip(evidence, paths, vectors, descriptors)
        ]
        add_pair = getattr(self.store, "add_evidence_and_embeddings", None)
        if callable(add_pair):
            add_pair(evidence, records)
        else:
            self.store.add_evidence_many(evidence)
            self.store.add_embeddings(records)
        return len(evidence), len(records)

    def _ingest_image_embeddings(
        self,
        asset: VideoAsset,
        segments: list[Segment],
    ) -> tuple[int, int, int]:
        """Embed only canonical images from local scene/change deduplication."""

        assert self.backends.embedding is not None
        keyframes = self._ensure_keyframes(asset)

        total = 0
        for offset in range(0, len(keyframes), self.config.embedding_batch_size):
            batch = keyframes[offset : offset + self.config.embedding_batch_size]
            source_hashes = {item.path: sha256_file(item.path) for item in batch}
            evidence = [
                Evidence(
                    id=f"{asset.id}-visual-image-{keyframe.timestamp_ms:010d}",
                    video_id=asset.id,
                    segment_id=self._segment_for(segments, keyframe.start_ms),
                    start_ms=keyframe.start_ms,
                    end_ms=keyframe.end_ms,
                    modality="visual",
                    content=(
                        f"canonical body-camera keyframe covering "
                        f"{keyframe.start_ms / 1000:.3f}s to {keyframe.end_ms / 1000:.3f}s"
                    ),
                    confidence=1.0,
                    source=self.backends.embedding.model_name,
                    metadata={
                        "media_path": str(keyframe.path),
                        "source_reference": str(keyframe.path),
                        "source_hash": source_hashes[keyframe.path],
                        "media_kind": "image",
                        "perceptual_hash": keyframe.perceptual_hash,
                        "sample_count": keyframe.frame_count,
                        "canonical_timestamp_ms": keyframe.timestamp_ms,
                        "processing_version": self.config.schema_version,
                    },
                )
                for keyframe in batch
            ]
            paths = [keyframe.path for keyframe in batch]
            _count, vectors = self._store_media_evidence(evidence, paths, media_kind="image")
            total += vectors
        sampled = sum(item.frame_count for item in keyframes)
        return total, sampled, max(0, sampled - len(keyframes))

    def _clap_cache_descriptor(
        self,
        asset: VideoAsset,
        start_ms: int,
        end_ms: int,
    ) -> tuple[str, str, str]:
        assert self.backends.audio_embedding is not None
        preprocessing = (
            f"{self.config.clap_preprocessing_version}:"
            f"{self.config.clap_window_ms}:{self.config.clap_stride_ms}"
        )
        content_hash = hashlib.sha256(
            "\0".join(
                (asset.sha256, str(start_ms), str(end_ms), "48000", "mono")
            ).encode("utf-8")
        ).hexdigest()
        cache_key = hashlib.sha256(
            "\0".join(
                (
                    content_hash,
                    "audio_event",
                    self.backends.audio_embedding.model_name,
                    "512",
                    preprocessing,
                )
            ).encode("utf-8")
        ).hexdigest()
        return cache_key, content_hash, preprocessing

    @staticmethod
    def _validate_clap_embedding(vector: list[float]) -> list[float]:
        if len(vector) != 512:
            raise RuntimeError(f"Expected 512 CLAP dimensions, got {len(vector)}")
        checked = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in checked):
            raise RuntimeError("CLAP embedding contains non-finite values")
        return checked

    def _ingest_audio_embeddings(
        self,
        asset: VideoAsset,
        segments: list[Segment],
    ) -> int:
        """Checkpoint 10-second CLAP windows independently of Gemini vectors."""

        assert self.backends.audio_embedding is not None
        intervals = clap_intervals(
            asset.duration_ms,
            window_ms=self.config.clap_window_ms,
            stride_ms=self.config.clap_stride_ms,
        )
        descriptors = [
            self._clap_cache_descriptor(asset, start_ms, end_ms)
            for start_ms, end_ms in intervals
        ]
        cache_keys = [item[0] for item in descriptors]
        cached = self.store.get_cached_embeddings(
            cache_keys,
            expected_dimensions=512,
        )
        vectors: list[list[float] | None] = [None] * len(intervals)
        missing_indexes: list[int] = []
        for index, cache_key in enumerate(cache_keys):
            if cache_key in cached:
                vectors[index] = self._validate_clap_embedding(cached[cache_key])
                self._cache_stats["clap_cache_hits"] += 1
            else:
                missing_indexes.append(index)

        if missing_indexes:
            missing_intervals = [intervals[index] for index in missing_indexes]
            encoded = iter(
                self.backends.audio_embedding.encode_audio_windows(
                    asset.path,
                    missing_intervals,
                )
            )
            for index in missing_indexes:
                try:
                    vector = self._validate_clap_embedding(next(encoded))
                except StopIteration as exc:
                    raise RuntimeError(
                        "CLAP backend returned fewer vectors than requested windows"
                    ) from exc
                cache_key, content_hash, preprocessing = descriptors[index]
                self.store.put_cached_embedding(
                    cache_key,
                    content_hash=content_hash,
                    modality="audio_event",
                    model=self.backends.audio_embedding.model_name,
                    dimensions=512,
                    preprocessing_version=preprocessing,
                    vector=vector,
                )
                vectors[index] = vector
                self._cache_stats["clap_cache_misses"] += 1
                self._cache_stats["clap_model_calls"] += 1
            try:
                next(encoded)
            except StopIteration:
                pass
            else:
                raise RuntimeError("CLAP backend returned more vectors than requested windows")

        if any(vector is None for vector in vectors):
            raise RuntimeError("CLAP cache resolution left unresolved windows")

        for offset in range(0, len(intervals), self.config.batch_size):
            evidence_batch: list[Evidence] = []
            embedding_batch: list[EmbeddingRecord] = []
            for ordinal in range(offset, min(len(intervals), offset + self.config.batch_size)):
                start_ms, end_ms = intervals[ordinal]
                cache_key, content_hash, preprocessing = descriptors[ordinal]
                evidence_id = f"{asset.id}-clap-{start_ms:010d}"
                metadata = {
                    "source_reference": asset.path,
                    "source_hash": asset.sha256,
                    "sample_rate": 48_000,
                    "window_ms": self.config.clap_window_ms,
                    "stride_ms": self.config.clap_stride_ms,
                    "padded_samples": max(
                        0,
                        (self.config.clap_window_ms - (end_ms - start_ms)) * 48,
                    ),
                    "processing_version": self.config.schema_version,
                    "preprocessing_version": preprocessing,
                    "cache_key": cache_key,
                    "content_hash": content_hash,
                }
                evidence = Evidence(
                    id=evidence_id,
                    video_id=asset.id,
                    segment_id=self._segment_for(segments, start_ms),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    modality="audio_event",
                    content=(
                        f"CLAP acoustic window from {start_ms / 1000:.3f}s "
                        f"to {end_ms / 1000:.3f}s"
                    ),
                    confidence=1.0,
                    source=self.backends.audio_embedding.model_name,
                    metadata=metadata,
                )
                vector = vectors[ordinal]
                assert vector is not None
                evidence_batch.append(evidence)
                embedding_batch.append(
                    EmbeddingRecord(
                        id=f"{evidence_id}-embedding",
                        evidence_id=evidence_id,
                        modality="audio_event",
                        model=self.backends.audio_embedding.model_name,
                        vector=vector,
                        metadata={
                            "embedding_input": "raw_audio",
                            "embedding_dimensions": 512,
                            **metadata,
                        },
                    )
                )
            self.store.add_evidence_and_embeddings(evidence_batch, embedding_batch)
        return len(intervals)
