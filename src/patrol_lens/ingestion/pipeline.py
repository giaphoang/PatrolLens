from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ..adapters.asr import ASRBackend, WordSpan
from ..adapters.audio import AudioBackend
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
    audio: AudioBackend | None = None
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
        self._keyframes: dict[str, list[VisualKeyframe]] = {}
        self._cache_stats = {
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
            "embedding_api_calls": 0,
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
                for name in ("visual", "embedding", "asr", "ocr", "audio")
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]

    def ingest_path(self, path: str | Path, *, force: bool = False) -> dict[str, Any]:
        return self.ingest_asset(probe_video(path), force=force)

    def ingest_asset(self, asset: VideoAsset, *, force: bool = False) -> dict[str, Any]:
        fingerprint = self._fingerprint()
        previous = self.store.ingestion_status(asset.id, fingerprint)
        if previous and previous["status"] == "complete" and not force:
            return {**previous["stats"], "skipped": True}

        self.store.upsert_asset(asset)
        completed = self.store.completed_ingestion_fingerprints(asset.id)
        if completed and not force:
            preserved = self.store.ingestion_status(asset.id, completed[0])
            if preserved:
                return {
                    **preserved["stats"],
                    "skipped": True,
                    "preserved_completed": True,
                }
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
            "audio": 0,
            "visual_vectors": 0,
            "video_embeddings": 0,
            "image_embeddings": 0,
            "sampled_frames": 0,
            "visual_keyframes": 0,
            "deduplicated_frames": 0,
            "embedding_vectors": 0,
            "fingerprint": fingerprint,
            "skipped": False,
        }
        self._cache_stats = {key: 0 for key in self._cache_stats}
        self.store.mark_ingestion(asset.id, fingerprint, "started", stats)
        try:
            audio_path = self._audio_path(asset) if (self.backends.asr or self.backends.audio) else None
            if self.backends.asr and audio_path:
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
            if self.backends.visual or self.backends.ocr:
                visual, ocr, vectors, ocr_vectors = self._ingest_frames(asset, segments)
                stats.update(visual=visual, ocr=ocr, visual_vectors=vectors)
                stats["embedding_vectors"] += ocr_vectors
            if self.backends.embedding:
                if self.config.embed_images:
                    image_vectors, sampled, duplicates = self._ingest_image_embeddings(asset, segments)
                    stats["image_embeddings"] = image_vectors
                    stats["sampled_frames"] = sampled
                    stats["visual_keyframes"] = image_vectors
                    stats["deduplicated_frames"] = duplicates
                    stats["embedding_vectors"] += stats["image_embeddings"]
            if self.backends.audio and audio_path:
                stats["audio"] = self._ingest_audio(asset, segments, audio_path)
            rebuild = getattr(self.vector_index, "rebuild", None)
            if callable(rebuild) and self.backends.visual:
                rebuild(modality="visual", model=self.backends.visual.model_name)
            if callable(rebuild) and self.backends.embedding:
                for modality in ("visual", "transcript", "ocr"):
                    rebuild(modality=modality, model=self.backends.embedding.model_name)
            self.store.set_metadata("index_version", self.config.schema_version)
            self.store.set_metadata("ingestion_fingerprint", fingerprint)
            if self.backends.embedding:
                self.store.set_metadata("embedding_model", self.backends.embedding.model_name)
                self.store.set_metadata("embedding_dimensions", self.config.embedding_dimensions)
            stats.update(self._cache_stats)
            self.store.mark_ingestion(asset.id, fingerprint, "complete", stats)
            return stats
        except Exception as exc:
            stats.update(self._cache_stats)
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

    def _ingest_audio(
        self,
        asset: VideoAsset,
        segments: list[Segment],
        audio_path: Path,
    ) -> int:
        assert self.backends.audio is not None
        intervals: list[tuple[int, int]] = []
        start_ms = 0
        while start_ms < asset.duration_ms:
            end_ms = min(asset.duration_ms, start_ms + self.config.audio_window_ms)
            intervals.append((start_ms, end_ms))
            if end_ms >= asset.duration_ms:
                break
            start_ms += self.config.audio_stride_ms
        analyze_many = getattr(self.backends.audio, "analyze_many", None)
        if callable(analyze_many):
            analyses = analyze_many(str(audio_path), intervals)
        else:
            analyses = [
                self.backends.audio.analyze(str(audio_path), start, end)
                for start, end in intervals
            ]

        batch_size = 500
        for offset in range(0, len(intervals), batch_size):
            batch_intervals = intervals[offset : offset + batch_size]
            batch_analyses = analyses[offset : offset + batch_size]
            evidence: list[Evidence] = []
            for ordinal, ((start_ms, end_ms), result) in enumerate(
                zip(batch_intervals, batch_analyses), start=offset
            ):
                source = self.backends.audio.model_name
                metadata: dict[str, Any] = {
                    "speech_activity": result.speech_activity,
                    "event_scores": result.event_scores,
                    "source_reference": str(audio_path),
                    "source_hash": asset.sha256,
                    "processing_version": self.config.schema_version,
                }
                evidence.append(
                    Evidence(
                        id=f"{asset.id}-audio-{ordinal:07d}",
                        video_id=asset.id,
                        segment_id=self._segment_for(segments, start_ms),
                        start_ms=start_ms,
                        end_ms=end_ms,
                        modality="audio_event",
                        content=result.content or f"audio segment from {start_ms / 1000:.3f}s to {end_ms / 1000:.3f}s",
                        confidence=max(result.confidence, result.speech_activity * 0.5),
                        source=source,
                        metadata=metadata,
                    )
                )
            self.store.add_evidence_many(evidence)
        return len(intervals)
