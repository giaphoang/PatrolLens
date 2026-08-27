from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ..adapters.asr import ASRBackend, WordSpan
from ..adapters.audio import AudioBackend
from ..adapters.media import (
    extract_audio,
    extract_audio_segment,
    extract_clip,
    extract_frame,
    extract_frame_sequence,
    iter_segments,
    probe_video,
)
from ..adapters.audio import AudioAnalysis
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
        if force or any(item != fingerprint for item in completed):
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
            "visual": 0,
            "ocr": 0,
            "audio": 0,
            "visual_vectors": 0,
            "video_embeddings": 0,
            "image_embeddings": 0,
            "embedding_vectors": 0,
            "fingerprint": fingerprint,
            "skipped": False,
        }
        self.store.mark_ingestion(asset.id, fingerprint, "started", stats)
        try:
            audio_path = self._audio_path(asset) if (self.backends.asr or self.backends.audio or self.backends.embedding) else None
            if self.backends.asr and audio_path:
                stats["transcript"], transcript_vectors = self._ingest_transcript(asset, segments, audio_path)
                stats["embedding_vectors"] += transcript_vectors
            if self.backends.visual or self.backends.ocr:
                visual, ocr, vectors, ocr_vectors = self._ingest_frames(asset, segments)
                stats.update(visual=visual, ocr=ocr, visual_vectors=vectors)
                stats["embedding_vectors"] += ocr_vectors
            if self.backends.embedding:
                if self.config.embed_video:
                    stats["video_embeddings"] = self._ingest_video_embeddings(asset, segments)
                    stats["embedding_vectors"] += stats["video_embeddings"]
                if self.config.embed_images:
                    stats["image_embeddings"] = self._ingest_image_embeddings(asset, segments)
                    stats["embedding_vectors"] += stats["image_embeddings"]
            if self.backends.audio and audio_path:
                stats["audio"], audio_vectors = self._ingest_audio(asset, segments, audio_path)
                stats["embedding_vectors"] += audio_vectors
            elif self.backends.embedding and audio_path:
                stats["audio"], audio_vectors = self._ingest_audio(asset, segments, audio_path)
                stats["embedding_vectors"] += audio_vectors
            rebuild = getattr(self.vector_index, "rebuild", None)
            if callable(rebuild) and self.backends.visual:
                rebuild(modality="visual", model=self.backends.visual.model_name)
            if callable(rebuild) and self.backends.embedding:
                for modality in ("visual", "transcript", "ocr", "audio_event"):
                    rebuild(modality=modality, model=self.backends.embedding.model_name)
            self.store.set_metadata("index_version", self.config.schema_version)
            self.store.set_metadata("ingestion_fingerprint", fingerprint)
            if self.backends.embedding:
                self.store.set_metadata("embedding_model", self.backends.embedding.model_name)
            self.store.mark_ingestion(asset.id, fingerprint, "complete", stats)
            return stats
        except Exception as exc:
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

    def _store_text_evidence(self, evidence: list[Evidence]) -> tuple[int, int]:
        """Persist exact text evidence and optional semantic vectors together."""

        if not evidence:
            return 0, 0
        if self.backends.embedding:
            vectors = self.backends.embedding.encode_texts([item.content for item in evidence])
            if len(vectors) != len(evidence):
                raise RuntimeError(
                    f"embedding backend returned {len(vectors)} vectors for {len(evidence)} text records"
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
                        "api_model": getattr(self.backends.embedding, "batch_model", self.backends.embedding.model_name),
                    },
                )
                for item, vector in zip(evidence, vectors)
            ]
            add_pair = getattr(self.store, "add_evidence_and_embeddings", None)
            if callable(add_pair):
                add_pair(evidence, records)
            else:
                self.store.add_evidence_many(evidence)
                self.store.add_embeddings(records)
            return len(evidence), len(records)
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
                    metadata={"word_count": len(chunk)},
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

    def _ingest_frames(
        self,
        asset: VideoAsset,
        segments: list[Segment],
    ) -> tuple[int, int, int, int]:
        frames = self._ensure_frame_sequence(asset)
        visual_count = ocr_count = vector_count = 0
        ocr_vectors = 0
        if self.backends.visual:
            for offset in range(0, len(frames), self.config.batch_size):
                batch = frames[offset : offset + self.config.batch_size]
                vectors = self._encode_batch([item[1] for item in batch])
                evidence: list[Evidence] = []
                embeddings: list[EmbeddingRecord] = []
                for (timestamp_ms, frame_path), vector in zip(batch, vectors):
                    evidence_id = f"{asset.id}-visual-{timestamp_ms:010d}"
                    item = Evidence(
                        id=evidence_id,
                        video_id=asset.id,
                        segment_id=self._segment_for(segments, timestamp_ms),
                        start_ms=timestamp_ms,
                        end_ms=min(asset.duration_ms, timestamp_ms + self.config.frame_step_ms),
                        modality="visual",
                        content=f"sampled body-camera frame at {timestamp_ms / 1000:.3f}s",
                        confidence=1.0,
                        source=self.backends.visual.model_name,
                        metadata={"frame_path": str(frame_path)},
                    )
                    evidence.append(item)
                    embeddings.append(
                        EmbeddingRecord(
                            id=f"{evidence_id}-embedding",
                            evidence_id=evidence_id,
                            modality="visual",
                            model=self.backends.visual.model_name,
                            vector=vector,
                            metadata={"frame_path": str(frame_path)},
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
                for ordinal, result in enumerate(self.backends.ocr.detect(str(frame_path))):
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
                            metadata={"frame_path": str(frame_path), "box": _json_safe(result.get("box"))},
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
        if len(evidence) != len(paths):
            raise ValueError("media evidence and paths must have the same length")
        vectors = self.backends.embedding.encode_media_many(paths)
        if len(vectors) != len(evidence):
            raise RuntimeError(
                f"embedding backend returned {len(vectors)} vectors for {len(evidence)} media records"
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
                    "api_model": getattr(self.backends.embedding, "batch_model", self.backends.embedding.model_name),
                    "media_path": str(path),
                },
            )
            for item, path, vector in zip(evidence, paths, vectors)
        ]
        add_pair = getattr(self.store, "add_evidence_and_embeddings", None)
        if callable(add_pair):
            add_pair(evidence, records)
        else:
            self.store.add_evidence_many(evidence)
            self.store.add_embeddings(records)
        return len(evidence), len(records)

    def _ingest_video_embeddings(self, asset: VideoAsset, segments: list[Segment]) -> int:
        """Embed bounded temporal video chunks in the shared visual space."""

        assert self.backends.embedding is not None
        if self.config.window_ms > 120_000:
            raise ValueError("Gemini Embedding 2 video chunks cannot exceed 120 seconds")
        clip_dir = self.store.root / "media" / "video_chunks" / asset.id
        total = 0
        for offset in range(0, len(segments), self.config.embedding_batch_size):
            batch = segments[offset : offset + self.config.embedding_batch_size]
            evidence: list[Evidence] = []
            paths: list[Path] = []
            for segment in batch:
                clip_path = clip_dir / f"{segment.id}.mp4"
                if not clip_path.exists():
                    extract_clip(asset.path, segment.start_ms, segment.end_ms, clip_path, max_width=960)
                evidence.append(
                    Evidence(
                        id=f"{segment.id}-visual-video",
                        video_id=asset.id,
                        segment_id=segment.id,
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        modality="visual",
                        content=(
                            f"body-camera video chunk from {segment.start_ms / 1000:.3f}s "
                            f"to {segment.end_ms / 1000:.3f}s"
                        ),
                        confidence=1.0,
                        source=self.backends.embedding.model_name,
                        metadata={"media_path": str(clip_path), "media_kind": "video"},
                    )
                )
                paths.append(clip_path)
            _count, vectors = self._store_media_evidence(evidence, paths, media_kind="video")
            total += vectors
        return total

    def _ingest_image_embeddings(self, asset: VideoAsset, segments: list[Segment]) -> int:
        """Embed one representative image per temporal chunk for image search."""

        assert self.backends.embedding is not None
        frame_records: list[tuple[int, Path]]
        if self.backends.ocr or self.backends.visual:
            frame_records = self._ensure_frame_sequence(asset)
        else:
            frame_records = []
        keyframe_dir = self.store.root / "media" / "keyframes" / asset.id
        selected: list[tuple[Segment, Path]] = []
        for ordinal, segment in enumerate(segments):
            midpoint = (segment.start_ms + segment.end_ms) // 2
            if frame_records:
                _timestamp, frame_path = min(frame_records, key=lambda item: abs(item[0] - midpoint))
            else:
                frame_path = keyframe_dir / f"frame-{ordinal:06d}-{midpoint}.jpg"
                if not frame_path.exists():
                    extract_frame(asset.path, midpoint, frame_path)
            selected.append((segment, frame_path))

        total = 0
        for offset in range(0, len(selected), self.config.embedding_batch_size):
            batch = selected[offset : offset + self.config.embedding_batch_size]
            evidence = [
                Evidence(
                    id=f"{segment.id}-visual-image",
                    video_id=asset.id,
                    segment_id=segment.id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    modality="visual",
                    content=(
                        f"representative body-camera image for {segment.start_ms / 1000:.3f}s "
                        f"to {segment.end_ms / 1000:.3f}s"
                    ),
                    confidence=1.0,
                    source=self.backends.embedding.model_name,
                    metadata={"media_path": str(frame_path), "media_kind": "image"},
                )
                for segment, frame_path in batch
            ]
            paths = [frame_path for _segment, frame_path in batch]
            _count, vectors = self._store_media_evidence(evidence, paths, media_kind="image")
            total += vectors
        return total

    def _ingest_audio(
        self,
        asset: VideoAsset,
        segments: list[Segment],
        audio_path: Path,
    ) -> tuple[int, int]:
        intervals: list[tuple[int, int]] = []
        start_ms = 0
        while start_ms < asset.duration_ms:
            end_ms = min(asset.duration_ms, start_ms + self.config.audio_window_ms)
            intervals.append((start_ms, end_ms))
            if end_ms >= asset.duration_ms:
                break
            start_ms += self.config.audio_stride_ms
        if self.backends.audio:
            analyze_many = getattr(self.backends.audio, "analyze_many", None)
            if callable(analyze_many):
                analyses = analyze_many(str(audio_path), intervals)
            else:
                analyses = [
                    self.backends.audio.analyze(str(audio_path), start, end)
                    for start, end in intervals
                ]
        else:
            analyses = [AudioAnalysis(-80.0, 0.0, None) for _interval in intervals]

        audio_dir = self.store.root / "media" / "audio_chunks" / asset.id
        vector_count = 0
        batch_size = self.config.embedding_batch_size if self.backends.embedding else 500
        for offset in range(0, len(intervals), batch_size):
            batch_intervals = intervals[offset : offset + batch_size]
            batch_analyses = analyses[offset : offset + batch_size]
            evidence: list[Evidence] = []
            paths: list[Path] = []
            for ordinal, ((start_ms, end_ms), result) in enumerate(
                zip(batch_intervals, batch_analyses), start=offset
            ):
                media_path: Path | None = None
                if self.backends.embedding:
                    if end_ms - start_ms > 180_000:
                        raise ValueError("Gemini Embedding 2 audio chunks cannot exceed 180 seconds")
                    media_path = audio_dir / f"{asset.id}-audio-{ordinal:07d}.wav"
                    if not media_path.exists():
                        extract_audio_segment(audio_path, start_ms, end_ms, media_path)
                    paths.append(media_path)
                source = self.backends.audio.model_name if self.backends.audio else "audio-window"
                metadata: dict[str, Any] = {
                    "rms_db": result.rms_db,
                    "speech_activity": result.speech_activity,
                    "pitch_hz": result.pitch_hz,
                    "event_scores": result.event_scores,
                }
                if media_path:
                    metadata.update({"media_path": str(media_path), "media_kind": "audio"})
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
            if self.backends.embedding:
                _count, vectors = self._store_media_evidence(evidence, paths, media_kind="audio")
                vector_count += vectors
            else:
                self.store.add_evidence_many(evidence)
        return len(intervals), vector_count
