from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.asr import ASRBackend, words_for_interval
from .adapters.audio import AudioBackend
from .adapters.media import extract_audio, extract_frame, iter_segments, probe_video
from .adapters.ocr import OCRBackend
from .domain import EmbeddingRecord, Observation, VideoAsset
from .storage import IndexStore


@dataclass
class IngestConfig:
    window_ms: int = 16_000
    stride_ms: int = 8_000
    frame_step_ms: int = 1_000
    enable_remote_annotations: bool = False


class Indexer:
    def __init__(
        self,
        store: IndexStore,
        *,
        config: IngestConfig | None = None,
        asr: ASRBackend | None = None,
        audio: AudioBackend | None = None,
        visual: Any | None = None,
        ocr: OCRBackend | None = None,
        text_encoder: Any | None = None,
        remote_annotator: Any | None = None,
    ) -> None:
        self.store = store
        self.config = config or IngestConfig()
        self.asr = asr
        self.audio = audio
        self.visual = visual
        self.ocr = ocr
        self.text_encoder = text_encoder
        self.remote_annotator = remote_annotator

    def index_path(self, path: str | Path) -> dict[str, Any]:
        asset = probe_video(path)
        return self.index_asset(asset)

    def index_asset(self, asset: VideoAsset) -> dict[str, Any]:
        self.store.upsert_asset(asset)
        self.store.set_metadata("index_version", "0.1.0")
        self.store.set_metadata("last_video_id", asset.id)
        segments = list(iter_segments(asset, window_ms=self.config.window_ms, stride_ms=self.config.stride_ms))
        for segment in segments:
            self.store.upsert_segment(segment)

        stats: dict[str, Any] = {"video_id": asset.id, "segments": len(segments), "transcript_observations": 0, "visual_embeddings": 0, "ocr_observations": 0, "audio_observations": 0}
        audio_path: Path | None = None
        if self.asr or self.audio:
            audio_path = self.store.root / "audio" / f"{asset.id}.wav"
            if not audio_path.exists():
                extract_audio(asset.path, audio_path)

        words = self.asr.transcribe(str(audio_path)) if self.asr and audio_path else []
        frame_cache: dict[int, tuple[Path, list[float] | None, list[dict[str, Any]]]] = {}
        for segment in segments:
            if words:
                interval_words = words_for_interval(words, segment.start_ms, segment.end_ms)
                if interval_words:
                    text = " ".join(word.text for word in interval_words).strip()
                    observation = Observation(
                        id=f"{segment.id}-transcript",
                        segment_id=segment.id,
                        video_id=asset.id,
                        modality="text",
                        start_ms=min(word.start_ms for word in interval_words),
                        end_ms=max(word.end_ms for word in interval_words),
                        text=text,
                        confidence=sum(word.confidence or 0.0 for word in interval_words) / len(interval_words),
                        metadata={"source": self.asr.model_name, "word_count": len(interval_words)},
                    )
                    self.store.add_observation(observation)
                    stats["transcript_observations"] += 1
                    self._embed_text(observation, modality="text")

            if self.audio and audio_path:
                analysis = self.audio.analyze(str(audio_path), segment.start_ms, segment.end_ms)
                if analysis.label:
                    observation = Observation(
                        id=f"{segment.id}-audio",
                        segment_id=segment.id,
                        video_id=asset.id,
                        modality="audio",
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        label=analysis.label,
                        confidence=analysis.confidence,
                        metadata={"rms_db": analysis.rms_db, "speech_activity": analysis.speech_activity, "source": self.audio.model_name},
                    )
                    self.store.add_observation(observation)
                    stats["audio_observations"] += 1

            if self.visual or self.ocr:
                for timestamp_ms in range(segment.start_ms, segment.end_ms, self.config.frame_step_ms):
                    if timestamp_ms not in frame_cache:
                        frame_path = self.store.root / "frames" / asset.id / f"{timestamp_ms:010d}.jpg"
                        if not frame_path.exists():
                            extract_frame(asset.path, timestamp_ms, frame_path)
                        vector = self.visual.encode_image(str(frame_path)) if self.visual else None
                        ocr_results = self.ocr.detect(str(frame_path)) if self.ocr else []
                        frame_cache[timestamp_ms] = (frame_path, vector, ocr_results)
                    frame_path, vector, ocr_results = frame_cache[timestamp_ms]
                    if vector is not None:
                        record = EmbeddingRecord(
                            id=f"{segment.id}-frame-{timestamp_ms}",
                            segment_id=segment.id,
                            modality="visual",
                            model=self.visual.model_name,
                            vector=vector,
                            metadata={"timestamp_ms": timestamp_ms, "frame_path": str(frame_path)},
                        )
                        self.store.add_embedding(record)
                        stats["visual_embeddings"] += 1
                    for index, item in enumerate(ocr_results):
                        text = str(item.get("text", "")).strip()
                        if not text:
                            continue
                        observation = Observation(
                            id=f"{segment.id}-ocr-{timestamp_ms}-{index}",
                            segment_id=segment.id,
                            video_id=asset.id,
                            modality="ocr",
                            start_ms=timestamp_ms,
                            end_ms=min(asset.duration_ms, timestamp_ms + self.config.frame_step_ms),
                            text=text,
                            confidence=item.get("confidence"),
                            metadata={"box": item.get("box"), "frame_path": str(frame_path), "source": self.ocr.model_name if self.ocr else None},
                        )
                        self.store.add_observation(observation)
                        stats["ocr_observations"] += 1
                        self._embed_text(observation, modality="ocr")

            if self.config.enable_remote_annotations and self.remote_annotator:
                self.remote_annotator.annotate_segment(asset, segment, self.store)

        self.store.set_metadata("text_model", self.text_encoder.model_name if self.text_encoder else None)
        self.store.set_metadata("last_index_stats", stats)
        return stats

    def _embed_text(self, observation: Observation, *, modality: str) -> None:
        if not self.text_encoder or not observation.text:
            return
        vector = self.text_encoder.encode_text(observation.text)
        self.store.add_embedding(
            EmbeddingRecord(
                id=f"{observation.id}-embedding",
                segment_id=observation.segment_id,
                modality=modality,  # type: ignore[arg-type]
                model=self.text_encoder.model_name,
                vector=vector,
                metadata={"observation_id": observation.id},
            )
        )
