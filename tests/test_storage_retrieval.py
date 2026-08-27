from __future__ import annotations

from patrol_lens.config import RetrievalConfig
from patrol_lens.domain import EmbeddingRecord, Evidence, VideoAsset
from patrol_lens.index import IndexStore, SQLiteVectorIndex
from patrol_lens.retrieval import CoarseRetriever, HeuristicQueryPlanner
from patrol_lens.text import HashEmbeddingEncoder


def make_store(tmp_path):
    store = IndexStore(tmp_path / "index")
    asset = VideoAsset("video-1", "/tmp/bodycam.mp4", "hash", 90 * 60 * 1000)
    store.upsert_asset(asset)
    return store


def test_normalized_evidence_round_trip_and_fts(tmp_path):
    store = make_store(tmp_path)
    evidence = Evidence(
        "speech-1",
        "video-1",
        12_000,
        15_000,
        "transcript",
        "You have the right to remain silent",
        0.93,
        "faster-whisper-large-v3-turbo",
    )
    store.add_evidence(evidence)

    hits = store.search_text("right remain silent", modalities=["transcript"])

    assert hits[0][0] == evidence
    assert store.evidence_between("video-1", 14_000, 16_000) == [evidence]
    store.close()


def test_multimodal_temporal_join_drops_unassociated_audio(tmp_path):
    store = make_store(tmp_path)
    encoder = HashEmbeddingEncoder()
    visual = Evidence(
        "visual-near", "video-1", 31_120, 32_120, "visual", "sampled frame", 1.0, encoder.model_name
    )
    audio_near = Evidence(
        "audio-near", "video-1", 31_700, 34_000, "audio_event",
        "elevated vocal intensity shouting", 0.85, "wave-rms-pitch-v2",
    )
    audio_far = Evidence(
        "audio-far", "video-1", 300_000, 304_000, "audio_event",
        "elevated vocal intensity shouting", 0.99, "wave-rms-pitch-v2",
    )
    store.add_evidence_many([visual, audio_near, audio_far])
    store.add_embeddings(
        [
            EmbeddingRecord(
                "visual-vector", visual.id, "visual", encoder.model_name,
                encoder.encode_text("person wearing a red jacket"),
            )
        ]
    )
    retriever = CoarseRetriever(
        store,
        planner=HeuristicQueryPlanner(),
        visual_encoder=encoder,
        vector_index=SQLiteVectorIndex(store),
        config=RetrievalConfig(top_k=10, temporal_tolerance_ms=4_000),
    )

    plan, candidates = retriever.retrieve("Find when the person in the red jacket started shouting")

    assert set(plan.required_modalities) == {"visual", "audio_event"}
    assert candidates
    assert candidates[0].start_ms < 31_700 < candidates[0].end_ms
    assert all(candidate.end_ms < 100_000 for candidate in candidates)
    store.close()


def test_ocr_discovery_uses_unknown_text_branch(tmp_path):
    store = make_store(tmp_path)
    store.add_evidence(
        Evidence("plate", "video-1", 5_000, 6_000, "ocr", "ABC 1234", 0.91, "paddleocr-en")
    )
    planner = HeuristicQueryPlanner()
    plan = planner.plan("Find all license plates and tell me what each one says")

    assert plan.ocr_queries == ["*"]
    assert set(plan.required_modalities) == {"visual", "ocr"}
    assert store.top_evidence("ocr")[0][0].content == "ABC 1234"
    store.close()
