from __future__ import annotations

import json

from patrol_lens.adapters.openrouter import parse_rerank_payload
from patrol_lens.domain import Candidate, EmbeddingRecord, Observation, Segment, VideoAsset
from patrol_lens.query import plan_query
from patrol_lens.retrieval import Retriever
from patrol_lens.storage import IndexStore
from patrol_lens.temporal import merge_candidates
from patrol_lens.text import HashEmbeddingEncoder


def make_store(tmp_path):
    store = IndexStore(tmp_path / "index")
    asset = VideoAsset("video-1", "/tmp/bodycam.mp4", "hash", 90 * 60 * 1000)
    store.upsert_asset(asset)
    return store, asset


def add_text(store, encoder, segment_id, start_ms, end_ms, text):
    segment = Segment(segment_id, "video-1", start_ms, end_ms)
    store.upsert_segment(segment)
    observation = Observation(f"{segment_id}-text", segment_id, "video-1", "text", start_ms, end_ms, text=text)
    store.add_observation(observation)
    store.add_embedding(EmbeddingRecord(f"{segment_id}-embedding", segment_id, "text", encoder.model_name, encoder.encode_text(text)))


def test_text_and_vector_search(tmp_path):
    store, _asset = make_store(tmp_path)
    encoder = HashEmbeddingEncoder()
    add_text(store, encoder, "s1", 0, 8000, "officer reads Miranda rights")
    add_text(store, encoder, "s2", 8000, 16000, "vehicle stopped on roadside")

    result = Retriever(store, text_encoder=encoder).search_json("Miranda rights", top_k=5)

    assert result["results"][0]["segment_id"] == "s1"
    assert result["results"][0]["start_s"] == 0.0


def test_query_plan_routes_modalities():
    plan = plan_query("Find every moment where someone raises their voice near a red shirt")

    assert plan.audio_intent == "elevated vocal intensity"
    assert "red shirt" in plan.visual_concepts
    assert plan.modality_weights["audio"] > plan.modality_weights["text"] / 2
    assert "multi_modal_relation" in plan.conjunctions


def test_temporal_merge_keeps_video_boundaries():
    first = Candidate(Segment("s1", "video-1", 0, 8000), score=0.4)
    second = Candidate(Segment("s2", "video-1", 7500, 14000), score=0.8)
    third = Candidate(Segment("s3", "video-2", 0, 8000), score=0.9)

    merged = merge_candidates([first, second, third])

    assert len(merged) == 2
    assert merged[0].segment.start_ms == 0
    assert merged[0].segment.end_ms == 14000
    assert merged[0].score == 0.8


def test_openrouter_decision_is_clamped_and_validated():
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "match": "yes",
                            "event_start_offset_ms": -100,
                            "event_end_offset_ms": 9000,
                            "evidence": [{"modality": "visual", "offset_ms": 200, "claim": "hands are behind the back"}],
                            "confidence": 0.84,
                        }
                    )
                }
            }
        ]
    }

    decision = parse_rerank_payload(payload, 8000)

    assert decision.match == "yes"
    assert decision.event_start_offset_ms == 0
    assert decision.event_end_offset_ms == 8000
    assert decision.confidence == 0.84
