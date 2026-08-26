from __future__ import annotations

import re

from .domain import QueryPlan


VISUAL_HINTS = {
    "red shirt",
    "red jacket",
    "person",
    "vehicle",
    "car",
    "truck",
    "handcuff",
    "pulled over",
    "night",
    "dark",
}
AUDIO_PATTERNS = [
    (r"\brais(?:e|es|ed|ing)\s+(?:their\s+)?voice\b", "elevated vocal intensity"),
    (r"\bshout(?:ing|s|ed)?\b", "elevated vocal intensity"),
    (r"\byell(?:ing|s|ed)?\b", "elevated vocal intensity"),
    (r"\bspeak(?:ing)?\s+loud(?:ly)?\b", "elevated vocal intensity"),
    (r"\bsiren(?:s)?\b", "siren"),
    (r"\bgunshot(?:s)?\b", "gunshot"),
]
OCR_HINTS = {"license plate", "plate", "what it says", "read", "sign", "text visible"}
TEMPORAL_HINTS = {"started", "before", "after", "while", "during", "moment", "instance"}


def plan_query(query: str) -> QueryPlan:
    lowered = query.lower()
    visual = sorted({hint for hint in VISUAL_HINTS if hint in lowered})
    audio_intent = next((intent for pattern, intent in AUDIO_PATTERNS if re.search(pattern, lowered)), None)
    ocr_terms = [hint for hint in OCR_HINTS if hint in lowered]
    temporal = [hint for hint in TEMPORAL_HINTS if re.search(rf"\b{re.escape(hint)}\b", lowered)]

    weights = {"text": 1.0, "visual": 1.0, "clip": 1.0, "audio": 0.5, "ocr": 0.5}
    if audio_intent:
        weights["audio"] = 2.0
    if ocr_terms:
        weights["ocr"] = 2.0
        weights["text"] = 1.5
    if visual:
        weights["visual"] = 1.5
    if any(term in lowered for term in ("miranda", "right to remain silent", "read rights")):
        weights["text"] = 2.5

    conjunctions = []
    if re.search(r"\b(and|while|during|when|near|with|within)\b", lowered):
        conjunctions.append("multi_modal_relation")

    return QueryPlan(
        original_text=query,
        modality_weights=weights,
        text_terms=[query],
        visual_concepts=visual or [query],
        ocr_terms=ocr_terms,
        audio_intent=audio_intent,
        temporal_constraints=temporal,
        conjunctions=conjunctions,
    )
