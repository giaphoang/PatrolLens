from __future__ import annotations

import re
from typing import Any, Protocol

from ..domain import Modality, QueryPlan


class JSONGenerator(Protocol):
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        media_paths: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]: ...


class QueryPlanner(Protocol):
    def plan(self, query: str) -> QueryPlan: ...


COLORS = "red|blue|green|yellow|orange|purple|pink|black|white|gray|grey|brown"
GARMENTS = "shirt|jacket|coat|hoodie|sweater|pants|shorts|hat"
VISUAL_TERMS = {
    "vehicle", "car", "truck", "license plate", "handcuff", "handcuffed", "handcuffing",
    "pulled over", "traffic stop", "night", "dark", "weapon", "gun", "person",
}
AUDIO_PATTERNS = {
    r"rais(?:e|es|ed|ing) (?:their |his |her )?voice": "elevated vocal intensity shouting yelling",
    r"shout(?:ing|s|ed)?": "elevated vocal intensity shouting",
    r"yell(?:ing|s|ed)?": "elevated vocal intensity yelling",
    r"scream(?:ing|s|ed)?": "screaming elevated vocal intensity",
    r"gunshot(?:s)?": "gunshot gunfire",
    r"siren(?:s)?": "siren emergency vehicle",
    r"dog bark(?:ing|s|ed)?": "dog barking",
}
TRANSCRIPT_HINTS = {
    "miranda", "right to remain silent", "rights", "says", "said", "tells", "asks",
    "reads", "speaks", "conversation", "mentions",
}
OCR_HINTS = {"license plate", "plate number", "visible text", "what each one says", "sign", "badge number"}


class HeuristicQueryPlanner:
    """Deterministic fallback and safe repair path for unavailable Gemini planning."""

    def plan(self, query: str) -> QueryPlan:
        lowered = " ".join(query.lower().split())
        visual: list[str] = []
        garments = [match.group(0) for match in re.finditer(rf"\b(?:{COLORS}) (?:{GARMENTS})\b", lowered)]
        visual.extend(garments)
        visual.extend(term for term in VISUAL_TERMS if term in lowered)

        audio = [description for pattern, description in AUDIO_PATTERNS.items() if re.search(pattern, lowered)]
        is_ocr = any(term in lowered for term in OCR_HINTS)
        is_transcript = any(term in lowered for term in TRANSCRIPT_HINTS) and not is_ocr

        if "license plate" in lowered and "license plate" not in visual:
            visual.append("license plate on a vehicle")
        if any(term in lowered for term in ("miranda", "right to remain silent", "read rights")):
            transcript = ["Miranda rights right to remain silent anything you say attorney"]
        elif is_transcript:
            transcript = [query]
        else:
            transcript = []

        ocr = ["*"] if is_ocr else []
        if not visual and not transcript and not audio and not ocr:
            visual = [query]
            transcript = [query]

        visual = list(dict.fromkeys(visual))
        audio = list(dict.fromkeys(audio))
        required: list[Modality] = []
        if visual:
            required.append("visual")
        if audio:
            required.append("audio_event")
        if transcript:
            required.append("transcript")
        if ocr:
            required.append("ocr")

        explicit_conjunction = bool(re.search(r"\b(and|while|when|during|where|with)\b", lowered))
        if (
            len(required) > 1
            and not explicit_conjunction
            and not (garments and audio)
            and not (is_ocr and visual)
        ):
            # Generic fallback branches should broaden discovery, not become an AND filter.
            required = required[:1] if len(visual) + len(audio) + len(ocr) else []

        target = "onset" if re.search(r"\b(start(?:ed|s|ing)?|began|begin(?:s|ning)?)\b", lowered) else "event"
        relation = "overlap" if len(required) > 1 or explicit_conjunction else "any"
        if " before " in f" {lowered} ":
            relation = "before"
        elif " after " in f" {lowered} ":
            relation = "after"

        weights = {"visual": 1.0, "transcript": 1.0, "ocr": 1.2, "audio_event": 1.2}
        if garments:
            weights["visual"] = 1.6
        if audio:
            weights["audio_event"] = 1.8
        if is_transcript:
            weights["transcript"] = 1.8
        if is_ocr:
            weights["ocr"] = 2.0
        return QueryPlan(
            original_text=query,
            visual_queries=visual,
            transcript_queries=transcript,
            ocr_queries=ocr,
            audio_queries=audio,
            required_modalities=list(dict.fromkeys(required)),
            modality_weights=weights,
            relation=relation,
            target=target,
            constraints={"all_instances": bool(re.search(r"\b(all|every)\b", lowered))},
        )


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visual_queries": {"type": "array", "items": {"type": "string"}},
        "transcript_queries": {"type": "array", "items": {"type": "string"}},
        "ocr_queries": {"type": "array", "items": {"type": "string"}},
        "audio_queries": {"type": "array", "items": {"type": "string"}},
        "required_modalities": {
            "type": "array",
            "items": {"type": "string", "enum": ["visual", "transcript", "ocr", "audio_event"]},
        },
        "modality_weights": {
            "type": "object",
            "properties": {
                "visual": {"type": "number"},
                "transcript": {"type": "number"},
                "ocr": {"type": "number"},
                "audio_event": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "relation": {"type": "string", "enum": ["overlap", "before", "after", "sequence", "any"]},
        "relation_tolerance_ms": {"type": "integer", "minimum": 0, "maximum": 30000},
        "target": {"type": "string"},
        "constraints": {
            "type": "object",
            "properties": {
                "all_instances": {"type": "boolean"},
                "nighttime": {"type": "boolean"},
                "speaker_attribution": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    "required": [
        "visual_queries", "transcript_queries", "ocr_queries", "audio_queries",
        "required_modalities", "modality_weights", "relation", "relation_tolerance_ms",
        "target", "constraints",
    ],
    "additionalProperties": False,
}


class GeminiQueryPlanner:
    def __init__(self, client: JSONGenerator, *, model: str | None = None) -> None:
        self.client = client
        self.model = model
        self.fallback = HeuristicQueryPlanner()

    def plan(self, query: str) -> QueryPlan:
        prompt = f"""You plan retrieval over timestamped body-camera evidence.
Return short open-vocabulary search phrases for only the needed branches.
OCR means literal visible text; ASR/transcript means spoken words; audio_event means
non-lexical sound or prosody; visual means appearance/action. Mark modalities required
only when the complete query logically requires their conjunction. Use '*' for OCR when
the user asks to discover unknown text such as every license plate. Retrieval finds
candidates; a later Gemini verifier decides whether the event truly occurred.

Investigator query: {query}"""
        try:
            data = self.client.generate_json(prompt, PLAN_SCHEMA, model=self.model)
            return self._parse(query, data)
        except Exception:  # noqa: BLE001 - planner outages deliberately fall back locally
            return self.fallback.plan(query)

    @staticmethod
    def _parse(query: str, data: dict[str, Any]) -> QueryPlan:
        allowed = {"visual", "transcript", "ocr", "audio_event"}
        weights = {
            key: min(3.0, max(0.1, float(value)))
            for key, value in dict(data.get("modality_weights", {})).items()
            if key in allowed
        }
        return QueryPlan(
            original_text=query,
            visual_queries=[str(item) for item in data.get("visual_queries", []) if str(item).strip()],
            transcript_queries=[str(item) for item in data.get("transcript_queries", []) if str(item).strip()],
            ocr_queries=[str(item) for item in data.get("ocr_queries", []) if str(item).strip()],
            audio_queries=[str(item) for item in data.get("audio_queries", []) if str(item).strip()],
            required_modalities=[item for item in data.get("required_modalities", []) if item in allowed],
            modality_weights=weights,
            relation=data.get("relation", "any"),
            relation_tolerance_ms=min(30_000, max(0, int(data.get("relation_tolerance_ms", 4_000)))),
            target=str(data.get("target", "event")),
            constraints=dict(data.get("constraints", {})),
        )
