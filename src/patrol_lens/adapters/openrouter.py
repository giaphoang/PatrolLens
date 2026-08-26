from __future__ import annotations

import base64
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..adapters.media import extract_clip
from ..domain import Candidate, EmbeddingRecord, Observation, RerankDecision, Segment, VideoAsset


RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "event_start_offset_ms": {"type": "integer", "minimum": 0},
        "event_end_offset_ms": {"type": "integer", "minimum": 0},
        "evidence": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"modality": {"type": "string"}, "offset_ms": {"type": "integer"}, "claim": {"type": "string"}}, "required": ["modality", "offset_ms", "claim"]}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["match", "event_start_offset_ms", "event_end_offset_ms", "evidence", "confidence"],
    "additionalProperties": False,
}


def _content_text(candidate: Candidate) -> str:
    lines = [f"Candidate interval: {candidate.segment.start_ms}–{candidate.segment.end_ms} ms"]
    for evidence in candidate.evidence:
        value = evidence.text or evidence.label or ""
        if value:
            lines.append(f"[{evidence.modality} {evidence.start_ms}–{evidence.end_ms} ms] {value}")
    return "\n".join(lines)


def parse_rerank_payload(payload: dict[str, Any], duration_ms: int) -> RerankDecision:
    try:
        message = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenRouter response did not contain a chat message") from exc
    if isinstance(message, list):
        message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
    if isinstance(message, str):
        message = message.strip()
        if message.startswith("```"):
            message = message.strip("`").removeprefix("json").strip()
        data = json.loads(message)
    elif isinstance(message, dict):
        data = message
    else:
        raise ValueError("OpenRouter response content was not JSON")
    match = data.get("match")
    if match not in {"yes", "no", "uncertain"}:
        raise ValueError("Invalid reranker match value")
    start = max(0, min(duration_ms, int(data.get("event_start_offset_ms", 0))))
    end = max(0, min(duration_ms, int(data.get("event_end_offset_ms", duration_ms))))
    warning = None
    if end < start:
        start, end = end, start
        warning = "reranker_interval_reordered"
    evidence = data.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    return RerankDecision(match, start, end, evidence, float(data.get("confidence", 0.0)), warning)


class OpenRouterReranker:
    def __init__(self, model: str, *, api_key: str | None = None, base_url: str = "https://openrouter.ai/api/v1/chat/completions", timeout_s: int = 90, send_video: bool = True, text_encoder: Any | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.send_video = send_video
        self.text_encoder = text_encoder
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for hosted reranking")
        if not self.model:
            raise RuntimeError("OPENROUTER_VLM_MODEL or --openrouter-model is required for hosted reranking")

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "X-Title": "PatrolLens"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter connection failed: {exc.reason}") from exc

    @staticmethod
    def _video_part(path: str) -> dict[str, Any]:
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}}

    def rerank(self, query: str, candidate: Candidate, asset_path: str | None = None) -> RerankDecision:
        parts: list[dict[str, Any]] = [{"type": "text", "text": f"Query: {query}\n{_content_text(candidate)}\nJudge only the supplied evidence. Return the exact JSON schema."}]
        temporary: str | None = None
        try:
            if self.send_video and asset_path:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
                    temporary = handle.name
                extract_clip(asset_path, candidate.segment.start_ms, candidate.segment.end_ms, temporary, fps=2)
                parts.append(self._video_part(temporary))
            body = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "You are a precise video-search verifier. Do not infer facts not visible or present in the supplied evidence."},
                    {"role": "user", "content": parts},
                ],
                "response_format": {"type": "json_schema", "json_schema": {"name": "video_search_judgement", "strict": True, "schema": RERANK_SCHEMA}},
                "provider": {"require_parameters": True, "data_collection": "deny"},
            }
            payload = self._request(body)
            return parse_rerank_payload(payload, candidate.segment.end_ms - candidate.segment.start_ms)
        finally:
            if temporary:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def annotate_segment(self, asset: VideoAsset, segment: Segment, store: Any) -> None:
        candidate = Candidate(segment=segment, evidence=store.get_observations(segment.id))
        decision = self.rerank("Describe searchable events, objects, and visible text in this interval", candidate, asset.path)
        if decision.evidence:
            text = "; ".join(str(item.get("claim", "")) for item in decision.evidence if item.get("claim"))
            if text:
                observation = Observation(f"{segment.id}-remote-caption", segment.id, asset.id, "clip", segment.start_ms, segment.end_ms, text=text, confidence=decision.confidence, metadata={"source": self.model})
                store.add_observation(observation)
                if self.text_encoder:
                    store.add_embedding(
                        EmbeddingRecord(
                            id=f"{observation.id}-embedding",
                            segment_id=segment.id,
                            modality="clip",
                            model=self.text_encoder.model_name,
                            vector=self.text_encoder.encode_text(text),
                            metadata={"observation_id": observation.id, "source": self.model},
                        )
                    )
