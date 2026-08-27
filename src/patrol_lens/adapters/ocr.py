from __future__ import annotations

from typing import Any, Protocol


class OCRBackend(Protocol):
    model_name: str

    def detect(self, image_path: str) -> list[dict[str, Any]]: ...


class NullOCR:
    model_name = "none"

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        return []


class PaddleOCRBackend:
    def __init__(self, language: str = "en") -> None:
        self.model_name = f"paddleocr-{language}"
        self.language = language
        self._engine = None

    def _load(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("PaddleOCR is not installed") from exc
            self._engine = PaddleOCR(lang=self.language, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        return self._engine

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        result = self._load().predict(image_path)
        observations: list[dict[str, Any]] = []
        for page in result or []:
            data = page if isinstance(page, dict) else getattr(page, "json", {})
            if callable(data):
                data = data()
            if isinstance(data, dict) and isinstance(data.get("res"), dict):
                data = data["res"]
            if not isinstance(data, dict):
                continue
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_polys", [])
            for index, text in enumerate(texts):
                observations.append({"text": str(text), "confidence": float(scores[index]) if index < len(scores) else None, "box": boxes[index] if index < len(boxes) else None})
        return observations
