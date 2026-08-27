from __future__ import annotations

from typing import Any, Protocol


class OCRBackend(Protocol):
    model_name: str

    def detect(self, image_path: str) -> list[dict[str, Any]]: ...


class NullOCR:
    model_name = "none"

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        return []
