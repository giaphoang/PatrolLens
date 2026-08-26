from __future__ import annotations

from typing import Any


class SigLIP2Encoder:
    """Lazy Hugging Face adapter for frame/text embeddings.

    The rest of PatrolLens only depends on encode_text/encode_image, so this
    adapter can be replaced by a local video encoder or a hosted embedding API.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._processor, self._model
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for SigLIP2; install patrol-lens[media]") from exc
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._torch = torch
        return self._processor, self._model

    @staticmethod
    def _vector(output: Any) -> list[float]:
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            output = output.pooler_output
        elif hasattr(output, "image_embeds") and output.image_embeds is not None:
            output = output.image_embeds
        elif hasattr(output, "text_embeds") and output.text_embeds is not None:
            output = output.text_embeds
        if hasattr(output, "detach"):
            output = output.detach().float().cpu().tolist()
        while output and isinstance(output[0], list):
            output = output[0]
        norm = sum(item * item for item in output) ** 0.5
        return [item / norm for item in output] if norm else list(output)

    def encode_text(self, text: str) -> list[float]:
        processor, model = self._load()
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self._torch.no_grad():
            if hasattr(model, "get_text_features"):
                output = model.get_text_features(**inputs)
            else:
                output = model(**inputs)
        return self._vector(output)

    def encode_image(self, image_path: str) -> list[float]:
        processor, model = self._load()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image embeddings") from exc
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self._torch.no_grad():
            if hasattr(model, "get_image_features"):
                output = model.get_image_features(**inputs)
            else:
                output = model(**inputs)
        return self._vector(output)
