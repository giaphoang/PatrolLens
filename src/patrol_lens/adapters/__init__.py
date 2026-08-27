"""Optional integrations for media decoding and model providers."""

from .openrouter import EmbeddingDimensionError, OpenRouterEmbeddingClient, OpenRouterJSONClient

__all__ = ["EmbeddingDimensionError", "OpenRouterEmbeddingClient", "OpenRouterJSONClient"]
