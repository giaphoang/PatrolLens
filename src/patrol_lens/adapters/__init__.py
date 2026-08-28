"""Optional integrations for media decoding and model providers."""

from .clap import ClapCoreMLBackend, ClapEmbeddingDimensionError
from .openrouter import EmbeddingDimensionError, OpenRouterEmbeddingClient, OpenRouterJSONClient

__all__ = [
    "ClapCoreMLBackend",
    "ClapEmbeddingDimensionError",
    "EmbeddingDimensionError",
    "OpenRouterEmbeddingClient",
    "OpenRouterJSONClient",
]
