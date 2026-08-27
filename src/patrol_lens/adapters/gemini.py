"""Backward-compatible model boundary.

The model is still Gemini, but requests now go through OpenRouter's
OpenAI-compatible Chat Completions API. New code may import
``OpenRouterJSONClient`` directly.
"""

from .openrouter import OpenRouterJSONClient

GeminiJSONClient = OpenRouterJSONClient

__all__ = ["GeminiJSONClient", "OpenRouterJSONClient"]
