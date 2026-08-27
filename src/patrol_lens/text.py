from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "at", "does", "each", "every", "find", "for", "from", "in", "instance", "instances",
    "is", "it", "moment", "moments", "of", "on", "someone", "the", "their", "to", "what", "where", "with",
}


def normalize_text(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.lower()))


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


def search_tokens(value: str) -> list[str]:
    tokens = [token for token in tokenize(value) if token not in STOPWORDS]
    return tokens or tokenize(value)


def fts_query(value: str) -> str:
    tokens = search_tokens(value)
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values) or not left_values:
        return 0.0
    numerator = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class HashEmbeddingEncoder:
    """Small dependency-free baseline encoder.

    It is intentionally lexical rather than semantic. It makes the complete
    pipeline runnable in a clean environment and can be replaced by SigLIP,
    CLAP, or another encoder without changing storage or retrieval APIs.
    """

    def __init__(self, dimensions: int = 256, model_name: str = "hash-256") -> None:
        self.dimensions = dimensions
        self.model_name = model_name

    def encode_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(item * item for item in vector))
        return [item / norm for item in vector] if norm else vector
