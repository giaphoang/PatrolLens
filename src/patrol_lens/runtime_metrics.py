"""Small, dependency-free process runtime measurements."""

from __future__ import annotations

import sys

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None  # type: ignore[assignment]


def peak_rss_mb() -> float | None:
    """Return the process high-water resident set size in MiB, when available."""

    if resource is None:
        return None
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the other common Unix implementations
    # report KiB.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(value / divisor, 3)
