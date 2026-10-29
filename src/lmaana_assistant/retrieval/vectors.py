"""Shared cosine-vector validation for storage and isolated evaluation."""

import math
from numbers import Real


def unit_vector(values: list[float], dimension: int) -> list[float] | None:
    """Return a finite unit vector; zero vectors have no cosine direction."""
    try:
        if dimension < 1 or len(values) != dimension:
            raise ValueError("Embedding vectors must have the declared positive dimension.")
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("Embedding vectors must contain finite numeric values.")
        length = math.hypot(*values)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Embedding vectors must contain finite numeric values.") from exc
    if not math.isfinite(length):
        raise ValueError("Embedding vector magnitude must be finite.")
    if not length:
        return None
    return [value / length for value in values]
