"""Deterministic metrics with explicit denominators and empty-reference handling."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float | None:
        return self.errors / self.reference_units if self.reference_units else None

    def to_dict(self) -> dict:
        return {**asdict(self), "errors": self.errors, "error_rate": self.error_rate}


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Minimum edit alignment; ties prefer fewer substitutions, then fewer deletions.

    Linear memory in hypothesis length. Units are whitespace words or Unicode
    code points as supplied by the caller, not phonemes or grapheme clusters.
    """
    previous = [(0, 0, index) for index in range(len(hypothesis) + 1)]
    for row, expected in enumerate(reference, 1):
        current = [(0, row, 0)]
        for column, actual in enumerate(hypothesis, 1):
            s, d, i = previous[column - 1]
            diagonal = (s + (expected != actual), d, i)
            s, d, i = previous[column]
            deletion = (s, d + 1, i)
            s, d, i = current[-1]
            insertion = (s, d, i + 1)
            current.append(
                min((diagonal, deletion, insertion), key=lambda item: (sum(item), *item))
            )
        previous = current
    return EditCounts(*previous[-1], reference_units=len(reference))


def sum_counts(items: Sequence[EditCounts]) -> EditCounts:
    return EditCounts(
        sum(item.substitutions for item in items),
        sum(item.deletions for item in items),
        sum(item.insertions for item in items),
        sum(item.reference_units for item in items),
    )


def mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def latency_summary(values: Sequence[float]) -> dict:
    """Linear-interpolated percentiles, position (n-1)*p; milliseconds throughout."""
    if any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("Latencies must be finite, nonnegative milliseconds.")
    ordered = sorted(values)

    def percentile(p: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * p
        lower = int(position)
        fraction = position - lower
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    return {
        "sample_count": len(ordered),
        "mean_ms": mean_defined(ordered),
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1] if ordered else None,
        "percentile_method": "linear-(n-1)*p",
    }
