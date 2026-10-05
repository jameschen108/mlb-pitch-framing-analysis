"""Compare independent fits using the product of their empirical posteriors.

Matching draw indices can preserve correlation from reused random seeds. These
helpers use every cross-fit combination, so ordering and draw counts do not matter.
Within-fit comparisons must instead retain their joint posterior draws.
"""

from __future__ import annotations

import numpy as np


def _draws(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or not len(out) or not np.isfinite(out).all():
        raise ValueError("Posterior draws must be a nonempty finite vector")
    return out


def _prob_above(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    ordered = np.sort(_draws(values))
    return (len(ordered) - np.searchsorted(ordered, thresholds, side="right")) / len(ordered)


def independent_lower_probability(a: np.ndarray, b: np.ndarray) -> float:
    """P(A < B), exact under the two empirical marginal posteriors."""
    return float(_prob_above(b, _draws(a)).mean())


def independent_all_lower_probability(a: np.ndarray, others: list[np.ndarray]) -> float:
    """P(A < B_j for every j), retaining the shared A in all comparisons."""
    if not others:
        raise ValueError("At least one comparison posterior is required")
    a = _draws(a)
    conditional = np.ones(len(a))
    for b in others:
        conditional *= _prob_above(b, a)
    return float(conditional.mean())


def independent_ratio_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Median and 95% ETI of A/B over all independent positive draws.

    This allocates len(a) * len(b) ratios (128 MB for two 4,000-draw fits).
    """
    a, b = _draws(a), _draws(b)
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError("Ratio summaries require positive posterior draws")
    ratio = (a[:, None] / b[None, :]).ravel()
    lo, median, hi = np.quantile(ratio, [0.025, 0.5, 0.975], overwrite_input=True)
    return {"ratio_median": float(median), "ratio_eti_lo": float(lo), "ratio_eti_hi": float(hi)}
