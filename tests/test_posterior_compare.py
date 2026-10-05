"""Independent season comparisons must not inherit matching-index correlation."""

import numpy as np
import pytest

from models.posterior_compare import (independent_all_lower_probability,
                                      independent_lower_probability,
                                      independent_ratio_summary)


def test_independent_probability_ignores_matching_indices_and_ties():
    a, b = np.array([1, 2, 3]), np.array([2, 3, 4])
    assert (a < b).mean() == 1  # The previous matching-index calculation.
    assert independent_lower_probability(a, b) == pytest.approx(2 / 3)
    assert independent_lower_probability(a[::-1], b) == pytest.approx(2 / 3)
    assert independent_lower_probability(a, np.repeat(b, 2)) == pytest.approx(2 / 3)


def test_independent_ratio_preserves_uncertainty_despite_aligned_draws():
    # Independent ratios are {0.5, 1, 1, 2}; aligned ratios would all be 1.
    expected = {"ratio_median": 1, "ratio_eti_lo": 0.5375, "ratio_eti_hi": 1.925}
    assert independent_ratio_summary([1, 2], [1, 2]) == pytest.approx(expected)
    assert independent_ratio_summary([2, 1], [1, 2]) == pytest.approx(expected)


def test_joint_probability_retains_shared_comparison_season():
    # Conditional success is 1 for A=1, and 0.5*0.5 for A=3.
    assert independent_all_lower_probability([1, 3], [[2, 4], [2, 4]]) == 0.625


def test_season_reports_use_independent_combinations(monkeypatch):
    import models.abs_era as ae
    from scripts.make_results import _variance_components

    fits = {s: {"tau": {g: np.array([2, 3, 4]) for g in ("catcher", "umpire", "pitcher")}}
            for s in (*ae.BASELINE_PERIOD, ae.ANTICIPATION_SEASON)}
    fits[ae.ABS_SEASON] = {"tau": {"catcher": np.array([1, 2, 3])}}
    monkeypatch.setattr(ae, "_load", lambda zone: fits)
    comp = ae.q1_comparison()
    assert np.allclose(comp["p_tau2026_lower"].to_numpy(), 2 / 3)
    assert ae.q1_verdict(comp) == "未達事先設定的全面變化判定門檻"
    table = _variance_components({"2023_holdout": fits[2023],
                                  "2025_holdout": {"tau": {g: np.array([1, 2, 3])
                                                            for g in ("catcher", "umpire", "pitcher")}}})
    assert table.filter(table["fit"] == "2025_holdout")["p_below_2023"].to_list() == pytest.approx([2 / 3] * 3)
    # Within a fit the identical umpire/catcher draws never satisfy umpire > catcher.
    # Applying the independent helper here would incorrectly report 1/3.
    assert table.filter(table["group"] == "catcher")["p_umpire_gt_catcher"].to_list() == [0, 0]
