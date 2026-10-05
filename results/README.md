# Results tables

**English** | [繁體中文](README.zh-TW.md)

Section numbers refer to the Results sections of the [README](../README.md).

The CSVs below are written to this directory from cached model output by `uv run python -m scripts.make_results`. That script refits nothing; if a cache is missing, it names the command that rebuilds it and skips the table.

Numbers not in these files are printed by the module that produces them: the 2021 and 2022 single-season fits and the engine comparison by [`models/compare_engines.py`](../models/compare_engines.py), the log loss by [`models/baseline_v2.py`](../models/baseline_v2.py), and METHODS §3 by [`models/identify.py`](../models/identify.py). v1's numbers come from its own modules at [`v1.0`](../../../tree/v1.0).

| File | What it backs |
|---|---|
| [`leaderboard_2023_holdout.csv`](leaderboard_2023_holdout.csv) | the README's 2023 leaderboard: all 102 catchers, with Δ, framing runs, 95% intervals and Savant's figure |
| [`leaderboard_2021_2022_train.csv`](leaderboard_2021_2022_train.csv) | the same for the pooled 2021–2022 fit, 148 catchers |
| [`leaderboard_2024_holdout.csv`](leaderboard_2024_holdout.csv) | the same for 2024, 100 catchers, with Savant's 2024 figure |
| [`leaderboard_2025_holdout.csv`](leaderboard_2025_holdout.csv) | the same for 2025, 110 catchers, with Savant's 2025 figure |
| [`leaderboard_v1_pooled_2021_2023.csv`](leaderboard_v1_pooled_2021_2023.csv) | v1's three-season VB leaderboard (section 8) |
| [`variance_components.csv`](variance_components.csv) | section 4's τ table and P(τ_umpire > τ_catcher), for 2023, 2024, 2025 and the pooled 2021–2022 fit, with the probability that each 2024 and 2025 component sits below 2023's |
| [`separability.csv`](separability.csv) | section 5's three minimum-pitch cutoffs (0, 500, 1,000) for each of the four fits, with the adjacent-rank probabilities |
| [`savant_decomposition.csv`](savant_decomposition.csv) | section 3's decomposition for 2023, 2024 and 2025: each pitch set and baseline correction, with the playing-time correlations |
| [`external_checks.csv`](external_checks.csv) | the original three checks (2021 → 2022, Savant 2021 and 2022), also contained in the two files below |
| [`year_over_year.csv`](year_over_year.csv) | section 7: every pair of seasons 2021–2025, both estimators, with the paired difference |
| [`vs_savant_by_season.csv`](vs_savant_by_season.csv) | section 7: each season 2021–2025 against Savant |
| [`baseline_transport.csv`](baseline_transport.csv) | sections 2 and 7: the train-only baseline on each later season, with log loss before and after re-estimating the intercept, and predicted against actual strike rate |
| [`shadow_information.csv`](shadow_information.csv) | the shadow zone's share of pitches and of Fisher information, and the standard-error factor (METHODS §2.3) |
| [`baseline_calibration.csv`](baseline_calibration.csv) | section 2's calibration bins across the shadow zone, out of fold |
| [`isotonic_shift.csv`](isotonic_shift.csv) | section 2's isotonic recalibration: how far per-catcher runs move |
| [`abs/`](abs/) | the postscript's tables: τ by season and zone definition, the 2026 comparison, persistence, the per-season baseline check |
| [`baseline_drift.csv`](baseline_drift.csv) | section 7: zone edges by season, and the checks on the 2025 row |
| [`sim_coverage.csv`](sim_coverage.csv) | section 6's eight scenarios × two estimators |
| [`sim_confound_sweep.csv`](sim_confound_sweep.csv) | the confounder-strength sweep |
| [`sensitivity_slope.csv`](sensitivity_slope.csv) | section 4's coefficient check, with the free fit and the `b = 1` fit side by side |
| [`sensitivity_slope_diff.csv`](sensitivity_slope_diff.csv) | what fixing `b` at 1 does to the leaderboard |
| [`sensitivity_threshold.csv`](sensitivity_threshold.csv) | the three shadow-zone thresholds promised in METHODS §2.3 |
| [`v1_fit_summary.csv`](v1_fit_summary.csv) | v1's fixed effects, including the estimated `logit_base` coefficient |
