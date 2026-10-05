# 結果表

[English](README.md) | **繁體中文**

文中的「第幾節」指 [README](../README.zh-TW.md) 結果段的節次。

下面這些 CSV 由 `uv run python -m scripts.make_results` 從模型快取寫進這個目錄。這支程式不重新擬合任何東西；快取缺了就印出該跑哪一行指令，並跳過那張表。

不在這些檔案裡的數字，由產生它的模組印出：2021 與 2022 的單季擬合與引擎對照在 [`models/compare_engines.py`](../models/compare_engines.py)，log loss 在 [`models/baseline_v2.py`](../models/baseline_v2.py)，METHODS §3 在 [`models/identify.py`](../models/identify.py)。v1 的數字來自 [`v1.0`](../../../tree/v1.0) 的模組。

| 檔案 | 對應的數字 |
|---|---|
| [`leaderboard_2023_holdout.csv`](leaderboard_2023_holdout.csv) | README 的 2023 榜單：全部 102 位捕手的 Δ、framing runs、95% 區間與 Savant 數字 |
| [`leaderboard_2021_2022_train.csv`](leaderboard_2021_2022_train.csv) | 2021–2022 合併擬合的同一組欄位，148 位捕手 |
| [`leaderboard_2024_holdout.csv`](leaderboard_2024_holdout.csv) | 2024 的同一組欄位，100 位捕手，附 Savant 2024 的數字 |
| [`leaderboard_2025_holdout.csv`](leaderboard_2025_holdout.csv) | 2025 的同一組欄位，110 位捕手，附 Savant 2025 的數字 |
| [`leaderboard_v1_pooled_2021_2023.csv`](leaderboard_v1_pooled_2021_2023.csv) | v1 的三季合併 VB 榜單（第 8 節） |
| [`variance_components.csv`](variance_components.csv) | 第 4 節的 τ 表與 P(τ 主審 > τ 捕手)，含 2023、2024、2025 與 2021–2022 合併擬合，以及 2024、2025 各成分低於 2023 的機率 |
| [`separability.csv`](separability.csv) | 第 5 節的三個最低球數門檻（0、500、1,000），四組擬合各一份，含相鄰名次的機率 |
| [`savant_decomposition.csv`](savant_decomposition.csv) | 第 3 節的拆解，含 2023、2024、2025：每種球的範圍與基準修正，附上與上場量的相關 |
| [`external_checks.csv`](external_checks.csv) | 最早的三項檢查（2021 → 2022、Savant 2021 與 2022），也都包含在下面兩個檔案裡 |
| [`year_over_year.csv`](year_over_year.csv) | 第 7 節：2021–2025 每一對球季，兩個估計式並排加配對差 |
| [`vs_savant_by_season.csv`](vs_savant_by_season.csv) | 第 7 節：2021–2025 每一季對照 Savant |
| [`baseline_transport.csv`](baseline_transport.csv) | 第 2、7 節：train-only 基準模型在之後各季的表現，含重估截距前後的 log loss，以及預測與實際的好球率 |
| [`shadow_information.csv`](shadow_information.csv) | shadow zone 佔的球數與 Fisher information 比例，以及標準誤的放大倍數（METHODS §2.3） |
| [`baseline_calibration.csv`](baseline_calibration.csv) | 第 2 節：shadow zone 內各分箱的樣本外校準 |
| [`isotonic_shift.csv`](isotonic_shift.csv) | 第 2 節：isotonic 重新校準後，每位捕手的 runs 移動多少 |
| [`abs/`](abs/) | 後記的表：各季與兩種好球帶定義下的 τ、2026 的比較、持續性、逐季基準模型的檢查 |
| [`baseline_drift.csv`](baseline_drift.csv) | 第 7 節：各季的好球帶邊緣，以及 2025 那一列的檢查 |
| [`sim_coverage.csv`](sim_coverage.csv) | 第 6 節的八情境 × 兩個估計式 |
| [`sim_confound_sweep.csv`](sim_confound_sweep.csv) | 混淆強度掃描 |
| [`sensitivity_slope.csv`](sensitivity_slope.csv) | 第 4 節的係數檢查，自由擬合與 `b = 1` 並排 |
| [`sensitivity_slope_diff.csv`](sensitivity_slope_diff.csv) | 把 `b` 固定為 1 之後榜單動了多少 |
| [`sensitivity_threshold.csv`](sensitivity_threshold.csv) | METHODS §2.3 承諾要報的三個 shadow zone 門檻 |
| [`v1_fit_summary.csv`](v1_fit_summary.csv) | v1 的固定效應，含估計出來的 `logit_base` 係數 |
