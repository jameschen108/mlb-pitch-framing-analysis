"""把 README 與 METHODS 引用的每個數字寫成 `results/` 底下的 CSV。

在這之前，repo 裡沒有任何一個發表數字是可查證的：`.gitignore` 排掉了
`models/artifacts/`、`sim/results/` 和所有 `*.parquet`，所以讀者要驗證榜單上的
一個區間，得先跑二十分鐘的 cross-fitting 或兩小時的模擬。圖是 commit 了，但圖
不能查——看圖的人沒辦法確認第三名的區間下界是不是真的是 +3.1。

這支程式只讀快取、不重新擬合任何模型。快取缺了就跳過那張表並印出該跑哪一行指令，
不會在跑了五張之後才中斷。

用法
----
    uv run python make_results.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from models.baseline_gam import ARTIFACT_DIR
from models.intervals import RUN_VALUE, leaderboard, separability

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SIM_DIR = Path(__file__).resolve().parent / "sim" / "results"
THRESHOLDS = (0, 500, 1000)
ETI = (5.5, 94.5)          # 89% 等尾區間，與 README §4 同一個慣例

_missing: list[str] = []


def _need(path: Path, how: str):
    """回傳存在的路徑，否則記下「該跑哪一行」並回傳 None。"""
    if path.exists():
        return path
    _missing.append(f"{path.relative_to(RESULTS_DIR.parent)}  →  {how}")
    return None


def _catcher_names() -> pl.DataFrame:
    """mlbam_id → 姓名。Savant 榜單是唯一有名字的來源，三季聯集取第一次出現。"""
    from data.official import fetch_official_framing

    frames = []
    for season in (2023, 2022, 2021):
        try:
            frames.append(fetch_official_framing(season).select("id", "name"))
        except Exception:
            continue
    if not frames:
        return pl.DataFrame({"catcher": [], "name": []},
                            schema={"catcher": pl.Int64, "name": pl.Utf8})
    return (pl.concat(frames, how="vertical").unique(subset="id", keep="first")
            .rename({"id": "catcher"}))


# ---- 榜單 ----

def _leaderboard_csv(post: dict, names: pl.DataFrame, savant_season: int | None) -> pl.DataFrame:
    """leaderboard() 的輸出加上姓名，以及（單季才有的）Savant 對照欄。"""
    lb = leaderboard(post).join(names, on="catcher", how="left")

    if "resid_delta" in post:
        resid = pl.DataFrame({
            "catcher": post["catcher_ids"],
            "resid_runs": post["resid_delta"] * post["n_pitches"] * RUN_VALUE,
        })
        lb = lb.join(resid, on="catcher", how="left")

    if savant_season is not None:
        from data.official import fetch_official_framing
        off = (fetch_official_framing(savant_season)
               .rename({"id": "catcher", "rv_tot": "savant_runs",
                        "pitches": "savant_pitches"})
               .select("catcher", "savant_pitches", "savant_runs"))
        lb = lb.join(off, on="catcher", how="left")

    cols = ["catcher", "name", "shadow_pitches", "delta", "delta_lo", "delta_hi",
            "runs", "runs_lo", "runs_hi", "p_positive"]
    cols += [c for c in ("resid_runs", "savant_pitches", "savant_runs") if c in lb.columns]
    return lb.select(cols).sort("delta", descending=True)   # 與 leaderboard() 同序


# ---- 變異成分 ----

def _variance_components(fits: dict[str, dict]) -> pl.DataFrame:
    """τ 的後驗平均與 89% 區間，外加 P(τ_umpire > τ_catcher)。

    v1 的 README 拿 τ 的點值直接比大小；這張表的存在就是為了讓那個比較看得到區間。
    """
    rows = []
    for fit_name, post in fits.items():
        tau = post["tau"]
        for group in ("catcher", "umpire", "pitcher"):
            v = np.asarray(tau[group])
            rows.append({
                "fit": fit_name, "group": group,
                "tau_mean": float(v.mean()), "tau_sd": float(v.std()),
                "tau_eti_lo": float(np.percentile(v, ETI[0])),
                "tau_eti_hi": float(np.percentile(v, ETI[1])),
                "eti_mass": 0.89,
                "p_umpire_gt_catcher": float(
                    (np.asarray(tau["umpire"]) > np.asarray(tau["catcher"])).mean()
                ) if group == "catcher" else None,
            })
    return pl.DataFrame(rows)


# ---- 分辨力 ----

def _separability(fits: dict[str, dict]) -> pl.DataFrame:
    rows = []
    for fit_name, post in fits.items():
        for mp in THRESHOLDS:
            sp = separability(post, min_pitches=mp)
            rows.append({
                "fit": fit_name, "min_shadow_pitches": mp,
                "n_catchers": sp["n_catchers"],
                "intervals_excluding_zero": sp["nonzero_95"],
                "share_excluding_zero": sp["nonzero_95"] / sp["n_catchers"],
                "pairs_total": sp["pairs_total"],
                "pairs_resolved_95": sp["pairs_resolved_95"],
                "share_pairs_resolved": sp["pairs_resolved_95"] / sp["pairs_total"],
                "p_rank1_gt_rank2": sp["rank1_vs_rank2_p"],
                "p_rank1_gt_rank10": sp["rank1_vs_rank10_p"],
            })
    return pl.DataFrame(rows)


# ---- 外部檢驗 ----

def _external_checks() -> pl.DataFrame | None:
    """README §7 的表：兩個估計式各自的相關，以及「階層 − 未調整」的差。

    差一律定義成 **hierarchical − residual_runs**。README 早先那張表的三列用了兩種
    方向（Savant 兩列是反的），區間因此蓋不住自己那一列的點差。結論不變——三列都
    蓋住 0——但方向要一致才查得下去。
    """
    if _need(ARTIFACT_DIR / "season_estimates.parquet",
             "uv run python -m models.validate") is None:
        return None

    from models.validate import (SEASONS, external_check_diffs, season_estimates,
                                 vs_savant, year_over_year)

    est = season_estimates()
    yoy, sv, diffs = year_over_year(est), vs_savant(est), external_check_diffs(est)

    rows = [{
        "check": f"year_over_year_{SEASONS[0]}_{SEASONS[1]}",
        "n_catchers": yoy["n_catchers"],
        "hierarchical": yoy["hierarchical"]["pearson"],
        "residual_runs": yoy["residual_runs"]["pearson"],
    }]
    for season in SEASONS:
        rows.append({
            "check": f"vs_savant_{season}",
            "n_catchers": sv[season]["n_catchers"],
            "hierarchical": sv[season]["hierarchical"]["pearson"],
            "residual_runs": sv[season]["residual_runs"]["pearson"],
        })

    for row, d in zip(rows, diffs.values()):
        row |= {"diff_hier_minus_resid": d["diff"],
                "diff_ci_lo": d["lo"], "diff_ci_hi": d["hi"],
                "n_boot": d["n_boot"]}
    return pl.DataFrame(rows)


# ---- 模擬 ----

def _sim_table(path: Path, by: list[str]) -> pl.DataFrame:
    df = pl.read_parquet(path)
    metrics = [c for c in ("bias", "rmse", "coverage", "coverage_extreme",
                           "coverage_central", "rank_spearman", "ci_width",
                           "divergences") if c in df.columns]
    return (df.group_by(by)
            .agg([pl.col(m).mean().alias(m) for m in metrics] + [pl.len().alias("n_reps")])
            .sort(by))


# ---- 敏感度：b 自由 vs b=1 ----

def _sensitivity() -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """offset 對照的兩張表：逐 arm 的摘要，以及兩個 arm 之間榜單動了多少。"""
    path = _need(ARTIFACT_DIR / "sensitivity_slope.pkl",
                 "uv run python -m models.sensitivity")
    if path is None:
        return None, None

    from models.sensitivity import comparison, summary

    with open(path, "rb") as fh:
        res = pickle.load(fh)
    return summary(res), pl.DataFrame([comparison(res)])


def _threshold_sensitivity() -> pl.DataFrame | None:
    """三個 shadow zone 門檻的對照——METHODS §2.3 事前承諾要報的那張表。"""
    path = _need(ARTIFACT_DIR / "sensitivity_threshold.pkl",
                 "uv run python -m models.sensitivity")
    if path is None:
        return None

    from models.sensitivity import threshold_summary

    with open(path, "rb") as fh:
        res = pickle.load(fh)
    return threshold_summary(res)


# ---- v1 的擬合摘要 ----

def _v1_fits() -> pl.DataFrame | None:
    """v1 VB 擬合存下來的固定效應。

    `slope` 就是 `logit_base` 的係數：1.03、1.026——不是 1。文件曾經把這一項稱為
    offset，而 offset 的定義是係數固定為 1。這張表是那個用語錯誤的證據，早在
    v1 就存在檔案裡了。
    """
    rows = []
    for key in ("2023", "2021_2023"):
        path = ARTIFACT_DIR / f"hierarchical_meta_{key}.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        rows.append({
            "fit": f"v1_vb_{key}", "n_rows": m["n"],
            "intercept": m["intercept"], "logit_base_slope": m["slope"],
            "tau_catcher": m["tau"]["catcher"], "tau_umpire": m["tau"]["umpire"],
            "tau_pitcher": m["tau"]["pitcher"],
        })
    if not rows:
        _need(ARTIFACT_DIR / "hierarchical_meta_2023.json",
              "uv run python -m models.hierarchical")
        return None
    return pl.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    names = _catcher_names()
    written: list[tuple[str, int]] = []

    def write(name: str, df: pl.DataFrame | None):
        if df is None:
            return
        path = RESULTS_DIR / f"{name}.csv"
        df.write_csv(path, float_precision=6)
        written.append((name, df.height))

    # 後驗快取：2023 holdout 與 2021–2022 合併 train pool
    fits: dict[str, dict] = {}
    for fit_name, fname, how, season in (
        ("2023_holdout", "holdout_2023_posterior.pkl",
         "uv run python -m models.holdout", 2023),
        ("2021_2022_train", "delta_posterior_train.pkl",
         "uv run python -m models.intervals", None),
    ):
        path = _need(ARTIFACT_DIR / fname, how)
        if path is None:
            continue
        with open(path, "rb") as fh:
            post = pickle.load(fh)
        fits[fit_name] = post
        write(f"leaderboard_{fit_name}", _leaderboard_csv(post, names, season))

    if fits:
        write("variance_components", _variance_components(fits))
        write("separability", _separability(fits))

    write("external_checks", _external_checks())

    sens, sens_diff = _sensitivity()
    write("sensitivity_slope", sens)
    write("sensitivity_slope_diff", sens_diff)
    write("sensitivity_threshold", _threshold_sensitivity())
    write("v1_fit_summary", _v1_fits())

    # v1 的三季合併榜單（VB，無區間）——README §8 的 +40 runs 出自這裡
    pooled = ARTIFACT_DIR / "hierarchical_framing_2021_2023.parquet"
    if _need(pooled, "uv run python -m models.hierarchical") is not None:
        write("leaderboard_v1_pooled_2021_2023",
              pl.read_parquet(pooled).rename({"mlbam_id": "catcher"})
              .join(names, on="catcher", how="left")
              .sort("framing_runs_model", descending=True))

    for name, fname, by in (
        ("sim_coverage", "sim_all_100reps.parquet", ["scenario", "estimator"]),
        ("sim_confound_sweep", "confound_sweep_50reps.parquet", ["strength", "estimator"]),
    ):
        path = _need(SIM_DIR / fname, "uv run python -m sim.run 100")
        if path is not None:
            write(name, _sim_table(path, by))

    print(f"寫入 {RESULTS_DIR.name}/")
    for name, n in written:
        print(f"  {name}.csv  ({n} 列)")
    if _missing:
        print("\n快取缺漏，以下表格沒產生：")
        for m in _missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
