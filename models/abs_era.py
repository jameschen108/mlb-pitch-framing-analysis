"""ABS 後記的 Q1：2026 年有了挑戰之後，主審的原判還受不受捕手影響。

事前登記在 NOTES.md（2026-09-29），這裡照它做，看到結果之後不改設計。

和 v2 不同的地方
----------------
v2 的流程、快取與數字一律不動；這裡是另一條流程，所以同一季的 τ 會和 v2 公布的不同。

- **好球帶**：2021–2026 全部用身高算的 ABS 帶標準化 `plate_z`（`data/heights.py`）。
  2026 的 Statcast 已經是這個定義，其他季要換過來才比得了。敏感度檢查另外用舊的
  姿勢定義（`zone="stance"`）跑 2021–2025。
- **基準模型**：每一季在自己內部依場次五折 cross-fit。v2 用一個固定在 2021–22 的
  模型，這一輪發現它到 2025 會把整體好球率高估 2.6 個百分點；跨季比較不能帶著這種偏差。
- **判決**：2026 用主審原判（`data/challenges.py` 還原），其他季沒有挑戰。

模型形式、shadow zone 門檻、NUTS 設定與 v2 相同。

用法
----
    uv run python -m models.abs_era fit --zone abs       # 2021–2026，每季擬合一次
    uv run python -m models.abs_era fit --zone stance    # 敏感度：2021–2025
    uv run python -m models.abs_era report
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

from models.baseline_gam import (ARTIFACT_DIR, DATA_PROCESSED, X_ABS_MAX, Z_STD_MAX,
                                 Z_STD_MIN, _matrix, fit_baseline)
from models.posterior_compare import (independent_all_lower_probability,
                                      independent_lower_probability,
                                      independent_ratio_summary)

SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
BASELINE_PERIOD = (2021, 2022, 2023, 2024)
ANTICIPATION_SEASON = 2025          # 春訓試辦過 ABS、好球帶已明顯變窄，單獨列出
ABS_SEASON = 2026
ZONES = {"abs": SEASONS, "stance": SEASONS[:-1]}   # 2026 沒有姿勢制的上下緣
K_FOLDS = 5
ABS_DIR = ARTIFACT_DIR / "abs"
MIN_SHADOW = 300
DECISION_P = 0.95


def season_frame(season: int, zone: str = "abs") -> pl.DataFrame:
    """該季的判定球，`plate_z_std` 依 zone 換成對應的標準化高度，再剔除極端座標。

    沿用 `plate_z_std` 這個欄名，是為了讓 `baseline_gam` 的特徵順序照舊；
    zone="abs" 時它的內容是 `plate_z_abs`。
    """
    from data.heights import attach_abs_zone

    if season not in ZONES[zone]:
        raise ValueError(f"{season} 沒有 {zone} 定義的好球帶")
    df = attach_abs_zone(pl.read_parquet(DATA_PROCESSED / f"statcast_{season}.parquet"))
    if zone == "abs":
        df = df.with_columns(plate_z_std=pl.col("plate_z_abs"))
    if "is_strike_final" not in df.columns:            # 2026 以前沒有挑戰
        df = df.with_columns(is_strike_final=pl.col("is_strike"))
    df = df.filter((pl.col("plate_x").abs() <= X_ABS_MAX)
                   & pl.col("plate_z_std").is_between(Z_STD_MIN, Z_STD_MAX))
    return df.with_columns(stand_i=(pl.col("stand") == "R").cast(pl.Int64),
                           p_throws_i=(pl.col("p_throws") == "R").cast(pl.Int64),
                           season=pl.lit(season, pl.Int32))


def _oof_path(season: int, zone: str) -> Path:
    return ABS_DIR / f"oof_{zone}_{season}.parquet"


def season_oof(season: int, zone: str = "abs", force: bool = False) -> pl.DataFrame:
    """該季內部依場次五折 cross-fit 的基準機率。"""
    from models.crossfit import fold_assignment
    from models.splits import SPLIT_SEED

    path = _oof_path(season, zone)
    if path.exists() and not force:
        return pl.read_parquet(path)
    df = season_frame(season, zone)
    games = df["game_pk"].unique().sort().to_numpy()
    df = df.with_columns(fold=df["game_pk"].replace_strict(
        fold_assignment(games, K_FOLDS, SPLIT_SEED), return_dtype=pl.Int32))
    oof = np.full(df.height, np.nan)
    fold = df["fold"].to_numpy()
    for k in range(K_FOLDS):
        gam = fit_baseline(df.filter(pl.col("fold") != k), verbose=False)
        oof[fold == k] = gam.predict_proba(_matrix(df.filter(pl.col("fold") == k)))
        print(f"  [{zone} {season}] fold {k + 1}/{K_FOLDS}", flush=True)
    out = df.select("game_pk", "fielder_2", "pitcher", "batter", "is_strike", "is_strike_final",
                    "plate_x", "plate_z_std", "season").with_columns(baseline_prob_oof=pl.Series(oof))
    ABS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return out


def shadow_frame(season: int, zone: str = "abs") -> pl.DataFrame:
    """shadow zone、主審、池化投手、logit_base——與 v2 的 shadow_frame 同一套規則。"""
    from data.umpires import attach_umpires
    from models.hierarchical_v2 import (MIN_PITCHER_PITCHES, POOLED_PITCHER_ID,
                                        SHADOW_HI, SHADOW_LO, _logit)

    df = attach_umpires(season_oof(season, zone), season).drop_nulls("umpire_id")
    p = df["baseline_prob_oof"].to_numpy()
    df = df.filter(pl.Series((p > SHADOW_LO) & (p < SHADOW_HI)))
    df = df.with_columns(logit_base=pl.Series(_logit(df["baseline_prob_oof"].to_numpy())))
    keep = (df.group_by("pitcher").agg(n=pl.len())
            .filter(pl.col("n") >= MIN_PITCHER_PITCHES)["pitcher"].to_list())
    return df.with_columns(pitcher_g=pl.when(pl.col("pitcher").is_in(keep))
                           .then(pl.col("pitcher")).otherwise(POOLED_PITCHER_ID))


def _fit_path(season: int, zone: str) -> Path:
    return ABS_DIR / f"fit_{zone}_{season}.pkl"


def fit(season: int, zone: str = "abs", force: bool = False) -> dict:
    """該季的階層模型（原判），每季只擬合一次，快取之後不重跑。"""
    path = _fit_path(season, zone)
    if path.exists() and not force:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    import jax
    from numpyro.diagnostics import summary as mcmc_summary

    from models.hierarchical_v2 import delta_draws, encode, run_nuts

    df = shadow_frame(season, zone)
    d = encode(df)
    mcmc = run_nuts(d, num_warmup=1000, num_samples=1000, chains=4, progress=False)
    jax.block_until_ready(mcmc.get_samples())
    s = mcmc.get_samples()
    draws, n = delta_draws(s, d, df["logit_base"].to_numpy())
    diag = mcmc_summary(mcmc.get_samples(group_by_chain=True))
    y, p = df["is_strike"].to_numpy(), df["baseline_prob_oof"].to_numpy()
    out = {
        "season": season, "zone": zone, "draws": draws, "catcher_ids": d["catcher_levels"],
        "n_pitches": n, "tau": {g: np.asarray(s[f"tau_{g}"]) for g in ("catcher", "umpire", "pitcher")},
        "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
        "r_hat_max": float(max(np.nanmax(np.asarray(v["r_hat"])) for v in diag.values())),
        "n_rows": df.height, "shadow_mean_resid": float((y - p).mean()),
    }
    ABS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


# ---------------------------------------------------------------- 報告

def _load(zone: str) -> dict[int, dict]:
    return {s: fit(s, zone) for s in ZONES[zone] if _fit_path(s, zone).exists()}


def tau_table(zone: str = "abs") -> pl.DataFrame:
    rows = []
    for s, f in _load(zone).items():
        for g, v in f["tau"].items():
            rows.append({"zone": zone, "season": s, "group": g, "tau_mean": float(v.mean()),
                         "tau_eti_lo": float(np.percentile(v, 2.5)),
                         "tau_eti_hi": float(np.percentile(v, 97.5))})
        rows[-3] |= {"n_rows": f["n_rows"], "n_catchers": len(f["catcher_ids"]),
                     "divergences": f["divergences"], "r_hat_max": f["r_hat_max"],
                     "shadow_mean_resid": f["shadow_mean_resid"]}
    return pl.DataFrame(rows)


def q1_comparison(zone: str = "abs") -> pl.DataFrame:
    """事前登記的主要比較：2026 對基準期每一季，外加 2025（不進判定）。

    各季按獨立後驗處理，使用邊際後驗的全部交叉組合；不按相同抽樣索引配對。
    聯合下降機率另列，事前規則仍使用四個邊際機率的門檻。
    """
    fits = _load(zone)
    t26 = fits[ABS_SEASON]["tau"]["catcher"]
    joint = independent_all_lower_probability(
        t26, [fits[s]["tau"]["catcher"] for s in BASELINE_PERIOD])
    rows = []
    for s in (*BASELINE_PERIOD, ANTICIPATION_SEASON):
        ts = fits[s]["tau"]["catcher"]
        rows.append({"zone": zone, "vs_season": s, "in_baseline_period": s in BASELINE_PERIOD,
                     "p_tau2026_lower": independent_lower_probability(t26, ts),
                     "p_tau2026_lower_than_all_baseline": joint,
                     **independent_ratio_summary(t26, ts)})
    return pl.DataFrame(rows)


def q1_verdict(comp: pl.DataFrame) -> str:
    p = comp.filter(pl.col("in_baseline_period"))["p_tau2026_lower"]
    if (p > DECISION_P).all():
        return "達到低於全部基準季的事前門檻"
    if (p < 1 - DECISION_P).all():
        return "達到高於全部基準季的事前門檻"
    return "未達事先設定的全面變化判定門檻"


def persistence(zone: str = "abs") -> pl.DataFrame:
    """相鄰球季捕手 Δ 的相關（兩季都 ≥ MIN_SHADOW 顆 shadow zone 球）。"""
    fits = _load(zone)
    est = {s: pl.DataFrame({"catcher": f["catcher_ids"], "delta": f["draws"].mean(0), "n": f["n_pitches"]})
           for s, f in fits.items()}
    rows = []
    for a, b in zip(sorted(est)[:-1], sorted(est)[1:]):
        j = est[a].join(est[b], on="catcher", suffix="_b").filter(
            (pl.col("n") >= MIN_SHADOW) & (pl.col("n_b") >= MIN_SHADOW))
        rows.append({"zone": zone, "season_a": a, "season_b": b, "n_catchers": j.height,
                     "pearson": float(np.corrcoef(j["delta"], j["delta_b"])[0, 1]),
                     "spearman": float(j.select(pl.corr("delta", "delta_b", method="spearman")).item())})
    return pl.DataFrame(rows)


def baseline_check() -> pl.DataFrame:
    """逐季基準模型的整體校準：cross-fit 在季內，預測與實際好球率應該幾乎一樣。"""
    rows = []
    for zone, seasons in ZONES.items():
        for s in seasons:
            if not _oof_path(s, zone).exists():
                continue
            df = pl.read_parquet(_oof_path(s, zone))
            rows.append({"zone": zone, "season": s, "n_called": df.height,
                         "strike_rate_actual": float(df["is_strike"].mean()),
                         "strike_rate_predicted": float(df["baseline_prob_oof"].mean())})
    return pl.DataFrame(rows)


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="ABS 後記 Q1")
    p.add_argument("cmd", choices=["fit", "report"])
    p.add_argument("--zone", default="abs", choices=list(ZONES))
    return p.parse_args()


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

    args = _parse_args()
    if args.cmd == "fit":
        for s in ZONES[args.zone]:
            f = fit(s, args.zone)
            print(f"[{args.zone} {s}] {f['n_rows']:,} 顆 shadow zone 球、{len(f['catcher_ids'])} 位捕手、"
                  f"divergence {f['divergences']}、max R-hat {f['r_hat_max']:.3f}", flush=True)
        raise SystemExit

    pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_rows(-1)
    print(baseline_check())
    for zone in ZONES:
        if not _load(zone):
            continue
        print(f"\n=== {zone} ===")
        print(tau_table(zone).filter(pl.col("group") == "catcher"))
        print(persistence(zone))
        if ABS_SEASON in _load(zone):
            comp = q1_comparison(zone)
            print(comp)
            print(f"判定（事前規則）：{q1_verdict(comp)}")
