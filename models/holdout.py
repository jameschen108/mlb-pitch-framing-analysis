"""2023 holdout：只跑一次，在所有選模決定都定案之後。

紀律
----
2023 從專案開始就被隔離。baseline 的模型形式、shadow zone 門檻、推論引擎、Δ 的
定義、要報哪些數字——全部在 2021–2022 上決定完畢，才動這一份。

基準機率來自 `models/baseline_v2` 的模型，它只在 2021–2022 的 train 分割上擬合過，
沒看過 2023 任何一顆球。（2021–22 自己用的是 cross-fit 的 out-of-fold 預測，兩者
形式不同但都是樣本外，這一點要在 METHODS 說明。）

為什麼現在才用
--------------
v1 報的是 2023 的數字，v2 前面所有工作都在 2021–2022。不跑這一份，新舊兩張表就
沒辦法並排——球季不同，任何差異都無法歸因。holdout 保留的目的就是留到定案再用
一次，到了這一步再不用它就失去意義了。

2024–2025
---------
v2 定案之後才抓的兩季，走完全相同的路：train-only 基準、同一組門檻、同一個引擎與
設定，每季只跑一次。抓下來的時候已經沒有決定可做，所以它們比 2023 更乾淨——2023
至少在專案開始前就存在於硬碟上。train pool 不因此擴大。

用法
----
    uv run python -m models.holdout                  # 2023
    uv run python -m models.holdout --season 2024
    uv run python -m models.holdout --transport      # 基準模型在各季的 log loss
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from data.umpires import attach_umpires
from models.baseline_gam import ARTIFACT_DIR, _matrix, load_modeling_frame
from models.hierarchical_v2 import (MIN_PITCHER_PITCHES, POOLED_PITCHER_ID,
                                    SHADOW_LO, SHADOW_HI, _logit)
from models.intervals import RUN_VALUE

SEASON = 2023
NEW_SEASONS = (2024, 2025)
HOLDOUT_SEASONS = (SEASON, *NEW_SEASONS)
TRANSPORT_PATH = ARTIFACT_DIR / "baseline_transport.parquet"


def cache_path(season: int = SEASON) -> Path:
    return ARTIFACT_DIR / f"holdout_{season}_posterior.pkl"


CACHE = cache_path(SEASON)


def holdout_frame(season: int = SEASON) -> pl.DataFrame:
    """該季的 shadow zone，基準機率由 train-only 模型預測。"""
    from models.baseline_v2 import fit_or_load

    df = load_modeling_frame(season)
    gam = fit_or_load()                                  # 只看過 2021–22 的 train 分割
    p = gam.predict_proba(_matrix(df))
    df = df.with_columns(baseline_prob_oof=pl.Series(p), season=pl.lit(season, pl.Int32))
    df = attach_umpires(df, season).drop_nulls("umpire_id")

    p = df["baseline_prob_oof"].to_numpy()
    df = df.filter(pl.Series((p > SHADOW_LO) & (p < SHADOW_HI)))
    df = df.with_columns(logit_base=pl.Series(_logit(df["baseline_prob_oof"].to_numpy())))
    keep = (df.group_by("pitcher").agg(n=pl.len())
            .filter(pl.col("n") >= MIN_PITCHER_PITCHES)["pitcher"].to_list())
    return df.with_columns(
        pitcher_g=pl.when(pl.col("pitcher").is_in(keep))
        .then(pl.col("pitcher")).otherwise(POOLED_PITCHER_ID)
    )


def run(season: int = SEASON, force: bool = False, n_draws: int = 1000) -> dict:
    import pickle
    cache = cache_path(season)
    if cache.exists() and not force:
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    import jax
    from numpyro.diagnostics import summary as mcmc_summary

    from models.hierarchical_v2 import encode, run_nuts

    df = holdout_frame(season)
    d = encode(df)
    mcmc = run_nuts(d, num_warmup=1000, num_samples=1000, chains=4, progress=False)
    jax.block_until_ready(mcmc.get_samples())
    s = mcmc.get_samples()

    lb = df["logit_base"].to_numpy()
    ci = d["catcher_idx"]
    n_c = len(d["catcher_levels"])
    a, b = np.asarray(s["a"]), np.asarray(s["b"])
    uc, uu, up = (np.asarray(s[f"u_{g}"]) for g in ("catcher", "umpire", "pitcher"))
    take = np.linspace(0, len(a) - 1, min(n_draws, len(a))).astype(int)

    order = np.argsort(ci, kind="stable")
    bounds = np.searchsorted(ci[order], np.arange(n_c + 1))
    draws = np.empty((len(take), n_c))
    for k, t in enumerate(take):
        eta = a[t] + b[t] * lb + uc[t][ci] + uu[t][d["umpire_idx"]] + up[t][d["pitcher_idx"]]
        delta = 1 / (1 + np.exp(-eta)) - 1 / (1 + np.exp(-(eta - uc[t][ci])))
        ds = delta[order]
        draws[k] = [ds[bounds[j]:bounds[j + 1]].mean() for j in range(n_c)]

    # 2023 的快取早於這一段，所以舊快取裡沒有 r_hat；新季才有
    diag = mcmc_summary(mcmc.get_samples(group_by_chain=True))
    r_hat_max = float(max(np.nanmax(np.asarray(v["r_hat"])) for v in diag.values()))

    r = df["is_strike"].to_numpy() - df["baseline_prob_oof"].to_numpy()
    out = {
        "draws": draws,
        "catcher_ids": d["catcher_levels"],
        "n_pitches": np.diff(bounds),
        "resid_delta": np.array([r[ci == j].mean() for j in range(n_c)]),
        "tau": {g: np.asarray(s[f"tau_{g}"]) for g in ("catcher", "umpire", "pitcher")},
        "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
        "r_hat_max": r_hat_max,
        "n_rows": df.height,
        "season": season,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


def baseline_transport(force: bool = False) -> pl.DataFrame:
    """train-only 基準模型搬到其他球季還準不準。

    模型是在 2021–22 上擬合的。主審的好球帶若逐年漂移，會先在這裡看到：log loss 變差、
    整體好球率的預測偏掉，shadow zone 的成員也跟著變。val 那一列是參考點——同一批
    球季、模型沒看過的場次。
    """
    if TRANSPORT_PATH.exists() and not force:
        return pl.read_parquet(TRANSPORT_PATH)

    from models.baseline_v2 import evaluate, fit_or_load
    from models.splits import train_val

    gam = fit_or_load()
    frames = [("2021_2022_val", train_val()[1])]
    frames += [(str(s), load_modeling_frame(s)) for s in HOLDOUT_SEASONS]
    rows = []
    for name, df in frames:
        p = gam.predict_proba(_matrix(df))
        m = evaluate(gam, df)
        rows.append({
            "data": name, "n_called": m["n"],
            "log_loss": m["log_loss"], "brier": m["brier"],
            "log_loss_base_rate": m["log_loss_base_rate"],
            "strike_rate_actual": float(df["is_strike"].mean()),
            "strike_rate_predicted": float(p.mean()),
            "shadow_share": float(((p > SHADOW_LO) & (p < SHADOW_HI)).mean()),
        })
    out = pl.DataFrame(rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(TRANSPORT_PATH)
    return out


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="holdout 球季的後驗與基準模型檢查")
    p.add_argument("--season", type=int, default=SEASON, choices=HOLDOUT_SEASONS)
    p.add_argument("--transport", action="store_true", help="只印基準模型在各季的 log loss")
    return p.parse_args()


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    from models.intervals import leaderboard, separability

    args = _parse_args()
    if args.transport:
        pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
        print(baseline_transport())
        raise SystemExit

    season = args.season
    post = run(season)
    tc, tu, tp = (post["tau"][g] for g in ("catcher", "umpire", "pitcher"))
    r_hat = f"，max R-hat {post['r_hat_max']:.3f}" if "r_hat_max" in post else ""
    print(f"{season} holdout：{post['n_rows']:,} 顆 shadow zone 球，"
          f"{len(post['catcher_ids'])} 位捕手，divergence {post['divergences']}{r_hat}\n")

    if season == SEASON:
        print("=== v1 的頭條宣稱，用 2023 重新檢驗 ===")
        print(f"v1（VB、全量、in-sample、無區間）：catcher 0.192  umpire 0.233  → 宣稱主審較大")
        print(f"v2（NUTS、shadow zone、樣本外基準）：")
    else:
        print(f"=== {season} 的變異成分 ===")
    for g, v in (("catcher", tc), ("umpire", tu), ("pitcher", tp)):
        print(f"    tau_{g:8s} {v.mean():.4f}   95% ETI [{np.percentile(v,2.5):.4f}, {np.percentile(v,97.5):.4f}]")
    print(f"    P(tau_umpire > tau_catcher) = {(tu > tc).mean():.3f}")

    lb = leaderboard(post)
    print(f"\n=== {season} leaderboard（前五、後三）===")
    cols = ["catcher", "shadow_pitches", "delta", "delta_lo", "delta_hi", "runs", "runs_lo", "runs_hi"]
    print(pl.concat([lb.head(5), lb.tail(3)]).select(cols).to_pandas().round(4).to_string(index=False))

    print(f"\n=== 分得出來嗎 ===")
    for mp in (0, 500, 1000):
        sp = separability(post, min_pitches=mp)
        tag = "全部" if mp == 0 else f"≥{mp} 球"
        print(f"  [{tag:>8}] {sp['n_catchers']:>3} 位   區間不含 0：{sp['nonzero_95']:>3}"
              f" ({sp['nonzero_95']/sp['n_catchers']:.0%})   配對分得出勝負："
              f"{sp['pairs_resolved_95']/sp['pairs_total']:.1%}   "
              f"P(第1>第2)={sp['rank1_vs_rank2_p']:.2f}")

    from data.official import fetch_official_framing
    off = fetch_official_framing(season).rename({"id": "catcher"})
    runs_resid = post["resid_delta"] * post["n_pitches"] * RUN_VALUE
    j = pl.DataFrame({"catcher": post["catcher_ids"], "hier": lb.sort("catcher")["runs"],
                      "resid": runs_resid}).join(off, on="catcher")
    print(f"\n=== 對照 Savant {season}（{j.height} 位共同捕手）===")
    for name, col in (("hierarchical", "hier"), ("residual_runs", "resid")):
        print(f"  {name:14s} Pearson {np.corrcoef(j[col], j['rv_tot'])[0,1]:.3f}   "
              f"Spearman {j.select(pl.corr(col,'rv_tot',method='spearman')).item():.3f}")
    if season == SEASON:
        print(f"\n  v1 在 2023 全量、未調整殘差上報的是 r = 0.990")
