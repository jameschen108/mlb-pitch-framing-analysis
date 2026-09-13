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

用法
----
    uv run python -m models.holdout
"""

from __future__ import annotations

import numpy as np
import polars as pl

from data.umpires import attach_umpires
from models.baseline_gam import ARTIFACT_DIR, _matrix, load_modeling_frame
from models.hierarchical_v2 import (MIN_PITCHER_PITCHES, POOLED_PITCHER_ID,
                                    SHADOW_LO, SHADOW_HI, _logit)
from models.intervals import RUN_VALUE

SEASON = 2023
CACHE = ARTIFACT_DIR / "holdout_2023_posterior.pkl"


def holdout_frame() -> pl.DataFrame:
    """2023 的 shadow zone，基準機率由 train-only 模型預測。"""
    from models.baseline_v2 import fit_or_load

    df = load_modeling_frame(SEASON)
    gam = fit_or_load()                                  # 只看過 2021–22 的 train 分割
    p = gam.predict_proba(_matrix(df))
    df = df.with_columns(baseline_prob_oof=pl.Series(p), season=pl.lit(SEASON, pl.Int32))
    df = attach_umpires(df, SEASON).drop_nulls("umpire_id")

    p = df["baseline_prob_oof"].to_numpy()
    df = df.filter(pl.Series((p > SHADOW_LO) & (p < SHADOW_HI)))
    df = df.with_columns(logit_base=pl.Series(_logit(df["baseline_prob_oof"].to_numpy())))
    keep = (df.group_by("pitcher").agg(n=pl.len())
            .filter(pl.col("n") >= MIN_PITCHER_PITCHES)["pitcher"].to_list())
    return df.with_columns(
        pitcher_g=pl.when(pl.col("pitcher").is_in(keep))
        .then(pl.col("pitcher")).otherwise(POOLED_PITCHER_ID)
    )


def run(force: bool = False, n_draws: int = 1000) -> dict:
    import pickle
    if CACHE.exists() and not force:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    import jax
    from models.hierarchical_v2 import encode, run_nuts

    df = holdout_frame()
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

    r = df["is_strike"].to_numpy() - df["baseline_prob_oof"].to_numpy()
    out = {
        "draws": draws,
        "catcher_ids": d["catcher_levels"],
        "n_pitches": np.diff(bounds),
        "resid_delta": np.array([r[ci == j].mean() for j in range(n_c)]),
        "tau": {g: np.asarray(s[f"tau_{g}"]) for g in ("catcher", "umpire", "pitcher")},
        "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
        "n_rows": df.height,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    from models.intervals import leaderboard, separability

    post = run()
    tc, tu, tp = (post["tau"][g] for g in ("catcher", "umpire", "pitcher"))
    print(f"2023 holdout：{post['n_rows']:,} 顆 shadow zone 球，"
          f"{len(post['catcher_ids'])} 位捕手，divergence {post['divergences']}\n")

    print("=== v1 的頭條宣稱，用 2023 重新檢驗 ===")
    print(f"v1（VB、全量、in-sample、無區間）：catcher 0.192  umpire 0.233  → 宣稱主審較大")
    print(f"v2（NUTS、shadow zone、樣本外基準）：")
    for g, v in (("catcher", tc), ("umpire", tu), ("pitcher", tp)):
        print(f"    tau_{g:8s} {v.mean():.4f}   89% ETI [{np.percentile(v,5.5):.4f}, {np.percentile(v,94.5):.4f}]")
    print(f"    P(tau_umpire > tau_catcher) = {(tu > tc).mean():.3f}")

    lb = leaderboard(post)
    print(f"\n=== 2023 leaderboard（前五、後三）===")
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
    off = fetch_official_framing(SEASON).rename({"id": "catcher"})
    runs_resid = post["resid_delta"] * post["n_pitches"] * RUN_VALUE
    j = pl.DataFrame({"catcher": post["catcher_ids"], "hier": lb.sort("catcher")["runs"],
                      "resid": runs_resid}).join(off, on="catcher")
    print(f"\n=== 對照 Savant 2023（{j.height} 位共同捕手）===")
    for name, col in (("hierarchical", "hier"), ("residual_runs", "resid")):
        print(f"  {name:14s} Pearson {np.corrcoef(j[col], j['rv_tot'])[0,1]:.3f}   "
              f"Spearman {j.select(pl.corr(col,'rv_tot',method='spearman')).item():.3f}")
    print(f"\n  v1 在 2023 全量、未調整殘差上報的是 r = 0.990")
