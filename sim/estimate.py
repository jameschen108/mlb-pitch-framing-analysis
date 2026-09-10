"""兩個估計式，都估同一個東西，才比得下去。

估計目標是**每位捕手在他自己那批球上的平均額外好球率**：

    Δ_c = mean_i [ P(strike | 捕手 c 接) − P(strike | 聯盟平均捕手接) ]

選這個而不是 u_catcher，是因為 residual runs 根本沒有 u_catcher 這個東西——它是
殘差平均，不是 logit 尺度的係數。硬要比 u 就是比兩個不同的量。Δ 是兩者都在估的。

（PLAN.md Step 5 講的「額外好球數」也是這個量，所以模擬檢驗的正是最後要報的數字。）
"""

from __future__ import annotations

import numpy as np
import polars as pl

Z95 = 1.959963985


def residual_runs(df: pl.DataFrame, n_catchers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """未調整殘差估計式（v1 的做法）：Δ̂ = mean(實際 − 基準)。

    區間用二項式 SE——這是 v1 那條路線自然會報的區間，不控制主審與投手。
    """
    c = df["fielder_2"].to_numpy()
    r = df["is_strike"].to_numpy() - df["baseline_prob"].to_numpy()
    v = df["baseline_prob"].to_numpy() * (1 - df["baseline_prob"].to_numpy())

    est = np.zeros(n_catchers)
    se = np.zeros(n_catchers)
    for j in range(n_catchers):
        m = c == j
        est[j] = r[m].mean()
        se[j] = np.sqrt(v[m].sum()) / m.sum()
    return est, est - Z95 * se, est + Z95 * se


def hierarchical(df: pl.DataFrame, n_catchers: int, warmup=500, samples=500,
                 chains=2, seed=0, n_draws=400) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """階層模型：從後驗樣本算 Δ 的後驗，區間取 2.5/97.5 百分位。"""
    import jax
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    from models.hierarchical_v2 import encode, run_nuts

    d = encode(df)
    mcmc = run_nuts(d, num_warmup=warmup, num_samples=samples, chains=chains,
                    seed=seed, progress=False)
    jax.block_until_ready(mcmc.get_samples())
    s, sc = mcmc.get_samples(), mcmc.get_samples(group_by_chain=True)

    lb = df["logit_base"].to_numpy()
    ci = d["catcher_idx"]
    a, b = np.asarray(s["a"]), np.asarray(s["b"])
    uc, uu, up = (np.asarray(s[f"u_{g}"]) for g in ("catcher", "umpire", "pitcher"))
    total = len(a)
    take = np.linspace(0, total - 1, min(n_draws, total)).astype(int)

    # 每個後驗抽樣算一次 Δ：σ(η) − σ(η − u_c)，再依捕手平均
    order = np.argsort(ci, kind="stable")
    bounds = np.searchsorted(ci[order], np.arange(n_catchers + 1))
    draws = np.empty((len(take), n_catchers))
    for k, t in enumerate(take):
        eta = a[t] + b[t] * lb + uc[t][ci] + uu[t][d["umpire_idx"]] + up[t][d["pitcher_idx"]]
        delta = 1 / (1 + np.exp(-eta)) - 1 / (1 + np.exp(-(eta - uc[t][ci])))
        ds = delta[order]
        draws[k] = [ds[bounds[j]:bounds[j + 1]].mean() for j in range(n_catchers)]

    diag = {
        "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
        "r_hat_max": float(max(split_gelman_rubin(np.asarray(sc[f"tau_{g}"]))
                               for g in ("catcher", "umpire", "pitcher"))),
        "ess_min": int(min(effective_sample_size(np.asarray(sc[f"tau_{g}"]))
                           for g in ("catcher", "umpire", "pitcher"))),
    }
    return draws.mean(0), np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0), diag


def score(est: np.ndarray, lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    return {
        "bias": float((est - truth).mean()),
        "rmse": float(np.sqrt(((est - truth) ** 2).mean())),
        "coverage": float(((lo <= truth) & (truth <= hi)).mean()),
        "rank_spearman": float(spearmanr(est, truth).statistic),
        "ci_width": float((hi - lo).mean()),
    }
