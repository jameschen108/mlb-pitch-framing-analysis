"""v2 階層模型：shadow zone + numpyro NUTS。

和 v1（models/hierarchical.py）的模型形式相同——offset 兩階段，基準模型的 logit
當固定 offset，三個交叉隨機截距在殘差上競爭：

    logit P(strike) = a + b·logit_base + u_catcher + u_umpire + u_pitcher
    u_g = τ_g · z_g,   z_g ~ N(0,1),   τ_g ~ HalfNormal(0.5)

換掉的是兩件事：

1. **推論引擎**。v1 用 statsmodels 變分貝葉斯，跑到迭代上限沒收斂，而且 VB 的
   後驗變異數系統性偏小，所以它的 effect_sd 不能拿來畫區間。這裡用 NUTS。
2. **樣本**。只留 shadow zone（0.2 < p̂ < 0.8）。第一個理由是實質的：判定明確的球
   對捕手效應幾乎沒有資訊（Fisher information 正比於 p(1−p)），砍掉 85% 的球只
   損失 39% 的資訊。計算成本是第二個理由，而且是「讓迭代變便宜」不是「讓它變可
   能」——全量的 NUTS 一季約一小時，跑得動，但這個專案要的是反覆擬合（三個門檻
   的敏感度、VB 對照、模擬），shadow zone 兩季合併一次只要七分鐘。

**隨機效果一律用非中心參數化。** τ 小的時候中心化寫法的後驗是漏斗形，NUTS 會噴
一堆 divergence，症狀看起來像跑不動、其實是幾何問題。

p̂ 用的是 out-of-fold 預測（models/crossfit），不是 in-sample。in-sample 的殘差會
被 fit 自動拉平，等於先抹掉一部分要測的訊號。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from data.umpires import attach_umpires
from models.crossfit import load_or_compute

SHADOW_LO, SHADOW_HI = 0.2, 0.8
MIN_PITCHER_PITCHES = 100
POOLED_PITCHER_ID = -1
TAU_PRIOR_SCALE = 0.5


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shadow_frame(
    seasons=None,
    lo: float = SHADOW_LO,
    hi: float = SHADOW_HI,
    min_pitcher_pitches: int = MIN_PITCHER_PITCHES,
) -> pl.DataFrame:
    """train pool 的 shadow zone 子集，含主審、池化投手、logit_base。"""
    pool = load_or_compute()
    if seasons is not None:
        pool = pool.filter(pl.col("season").is_in(list(seasons)))

    frames = []
    for s in pool["season"].unique().sort().to_list():
        frames.append(attach_umpires(pool.filter(pl.col("season") == s), s))
    df = pl.concat(frames, how="vertical").drop_nulls("umpire_id")

    p = df["baseline_prob_oof"].to_numpy()
    df = df.filter(pl.Series((p > lo) & (p < hi)))
    df = df.with_columns(
        logit_base=pl.Series(_logit(df["baseline_prob_oof"].to_numpy()))
    )

    keep = (
        df.group_by("pitcher").agg(n=pl.len())
        .filter(pl.col("n") >= min_pitcher_pitches)["pitcher"].to_list()
    )
    return df.with_columns(
        pitcher_g=pl.when(pl.col("pitcher").is_in(keep))
        .then(pl.col("pitcher"))
        .otherwise(POOLED_PITCHER_ID)
    )


def encode(df: pl.DataFrame) -> dict:
    """把三組 id 編成 0..n-1 的索引，回傳給模型用的 numpy 陣列。"""
    out = {"y": df["is_strike"].to_numpy().astype(np.float32),
           "logit_base": df["logit_base"].to_numpy().astype(np.float32)}
    for name, col in (("catcher", "fielder_2"), ("umpire", "umpire_id"), ("pitcher", "pitcher_g")):
        levels = np.sort(df[col].unique().to_numpy())
        out[f"{name}_idx"] = np.searchsorted(levels, df[col].to_numpy()).astype(np.int32)
        out[f"{name}_levels"] = levels
    return out


def model(logit_base, catcher_idx, umpire_idx, pitcher_idx,
          n_catcher, n_umpire, n_pitcher, y=None):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    a = numpyro.sample("a", dist.Normal(0.0, 2.0))
    b = numpyro.sample("b", dist.Normal(1.0, 1.0))

    eta = a + b * logit_base
    for name, idx, n in (("catcher", catcher_idx, n_catcher),
                         ("umpire", umpire_idx, n_umpire),
                         ("pitcher", pitcher_idx, n_pitcher)):
        tau = numpyro.sample(f"tau_{name}", dist.HalfNormal(TAU_PRIOR_SCALE))
        with numpyro.plate(f"n_{name}", n):
            z = numpyro.sample(f"z_{name}", dist.Normal(0.0, 1.0))   # 非中心
        u = numpyro.deterministic(f"u_{name}", tau * z)
        eta = eta + u[idx]

    numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=y)


def run_nuts(data: dict, num_warmup=500, num_samples=500, chains=2, seed=0, progress=True):
    """跑 NUTS。呼叫端負責在 **import jax 之前** 設好裝置數：

        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

    不要在這裡呼叫 numpyro.set_host_device_count()——XLA 一旦初始化就鎖住了，
    同一個 process 裡先跑 2 鏈再跑 4 鏈，後者會安靜地退化成循序執行，只在
    stderr 留一行 warning，而 wall time 會莫名其妙變兩倍。踩過一次。

    另外量時間一定要 jax.block_until_ready(mcmc.get_samples())：JAX 的陣列是
    lazy 的，run() 回來時計算還沒完成，不 block 會量到派工時間而不是計算時間。
    """
    import jax
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=chains, progress_bar=progress)
    mcmc.run(
        jax.random.PRNGKey(seed),
        logit_base=data["logit_base"],
        catcher_idx=data["catcher_idx"],
        umpire_idx=data["umpire_idx"],
        pitcher_idx=data["pitcher_idx"],
        n_catcher=len(data["catcher_levels"]),
        n_umpire=len(data["umpire_levels"]),
        n_pitcher=len(data["pitcher_levels"]),
        y=data["y"],
    )
    return mcmc
