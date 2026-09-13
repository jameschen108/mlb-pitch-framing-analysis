"""識別性診斷：捕手效應和誰分不開，分不開到什麼程度。

隨機效應模型可以「控制」主審和投手，但控制得動的前提是設計上有足夠的交錯。如果
某位捕手幾乎只跟固定幾位投手搭配，他的效應和那些投手的效應在資料裡就是部分不可
分離的——再好的推論引擎也救不了。

三個診斷，兩個看設計、一個看後驗：

1. **捕手 × 主審交叉表的稀疏度**：捕手接觸到多少不同主審。交錯夠密，u_catcher 和
   v_umpire 才分得開。
2. **投捕綁定的集中度**：一位捕手的球有多少比例來自他最常搭配的投手；一位投手的球
   又有多少比例由同一位捕手接。後者才是混淆的來源。
3. **後驗相關**：同一組後驗抽樣裡，u_catcher 與他常搭投手的 u_pitcher 之間的相關。
   設計上的綁定會直接表現成後驗上的負相關（此消彼長）。

模擬（見 sim/）量化了這件事的後果：與捕手相關的混淆與真效果等大時，涵蓋率從 94%
掉到 88%；兩倍時 66%。這裡的數字是在說「真實資料離那條線多遠」。

用法
----
    uv run python -m models.identify
"""

from __future__ import annotations

import numpy as np
import polars as pl

SEASON = 2023


def crosstab_sparsity(df: pl.DataFrame) -> dict:
    """捕手 × 主審的交錯程度。"""
    n_c = df["fielder_2"].n_unique()
    n_u = df["umpire_id"].n_unique()
    pairs = df.select("fielder_2", "umpire_id").unique().height
    per_catcher = (df.group_by("fielder_2")
                   .agg(umps=pl.col("umpire_id").n_unique(), n=pl.len())
                   .filter(pl.col("n") >= 300))
    return {
        "catchers": n_c, "umpires": n_u,
        "observed_pairs": pairs, "possible_pairs": n_c * n_u,
        "fill_rate": pairs / (n_c * n_u),
        "umps_per_catcher_median": float(per_catcher["umps"].median()),
        "umps_per_catcher_min": int(per_catcher["umps"].min()),
        "n_qualified": per_catcher.height,
    }


def battery_concentration(df: pl.DataFrame, min_pitches: int = 300) -> dict:
    """投捕綁定：誰把誰綁住了。"""
    cp = df.group_by("fielder_2", "pitcher").agg(n=pl.len())

    # 一位捕手的球，最多有多少比例來自單一投手
    c_tot = cp.group_by("fielder_2").agg(tot=pl.col("n").sum())
    c = (cp.join(c_tot, on="fielder_2")
         .with_columns(share=pl.col("n") / pl.col("tot"))
         .group_by("fielder_2").agg(top=pl.col("share").max(), tot=pl.col("tot").first())
         .filter(pl.col("tot") >= min_pitches))

    # 一位投手的球，有多少比例由同一位捕手接 —— 混淆的來源在這一側
    p_tot = cp.group_by("pitcher").agg(tot=pl.col("n").sum())
    p = (cp.join(p_tot, on="pitcher")
         .with_columns(share=pl.col("n") / pl.col("tot"))
         .group_by("pitcher").agg(top=pl.col("share").max(), tot=pl.col("tot").first())
         .filter(pl.col("tot") >= 100))

    return {
        "catcher_top_pitcher_share_median": float(c["top"].median()),
        "catcher_top_pitcher_share_p90": float(c["top"].quantile(0.9)),
        "pitcher_top_catcher_share_median": float(p["top"].median()),
        "pitcher_top_catcher_share_p90": float(p["top"].quantile(0.9)),
        "n_catchers": c.height, "n_pitchers": p.height,
    }


def posterior_correlation(n_draws: int = 1000) -> dict:
    """u_catcher 與其常搭投手 u_pitcher 的後驗相關。

    對每位捕手，取他球數最多的那位投手，算兩者效應在後驗抽樣之間的相關。設計上
    綁得越緊，這個相關越負（資料分不出是誰的功勞，兩者此消彼長）。
    """
    import jax

    from models.hierarchical_v2 import POOLED_PITCHER_ID, encode, run_nuts
    from models.holdout import holdout_frame

    df = holdout_frame()
    d = encode(df)
    mcmc = run_nuts(d, num_warmup=1000, num_samples=1000, chains=4, progress=False)
    jax.block_until_ready(mcmc.get_samples())
    s = mcmc.get_samples()
    uc, up = np.asarray(s["u_catcher"]), np.asarray(s["u_pitcher"])

    # 先排除池化組：它把所有 <100 球的投手併在一起，球數幾乎總是最大的一組，
    # 不排除的話每位捕手的「最常搭配投手」都會是它，診斷等於沒做（踩過）。
    pairs = (df.filter(pl.col("pitcher_g") != POOLED_PITCHER_ID)
             .group_by("fielder_2", "pitcher_g").agg(n=pl.len())
             .sort("n", descending=True).unique(subset="fielder_2", keep="first"))
    tot = df.group_by("fielder_2").agg(tot=pl.len())
    pairs = pairs.join(tot, on="fielder_2").filter(pl.col("tot") >= 300)

    cid, pid = d["catcher_levels"], d["pitcher_levels"]
    cors = []
    for row in pairs.iter_rows(named=True):
        i = int(np.searchsorted(cid, row["fielder_2"]))
        j = int(np.searchsorted(pid, row["pitcher_g"]))
        if j < len(pid) and pid[j] == row["pitcher_g"]:
            cors.append(np.corrcoef(uc[:, i], up[:, j])[0, 1])
    cors = np.array(cors)
    return {
        "n_pairs": len(cors),
        "median": float(np.median(cors)),
        "p10": float(np.percentile(cors, 10)),
        "min": float(cors.min()),
        "frac_below_-0.2": float((cors < -0.2).mean()),
    }


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    from models.holdout import holdout_frame

    df = holdout_frame()
    print(f"{SEASON} shadow zone：{df.height:,} 顆球\n")

    x = crosstab_sparsity(df)
    print("=== 1. 捕手 × 主審交錯 ===")
    print(f"  {x['catchers']} 位捕手 × {x['umpires']} 位主審 = {x['possible_pairs']:,} 個可能配對")
    print(f"  實際觀察到 {x['observed_pairs']:,} 個（{x['fill_rate']:.1%}）")
    print(f"  ≥300 球的 {x['n_qualified']} 位捕手，接觸主審數中位數 {x['umps_per_catcher_median']:.0f}"
          f"，最少 {x['umps_per_catcher_min']}")

    y = battery_concentration(df)
    print(f"\n=== 2. 投捕綁定 ===")
    print(f"  捕手的球來自單一投手的最大比例：中位數 {y['catcher_top_pitcher_share_median']:.1%}"
          f"，第 90 百分位 {y['catcher_top_pitcher_share_p90']:.1%}")
    print(f"  投手的球由單一捕手接的最大比例：中位數 {y['pitcher_top_catcher_share_median']:.1%}"
          f"，第 90 百分位 {y['pitcher_top_catcher_share_p90']:.1%}")

    z = posterior_correlation()
    print(f"\n=== 3. 後驗相關（捕手 vs 其最常搭配的投手，{z['n_pairs']} 對）===")
    print(f"  中位數 ρ = {z['median']:+.3f}   第 10 百分位 {z['p10']:+.3f}   最負 {z['min']:+.3f}")
    print(f"  ρ < −0.2 的比例：{z['frac_below_-0.2']:.1%}")
