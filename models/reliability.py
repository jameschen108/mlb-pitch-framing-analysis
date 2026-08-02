"""週 8：framing 估計的信度分析。

兩種角度量化「framing 有多少是穩定技術、多少是噪音」：

1. split-half（同季分半）：把每位捕手該季的球隨機分兩半，各算 framing 率，
   跨捕手相關。用 Spearman–Brown 校正回全樣本信度：R = 2r / (1 + r)。
2. 年度間相關：2022 的估計預測 2023，量化技術的跨季持續性。

信度用「簡單版」每球殘差率（extra strikes per pitch × RUN_VALUE × 1000），
對半量、跨季量都可比，也不必每半/每季重跑階層模型。
"""

from __future__ import annotations

import numpy as np
import polars as pl

RUN_VALUE = 0.125


def _catcher_rate(df: pl.DataFrame, group_extra: str = "half") -> pl.DataFrame:
    """每位捕手（在指定分組內）的 framing runs per 1000 called pitches。"""
    keys = ["fielder_2"] + ([group_extra] if group_extra else [])
    return (
        df.group_by(keys)
        .agg(
            n=pl.len(),
            extra=(pl.col("is_strike") - pl.col("baseline_strike_prob")).sum(),
        )
        .with_columns(rate=pl.col("extra") * RUN_VALUE / pl.col("n") * 1000)
    )


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r)


def split_half_reliability(
    df: pl.DataFrame,
    min_pitches: int = 2000,
    n_splits: int = 20,
    seed: int = 0,
) -> dict:
    """多次隨機分半，回傳平均 half-half 相關與 Spearman–Brown 校正信度。"""
    rng = np.random.default_rng(seed)
    n = df.height

    # 合格捕手以整季判定球數認定，不隨分半改變
    qualified = (
        df.group_by("fielder_2").agg(total=pl.len()).filter(pl.col("total") >= min_pitches)
    )

    rs = []
    for _ in range(n_splits):
        d = df.with_columns(half=pl.Series(rng.integers(0, 2, size=n)))
        wide = (
            _catcher_rate(d, "half")
            .pivot(values="rate", index="fielder_2", on="half")
            .join(qualified, on="fielder_2", how="inner")
            .drop_nulls()
        )
        rs.append(np.corrcoef(wide["0"].to_numpy(), wide["1"].to_numpy())[0, 1])

    r = float(np.mean(rs))
    return {
        "half_half_r": r,
        "half_half_r_sd": float(np.std(rs)),
        "spearman_brown_R": spearman_brown(r),
        "n_catchers": qualified.height,
        "n_splits": n_splits,
        "min_pitches": min_pitches,
    }


def season_rate_leaderboard(df: pl.DataFrame, min_pitches: int = 1000) -> pl.DataFrame:
    """整季每位捕手的 framing 率（供年度間相關）。"""
    return (
        _catcher_rate(df, group_extra="")
        .filter(pl.col("n") >= min_pitches)
        .rename({"fielder_2": "mlbam_id"})
    )


def year_over_year(df_a: pl.DataFrame, df_b: pl.DataFrame, min_pitches: int = 1000) -> dict:
    """兩季 framing 率的跨季相關（只算兩季都合格的捕手）。"""
    a = season_rate_leaderboard(df_a, min_pitches).select("mlbam_id", pl.col("rate").alias("rate_a"))
    b = season_rate_leaderboard(df_b, min_pitches).select("mlbam_id", pl.col("rate").alias("rate_b"))
    m = a.join(b, on="mlbam_id", how="inner")
    x = m["rate_a"].to_numpy()
    y = m["rate_b"].to_numpy()
    return {
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "n_catchers": m.height,
        "merged": m,
    }


def pairwise_year_correlations(season_dfs: dict, min_pitches: int = 1000):
    """多季兩兩 framing 率相關矩陣。season_dfs: {year: baseline_df}。

    回傳 (years, r_matrix, n_matrix)。
    """
    years = sorted(season_dfs)
    k = len(years)
    r = np.full((k, k), np.nan)
    n = np.zeros((k, k), dtype=int)
    for i, ya in enumerate(years):
        for j, yb in enumerate(years):
            if j <= i:
                continue
            res = year_over_year(season_dfs[ya], season_dfs[yb], min_pitches)
            r[i, j] = r[j, i] = res["pearson_r"]
            n[i, j] = n[j, i] = res["n_catchers"]
    np.fill_diagonal(r, 1.0)
    return years, r, n


def pooled_leaderboard(season_dfs: dict, min_pitches: int = 3000) -> pl.DataFrame:
    """多季合併 framing runs 排行榜（樣本更大、估計更穩）。

    每季殘差相對於該季自己的基準模型，跨季加總。
    """
    stacked = pl.concat(
        [df.select("fielder_2", "is_strike", "baseline_strike_prob") for df in season_dfs.values()],
        how="vertical",
    )
    return (
        stacked.group_by("fielder_2")
        .agg(
            called_pitches=pl.len(),
            extra_strikes=(pl.col("is_strike") - pl.col("baseline_strike_prob")).sum(),
        )
        .with_columns(
            framing_runs=pl.col("extra_strikes") * RUN_VALUE,
            framing_runs_per_1000=pl.col("extra_strikes") * RUN_VALUE / pl.col("called_pitches") * 1000,
        )
        .filter(pl.col("called_pitches") >= min_pitches)
        .rename({"fielder_2": "mlbam_id"})
        .sort("framing_runs", descending=True)
    )
