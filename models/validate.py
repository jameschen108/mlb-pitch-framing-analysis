"""Step 7：外部驗證。

兩件事，v1 都做過，這一版要補的是**把階層估計的版本也算一次，跟 residual runs
並排**——v1 兩列都只算了未調整的殘差版。

| 驗證 | 能證明 | 不能證明 |
|---|---|---|
| 年度穩定性 | 在測量某個持續的東西 | 那東西就是 framing 能力 |
| 對照 Savant | 沒算錯 | 沒有共享偏誤 |

第二列的右欄是重點，而且 v1 的 README 已經寫過這句提醒了。v2 的貢獻是模擬把它從
一句提醒變成一個數字（見 sim/：未調整估計式在有混淆時 coverage 74–83%）。

**2023 是 holdout，這裡只用 2021 和 2022。** 跨季穩定性因此只有一組年度配對，
沒辦法看衰減曲線——那是 holdout 紀律的代價，要在 METHODS 講明。

用法
----
    uv run python -m models.validate
"""

from __future__ import annotations

import numpy as np
import polars as pl

from data.official import fetch_official_framing
from models.intervals import RUN_VALUE

SEASONS = (2021, 2022)
MIN_SHADOW = 300          # 跨季比較的最低 shadow zone 球數（兩季都要達到）


def _season_posterior(season: int, n_draws: int = 600) -> pl.DataFrame:
    """單季的階層 Δ 後驗摘要 + 同一批球的 residual runs。"""
    import jax

    from models.hierarchical_v2 import encode, run_nuts, shadow_frame

    df = shadow_frame(seasons=[season])
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

    # residual runs 算在**完全同一批球**上，否則差異會混進子集效應
    r = df["is_strike"].to_numpy() - df["baseline_prob_oof"].to_numpy()
    rr = np.array([r[ci == j].mean() for j in range(n_c)])
    n = np.diff(bounds)

    return pl.DataFrame({
        "catcher": d["catcher_levels"],
        "season": season,
        "shadow_pitches": n,
        "hier_delta": draws.mean(0),
        "resid_delta": rr,
        "hier_runs": draws.mean(0) * n * RUN_VALUE,
        "resid_runs": rr * n * RUN_VALUE,
    })


def season_estimates(force: bool = False) -> pl.DataFrame:
    from models.baseline_gam import ARTIFACT_DIR
    path = ARTIFACT_DIR / "season_estimates.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    df = pl.concat([_season_posterior(s) for s in SEASONS], how="vertical")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def year_over_year(est: pl.DataFrame, min_shadow: int = MIN_SHADOW) -> dict:
    """2021 → 2022 的相關，兩個估計式各算一次。"""
    a = est.filter(pl.col("season") == SEASONS[0])
    b = est.filter(pl.col("season") == SEASONS[1])
    j = a.join(b, on="catcher", suffix="_b").filter(
        (pl.col("shadow_pitches") >= min_shadow) & (pl.col("shadow_pitches_b") >= min_shadow)
    )
    out = {"n_catchers": j.height, "min_shadow": min_shadow}
    for est_name, col in (("hierarchical", "hier_delta"), ("residual_runs", "resid_delta")):
        x, y = j[col].to_numpy(), j[f"{col}_b"].to_numpy()
        out[est_name] = {
            "pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(j.select(pl.corr(col, f"{col}_b", method="spearman")).item()),
        }
    return out


def vs_savant(est: pl.DataFrame) -> dict:
    """對照 Savant 公布的 framing run value。"""
    out = {}
    for season in SEASONS:
        off = fetch_official_framing(season).rename({"id": "catcher"})
        j = est.filter(pl.col("season") == season).join(off, on="catcher")
        row = {"n_catchers": j.height}
        for est_name, col in (("hierarchical", "hier_runs"), ("residual_runs", "resid_runs")):
            row[est_name] = {
                "pearson": float(np.corrcoef(j[col].to_numpy(), j["rv_tot"].to_numpy())[0, 1]),
                "spearman": float(j.select(pl.corr(col, "rv_tot", method="spearman")).item()),
            }
        out[season] = row
    return out


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

    est = season_estimates()
    print(f"單季估計：{est.height} 列（{SEASONS[0]}、{SEASONS[1]}）\n")

    yoy = year_over_year(est)
    print(f"=== 年度穩定性 {SEASONS[0]} → {SEASONS[1]} "
          f"（兩季都 ≥{yoy['min_shadow']} 顆，{yoy['n_catchers']} 位）===")
    for k in ("hierarchical", "residual_runs"):
        print(f"  {k:14s} Pearson {yoy[k]['pearson']:.3f}   Spearman {yoy[k]['spearman']:.3f}")

    print(f"\n=== 對照 Savant ===")
    sv = vs_savant(est)
    for season, row in sv.items():
        print(f"  {season}（{row['n_catchers']} 位共同捕手）")
        for k in ("hierarchical", "residual_runs"):
            print(f"    {k:14s} Pearson {row[k]['pearson']:.3f}   Spearman {row[k]['spearman']:.3f}")
