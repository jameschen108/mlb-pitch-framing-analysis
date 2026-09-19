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
N_BOOT = 8000             # 配對 bootstrap 重抽次數


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


def paired_bootstrap_diff(pair_a: tuple, pair_b: tuple,
                          n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """corr(pair_a) − corr(pair_b) 的 bootstrap 區間，重抽單位是捕手。

    `pair_a` / `pair_b` 各是一組 (x, y)，兩組必須排在**同一批捕手**上。年度穩定性
    傳的是 (估計_2021, 估計_2022)，對照 Savant 傳的是 (估計, savant)；兩種比較都
    化成「同一組捕手上，兩個相關的差」。

    重抽必須**配對**：每次抽一組捕手索引，兩邊共用。分開抽會把「哪些捕手入樣」的
    變異算進去兩次，區間因此過寬——而過寬的區間在這裡剛好支持「兩個估計式沒有
    差別」，也就是往我想要的方向錯。

    重抽單位是捕手而不是場次（對比 METHODS §2.2）：被比較的量本身已經是每位捕手
    一個數字，場次層級的聚類吸收在產生那些數字的那一步裡了。
    """
    a1, a2 = (np.asarray(v) for v in pair_a)
    b1, b2 = (np.asarray(v) for v in pair_b)
    n = len(a1)
    if not all(len(v) == n for v in (a2, b1, b2)):
        raise ValueError("四個向量必須等長且排在同一批捕手上")

    r = lambda x, y: np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, size=n)
        d[k] = r(a1[i], a2[i]) - r(b1[i], b2[i])
    return {
        "diff": float(r(a1, a2) - r(b1, b2)),
        "lo": float(np.nanpercentile(d, 2.5)),
        "hi": float(np.nanpercentile(d, 97.5)),
        "n_boot": n_boot,
        "n_catchers": n,
    }


def external_check_diffs(est: pl.DataFrame, min_shadow: int = MIN_SHADOW,
                         n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """README §7 那張表的最後一欄：每一列「階層 − 未調整」的差，附 95% 區間。

    三個外部檢驗全部蓋住 0，而模擬把同樣這兩個估計式分到 93% vs 74% coverage。
    差別不在哪個估計式比較好，而在這三個檢驗有沒有能力回答那個問題——46 到 60 位
    捕手，只有大於約 0.1 的相關差看得出來。
    """
    out = {}

    a = est.filter(pl.col("season") == SEASONS[0])
    b = est.filter(pl.col("season") == SEASONS[1])
    j = a.join(b, on="catcher", suffix="_b").filter(
        (pl.col("shadow_pitches") >= min_shadow)
        & (pl.col("shadow_pitches_b") >= min_shadow)
    )
    out[f"year over year {SEASONS[0]}-{SEASONS[1]}"] = paired_bootstrap_diff(
        (j["hier_delta"].to_numpy(), j["hier_delta_b"].to_numpy()),
        (j["resid_delta"].to_numpy(), j["resid_delta_b"].to_numpy()),
        n_boot=n_boot, seed=seed,
    )

    for season in SEASONS:
        off = fetch_official_framing(season).rename({"id": "catcher"})
        k = est.filter(pl.col("season") == season).join(off, on="catcher")
        t = k["rv_tot"].to_numpy()
        out[f"vs Savant {season}"] = paired_bootstrap_diff(
            (k["hier_runs"].to_numpy(), t),
            (k["resid_runs"].to_numpy(), t),
            n_boot=n_boot, seed=seed,
        )
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

    print(f"\n=== 兩個估計式的差，配對 bootstrap {N_BOOT} 次 ===")
    print("  外部檢驗的區間全部蓋住 0；模擬則把它們分得很開（sim/）。")
    for name, d in external_check_diffs(est).items():
        print(f"  {name:28s} Δr = {d['diff']:+.3f}   95% CI [{d['lo']:+.3f}, {d['hi']:+.3f}]")
