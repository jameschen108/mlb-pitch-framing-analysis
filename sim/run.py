"""模擬主程式：五個情境 × 兩個估計式 × 四個指標。

每次重複記一列，最後才彙總——這樣 coverage 的蒙地卡羅誤差算得出來，也能事後
換指標而不用重跑。結果存 parquet。

用法
----
    uv run python -m sim.run              # 預設 100 次重複
    uv run python -m sim.run 20           # 快速檢查
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
N_REPS = 100
WARMUP, SAMPLES, CHAINS = 500, 500, 2


def main(n_reps: int = N_REPS, only: tuple[str, ...] | None = None) -> None:
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
    import numpy as np
    import polars as pl

    from sim.estimate import hierarchical, residual_runs, score
    from sim.generate import N_CATCHERS, SCENARIOS, _real_shadow_pool, simulate

    pool = _real_shadow_pool()
    rows, t_start = [], time.time()
    scenarios = only or SCENARIOS

    for scenario in scenarios:
        for rep in range(n_reps):
            df, truth = simulate(scenario, seed=rep, pool=pool)
            tru = truth.effect_delta

            e, lo, hi = residual_runs(df, N_CATCHERS)
            rows.append({"scenario": scenario, "rep": rep, "estimator": "residual_runs",
                         **score(e, lo, hi, tru), "divergences": 0, "r_hat_max": float("nan")})

            e, lo, hi, diag = hierarchical(df, N_CATCHERS, warmup=WARMUP,
                                           samples=SAMPLES, chains=CHAINS, seed=rep)
            rows.append({"scenario": scenario, "rep": rep, "estimator": "hierarchical",
                         **score(e, lo, hi, tru), **{k: diag[k] for k in ("divergences", "r_hat_max")}})

            if rep % 10 == 0:
                el = time.time() - t_start
                done = scenarios.index(scenario) * n_reps + rep + 1
                total = len(scenarios) * n_reps
                print(f"[{el/60:5.1f}分] {scenario:16s} rep {rep:>3}/{n_reps}  "
                      f"整體 {done}/{total}  預估剩 {el/done*(total-done)/60:.0f} 分", flush=True)

    res = pl.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    tag = "all" if only is None else "-".join(only)
    out = RESULTS / f"sim_{tag}_{n_reps}reps.parquet"
    res.write_parquet(out)
    print(f"\n存檔 → {out}  ({res.height} 列，{(time.time()-t_start)/60:.1f} 分)")
    summarise(res)


def summarise(res) -> None:
    import numpy as np
    import polars as pl

    agg = (res.group_by("scenario", "estimator")
           .agg(bias=pl.col("bias").mean(),
                bias_mcse=pl.col("bias").std() / np.sqrt(pl.col("bias").len()),
                rmse=pl.col("rmse").mean(),
                coverage=pl.col("coverage").mean(),
                cov_extreme=pl.col("coverage_extreme").mean(),
                cov_mcse=pl.col("coverage").std() / np.sqrt(pl.col("coverage").len()),
                rank=pl.col("rank_spearman").mean(),
                ci_width=pl.col("ci_width").mean(),
                div=pl.col("divergences").sum()))

    from sim.generate import SCENARIOS
    order = {s: i for i, s in enumerate(SCENARIOS)}
    agg = agg.with_columns(o=pl.col("scenario").replace_strict(order, return_dtype=pl.Int32)).sort("o", "estimator")

    print(f"\n{'情境':<19}{'估計式':<16}{'bias':>9}{'RMSE':>8}{'cover':>8}{'尾端':>8}{'rank':>7}")
    print("-" * 76)
    for r in agg.iter_rows(named=True):
        print(f"{r['scenario']:<19}{r['estimator']:<16}"
              f"{r['bias']:>+9.4f}{r['rmse']:>8.4f}"
              f"{r['coverage']:>8.1%}{r['cov_extreme']:>8.1%}{r['rank']:>7.3f}")
    print(f"\ncoverage 的蒙地卡羅誤差約 ±{agg['cov_mcse'].max():.1%}；"
          f"divergence 總計 {agg['div'].sum()}")


def confound_sweep(n_reps: int = 50, strengths=(0.0, 0.25, 0.5, 1.0, 2.0)) -> None:
    """混淆強度掃描：要多強的未觀測混淆，區間才會失去意義。

    回答的是 v1 的 Limitations 只用一句話帶過的東西——「捕手和投手不是隨機配對」
    到底有多要緊。單挑一個強度只會得到一個任意的答案。
    """
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
    import polars as pl

    from sim.estimate import hierarchical, residual_runs, score
    from sim.generate import N_CATCHERS, _real_shadow_pool, simulate

    pool, rows = _real_shadow_pool(), []
    for k in strengths:
        for rep in range(n_reps):
            df, truth = simulate("omitted_covariate", seed=rep, pool=pool, confound_strength=k)
            tru = truth.effect_delta
            e, lo, hi = residual_runs(df, N_CATCHERS)
            rows.append({"strength": k, "estimator": "residual_runs", **score(e, lo, hi, tru)})
            e, lo, hi, _ = hierarchical(df, N_CATCHERS, warmup=WARMUP, samples=SAMPLES,
                                        chains=CHAINS, seed=rep)
            rows.append({"strength": k, "estimator": "hierarchical", **score(e, lo, hi, tru)})
        print(f"  強度 {k} 完成", flush=True)

    res = pl.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    res.write_parquet(RESULTS / f"confound_sweep_{n_reps}reps.parquet")
    agg = (res.group_by("strength", "estimator")
           .agg(coverage=pl.col("coverage").mean(), rmse=pl.col("rmse").mean(),
                rank=pl.col("rank_spearman").mean())
           .sort("strength", "estimator"))
    print(f"\n{'混淆強度':<10}{'估計式':<16}{'coverage':>10}{'RMSE':>9}{'rank':>8}")
    print("-" * 53)
    for r in agg.iter_rows(named=True):
        print(f"{r['strength']:<10}{r['estimator']:<16}{r['coverage']:>9.1%}"
              f"{r['rmse']:>9.4f}{r['rank']:>8.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        confound_sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else:
        reps = int(sys.argv[1]) if len(sys.argv) > 1 else N_REPS
        main(reps, tuple(sys.argv[2:]) or None)
