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


def main(n_reps: int = N_REPS) -> None:
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
    import numpy as np
    import polars as pl

    from sim.estimate import hierarchical, residual_runs, score
    from sim.generate import N_CATCHERS, SCENARIOS, _real_shadow_pool, simulate

    pool = _real_shadow_pool()
    rows, t_start = [], time.time()

    for scenario in SCENARIOS:
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
                done = SCENARIOS.index(scenario) * n_reps + rep + 1
                total = len(SCENARIOS) * n_reps
                print(f"[{el/60:5.1f}分] {scenario:16s} rep {rep:>3}/{n_reps}  "
                      f"整體 {done}/{total}  預估剩 {el/done*(total-done)/60:.0f} 分", flush=True)

    res = pl.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"sim_{n_reps}reps.parquet"
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
                cov_mcse=pl.col("coverage").std() / np.sqrt(pl.col("coverage").len()),
                rank=pl.col("rank_spearman").mean(),
                ci_width=pl.col("ci_width").mean(),
                div=pl.col("divergences").sum()))

    from sim.generate import SCENARIOS
    order = {s: i for i, s in enumerate(SCENARIOS)}
    agg = agg.with_columns(o=pl.col("scenario").replace_strict(order, return_dtype=pl.Int32)).sort("o", "estimator")

    print(f"\n{'情境':<17}{'估計式':<16}{'bias':>9}{'RMSE':>8}{'coverage':>10}{'rank':>7}{'CI寬':>8}")
    print("-" * 75)
    for r in agg.iter_rows(named=True):
        print(f"{r['scenario']:<17}{r['estimator']:<16}"
              f"{r['bias']:>+9.4f}{r['rmse']:>8.4f}"
              f"{r['coverage']:>9.1%}{'':1}{r['rank']:>7.3f}{r['ci_width']:>8.4f}")
    print(f"\ncoverage 的蒙地卡羅誤差約 ±{agg['cov_mcse'].max():.1%}；"
          f"divergence 總計 {agg['div'].sum()}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_REPS)
