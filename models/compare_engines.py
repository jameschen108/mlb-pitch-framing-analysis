"""VB vs NUTS，以及變異成分的季別穩定性。

要回答的是：v1 報的「主審變異大於捕手變異」，換掉推論引擎之後還在不在。

問題是 v1 和 v2 之間同時變了四件事——季別、引擎（VB vs NUTS）、子集（全量 vs
shadow zone）、基準機率（in-sample vs out-of-fold）。四個一起變就無法歸因，所以
排成三格，一次只動一個：

    A  VB   / 全量   / in-sample     ← v1 的原流程，只換季別
    B  VB   / shadow / out-of-fold
    C  NUTS / shadow / out-of-fold

A vs B 隔離子集與基準機率，B vs C 隔離引擎。

**注意 2023 是 holdout，這裡只跑 2021 和 2022。** 和 v1 的 2023 數字對照時，用的是
它 README 已經發表的值，不是重跑出來的——holdout 沒有被動用。

用法
----
    uv run python -m models.compare_engines            # 三格對照（2022）
    uv run python -m models.compare_engines seasons    # 季別穩定性
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import polars as pl

import models.hierarchical as v1
from models.hierarchical_v2 import encode, run_nuts, shadow_frame

CHAINS, WARMUP, SAMPLES = 4, 1000, 1000


def set_devices(n: int = CHAINS) -> None:
    """必須在 import jax 之前呼叫，見 hierarchical_v2.run_nuts 的 docstring。"""
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n}"


def fit_vb(df: pl.DataFrame) -> tuple[dict, pl.DataFrame, float]:
    t0 = time.time()
    model, res = v1.fit(df)
    eff = v1.extract_effects(model, res).filter(pl.col("group") == "catcher")
    return v1.variance_components(model, res), eff, time.time() - t0


def fit_nuts(df: pl.DataFrame) -> tuple[dict, pl.DataFrame, float, dict]:
    import jax
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    d = encode(df)
    t0 = time.time()
    mcmc = run_nuts(d, num_warmup=WARMUP, num_samples=SAMPLES, chains=CHAINS, progress=False)
    jax.block_until_ready(mcmc.get_samples())      # lazy array，不 block 會量到派工時間
    elapsed = time.time() - t0

    s, sc = mcmc.get_samples(), mcmc.get_samples(group_by_chain=True)
    tau = {g: np.asarray(s[f"tau_{g}"]) for g in ("catcher", "umpire", "pitcher")}
    u = np.asarray(s["u_catcher"])
    diag = {
        "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
        "r_hat": {g: float(split_gelman_rubin(np.asarray(sc[f"tau_{g}"]))) for g in tau},
        "ess": {g: int(effective_sample_size(np.asarray(sc[f"tau_{g}"]))) for g in tau},
        "p_umpire_gt_catcher": float((tau["umpire"] > tau["catcher"]).mean()),
    }
    eff = pl.DataFrame({"mlbam_id": d["catcher_levels"],
                        "effect_mean": u.mean(0), "effect_sd": u.std(0)})
    return {g: float(v.mean()) for g, v in tau.items()}, eff, elapsed, diag


def three_cells(season: int = 2022) -> None:
    df_a = v1.load_frame(season)                    # 全量 + in-sample 基準
    df_b = shadow_frame(seasons=[season])           # shadow + out-of-fold 基準

    tau_a, eff_a, t_a = fit_vb(df_a)
    tau_b, eff_b, t_b = fit_vb(df_b)
    tau_c, eff_c, t_c, diag = fit_nuts(df_b)

    print(f"\n{season} 三格對照")
    hdr = f"{'':32s} {'n':>9} {'catcher':>9} {'umpire':>9} {'pitcher':>9} {'秒':>6}"
    print(hdr); print("-" * len(hdr))
    for tag, tau, n, el in (("A  VB   / 全量   / in-sample", tau_a, df_a.height, t_a),
                            ("B  VB   / shadow / out-of-fold", tau_b, df_b.height, t_b),
                            ("C  NUTS / shadow / out-of-fold", tau_c, df_b.height, t_c)):
        print(f"{tag:32s} {n:>9,} {tau['catcher']:9.4f} {tau['umpire']:9.4f} "
              f"{tau['pitcher']:9.4f} {el:6.0f}")

    print(f"\nNUTS 診斷：divergences {diag['divergences']}/{CHAINS*SAMPLES}  "
          f"r_hat {max(diag['r_hat'].values()):.4f}  ESS(min) {min(diag['ess'].values())}")
    print(f"P(tau_umpire > tau_catcher) = {diag['p_umpire_gt_catcher']:.3f}")

    j = eff_b.join(eff_c, on="mlbam_id", suffix="_nuts")
    ratio = j["effect_sd"].median() / j["effect_sd_nuts"].median()
    print(f"\nB vs C（同資料，只差引擎），共同捕手 {j.height} 位")
    print(f"  效果 Pearson  {np.corrcoef(j['effect_mean'], j['effect_mean_nuts'])[0, 1]:.4f}")
    print(f"  效果 Spearman {j.select(pl.corr('effect_mean', 'effect_mean_nuts', method='spearman')).item():.4f}")
    print(f"  VB 宣稱的不確定性是 NUTS 的 {ratio:.2f} 倍")


def season_stability() -> None:
    print(f"\n變異成分的季別穩定性（NUTS / shadow / out-of-fold）")
    hdr = f"{'':16s} {'n':>9} {'catcher':>9} {'umpire':>9} {'pitcher':>9} {'P(ump>cat)':>11}"
    print(hdr); print("-" * len(hdr))
    for tag, seasons in (("2021", [2021]), ("2022", [2022]), ("2021+2022", None)):
        df = shadow_frame(seasons=seasons)
        tau, _, _, diag = fit_nuts(df)
        print(f"{tag:16s} {df.height:>9,} {tau['catcher']:9.4f} {tau['umpire']:9.4f} "
              f"{tau['pitcher']:9.4f} {diag['p_umpire_gt_catcher']:11.3f}", flush=True)


if __name__ == "__main__":
    set_devices()
    if len(sys.argv) > 1 and sys.argv[1] == "seasons":
        season_stability()
    else:
        three_cells()
