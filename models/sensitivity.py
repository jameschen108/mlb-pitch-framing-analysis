"""Step 8：兩組敏感度——b 的係數，以及 shadow zone 的門檻。

文件一度把 `b·logit_base` 稱為 offset。offset 的定義是係數固定為 1，而這裡的 b 有
先驗、會被資料更新，所以那個詞是錯的。但「用語錯了」和「結論錯了」是兩件事，而
只有後者才真的要緊。這支程式量的是後者：同一批球、同一個種子，一次讓 b 自由，一次
把它釘死在 1，然後比 τ、比每位捕手的 Δ、比榜單。

**跑在 2021–2022 的 train pool，不跑 2023。** 2023 是鎖定的評估集，已經花掉一次
（models/holdout.py）。為了敏感度再擬合一次就是再花一次，而敏感度檢查本來就該在
開發資料上做——這正是 train pool 存在的理由。

第二組是門檻。METHODS §2.3 事前承諾過三個 shadow zone 門檻都要報告，也承諾過不挑
區間最窄的那個。承諾寫下來了，數字一直沒補；`run_thresholds()` 在 0.15/0.85 與
0.25/0.75 各擬合一次，0.20/0.80 沿用上面那個自由 arm，不重跑。

每次擬合約七到十分鐘（門檻越寬球越多越慢），結果快取在 models/artifacts/（gitignored）。

用法
----
    uv run python -m models.sensitivity
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl

from models.baseline_gam import ARTIFACT_DIR
from models.intervals import RUN_VALUE, separability

CACHE = ARTIFACT_DIR / "sensitivity_slope.pkl"
WARMUP, SAMPLES, CHAINS = 1000, 1000, 4
SEED = 0
ARMS = {"free": None, "fixed_1": 1.0}       # slope=None 就是自由估計

T_CACHE = ARTIFACT_DIR / "sensitivity_threshold.pkl"
REFERENCE = (0.20, 0.80)                    # 主文用的門檻
ALTERNATES = ((0.15, 0.85), (0.25, 0.75))   # METHODS §2.3 事前承諾要一起報的兩個


def run(force: bool = False, n_draws: int = 1000) -> dict:
    """兩個 arm 各擬合一次，回傳 b、τ 與每位捕手的 Δ 後驗。"""
    if CACHE.exists() and not force:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    import jax

    from models.hierarchical_v2 import delta_draws, encode, run_nuts, shadow_frame

    df = shadow_frame()                      # 2021 + 2022
    d = encode(df)
    lb = df["logit_base"].to_numpy()

    out = {"n_rows": df.height, "arms": {}}
    for arm, slope in ARMS.items():
        print(f"[{arm}] fit 中…（slope={'自由' if slope is None else slope}）", flush=True)
        mcmc = run_nuts(d, num_warmup=WARMUP, num_samples=SAMPLES, chains=CHAINS,
                        seed=SEED, progress=False, slope=slope)
        jax.block_until_ready(mcmc.get_samples())
        s = mcmc.get_samples()

        draws, n = delta_draws(s, d, lb, n_draws=n_draws)
        out["arms"][arm] = {
            "a": np.asarray(s["a"]),
            "b": np.asarray(s["b"]),
            "tau": {g: np.asarray(s[f"tau_{g}"]) for g in ("catcher", "umpire", "pitcher")},
            "draws": draws,
            "catcher_ids": d["catcher_levels"],
            "n_pitches": n,
            "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
        }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


def summary(res: dict) -> pl.DataFrame:
    """每個 arm 一列：b、三個 τ、榜單全距、divergence。"""
    rows = []
    for arm, r in res["arms"].items():
        runs = r["draws"] * r["n_pitches"] * RUN_VALUE
        mean_runs = runs.mean(0)
        row = {
            "arm": arm,
            "n_catchers": len(r["catcher_ids"]),
            "b_mean": float(r["b"].mean()),
            "b_lo": float(np.percentile(r["b"], 2.5)),
            "b_hi": float(np.percentile(r["b"], 97.5)),
            "a_mean": float(r["a"].mean()),
            "leaderboard_spread_runs": float(mean_runs.max() - mean_runs.min()),
            "divergences": r["divergences"],
        }
        for g, v in r["tau"].items():
            row[f"tau_{g}"] = float(v.mean())
            row[f"tau_{g}_lo"] = float(np.percentile(v, 5.5))
            row[f"tau_{g}_hi"] = float(np.percentile(v, 94.5))
        row["p_umpire_gt_catcher"] = float(
            (r["tau"]["umpire"] > r["tau"]["catcher"]).mean()
        )
        rows.append(row)
    return pl.DataFrame(rows)


def comparison(res: dict) -> dict:
    """自由 b 與 b=1 之間，榜單實際差多少。

    問的是「結論會不會變」，所以比的是會被發表的那些量：每位捕手的 framing runs、
    名次、以及榜單全距。和 README §2 報告校準修正時用的是同一把尺——首尾位移對上
    榜單全距——這樣兩個敏感度檢查才讀得起來。
    """
    a, b = res["arms"]["free"], res["arms"]["fixed_1"]
    assert np.array_equal(a["catcher_ids"], b["catcher_ids"])

    ra = (a["draws"] * a["n_pitches"] * RUN_VALUE).mean(0)
    rb = (b["draws"] * b["n_pitches"] * RUN_VALUE).mean(0)
    rank_a, rank_b = np.argsort(np.argsort(-ra)), np.argsort(np.argsort(-rb))

    return {
        "n_catchers": len(ra),
        "runs_pearson": float(np.corrcoef(ra, rb)[0, 1]),
        "runs_spearman": float(pl.DataFrame({"a": ra, "b": rb})
                               .select(pl.corr("a", "b", method="spearman")).item()),
        "runs_max_abs_diff": float(np.abs(ra - rb).max()),
        "runs_mean_abs_diff": float(np.abs(ra - rb).mean()),
        "spread_free": float(ra.max() - ra.min()),
        "spread_fixed": float(rb.max() - rb.min()),
        "max_rank_change": int(np.abs(rank_a - rank_b).max()),
        "top10_overlap": int(len(set(np.argsort(-ra)[:10]) & set(np.argsort(-rb)[:10]))),
        "tau_catcher_shift": float(a["tau"]["catcher"].mean() - b["tau"]["catcher"].mean()),
        "tau_umpire_shift": float(a["tau"]["umpire"].mean() - b["tau"]["umpire"].mean()),
    }



# ---- 門檻敏感度 ----

def run_thresholds(force: bool = False, n_draws: int = 1000) -> dict:
    """在 0.15/0.85 與 0.25/0.75 上各擬合一次，0.20/0.80 沿用 run() 的自由 arm。

    METHODS §2.3 事前承諾過三個門檻都要報，也承諾過不挑區間最窄的那個。承諾寫下
    來了，數字一直沒補；這裡補。

    三個門檻的球集不同，捕手集合也就不同（門檻越窄，達到門檻的捕手越少），所以跨
    門檻的比較一律只取共同捕手。Δ 本身是「每顆 shadow zone 球的額外好球率」，分母
    跟著門檻走，換算成 runs 之後才可比。
    """
    if T_CACHE.exists() and not force:
        with open(T_CACHE, "rb") as fh:
            return pickle.load(fh)

    import jax

    from models.hierarchical_v2 import delta_draws, encode, run_nuts, shadow_frame

    out = {"arms": {}}

    ref = run()["arms"]["free"]              # 0.20/0.80，已經跑過，不重跑
    out["arms"][REFERENCE] = {k: ref[k] for k in
                              ("tau", "draws", "catcher_ids", "n_pitches", "divergences")}
    out["arms"][REFERENCE]["n_rows"] = run()["n_rows"]

    for lo, hi in ALTERNATES:
        print(f"[{lo}/{hi}] fit 中…", flush=True)
        df = shadow_frame(lo=lo, hi=hi)
        d = encode(df)
        mcmc = run_nuts(d, num_warmup=WARMUP, num_samples=SAMPLES, chains=CHAINS,
                        seed=SEED, progress=False)
        jax.block_until_ready(mcmc.get_samples())
        s = mcmc.get_samples()

        draws, n = delta_draws(s, d, df["logit_base"].to_numpy(), n_draws=n_draws)
        out["arms"][(lo, hi)] = {
            "tau": {g: np.asarray(s[f"tau_{g}"]) for g in ("catcher", "umpire", "pitcher")},
            "draws": draws,
            "catcher_ids": d["catcher_levels"],
            "n_pitches": n,
            "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
            "n_rows": df.height,
        }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(T_CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


def threshold_summary(res: dict) -> pl.DataFrame:
    """每個門檻一列：規模、τ、分辨力，以及對照 0.20/0.80 的榜單相關。

    「不挑區間最窄的那個」這句承諾要能被查核，所以 `mean_ci_width_runs` 必須在表上：
    讀者可以自己確認主文用的門檻不是三個裡面最窄的。
    """
    ref = res["arms"][REFERENCE]
    ref_runs = dict(zip(ref["catcher_ids"],
                        (ref["draws"] * ref["n_pitches"] * RUN_VALUE).mean(0)))

    rows = []
    for (lo, hi), r in sorted(res["arms"].items()):
        runs = r["draws"] * r["n_pitches"] * RUN_VALUE
        mean_runs = runs.mean(0)
        width = np.percentile(runs, 97.5, axis=0) - np.percentile(runs, 2.5, axis=0)
        sep = separability(r, min_pitches=0)

        common = [i for i, c in enumerate(r["catcher_ids"]) if c in ref_runs]
        x = mean_runs[common]
        y = np.array([ref_runs[c] for c in r["catcher_ids"][common]])

        rows.append({
            "lo": lo, "hi": hi,
            "is_reference": (lo, hi) == REFERENCE,
            "n_rows": r["n_rows"],
            "n_catchers": len(r["catcher_ids"]),
            "tau_catcher": float(r["tau"]["catcher"].mean()),
            "tau_umpire": float(r["tau"]["umpire"].mean()),
            "tau_pitcher": float(r["tau"]["pitcher"].mean()),
            "p_umpire_gt_catcher": float(
                (r["tau"]["umpire"] > r["tau"]["catcher"]).mean()),
            "intervals_excluding_zero": sep["nonzero_95"],
            "share_excluding_zero": sep["nonzero_95"] / sep["n_catchers"],
            "share_pairs_resolved": sep["pairs_resolved_95"] / sep["pairs_total"],
            "mean_ci_width_runs": float(width.mean()),
            "leaderboard_spread_runs": float(mean_runs.max() - mean_runs.min()),
            "n_common_with_reference": len(common),
            "runs_pearson_vs_reference": (
                1.0 if (lo, hi) == REFERENCE else float(np.corrcoef(x, y)[0, 1])),
            "divergences": r["divergences"],
        })
    return pl.DataFrame(rows)


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

    res = run()
    s, c = summary(res), comparison(res)

    print(f"\ntrain pool（2021–2022）{res['n_rows']:,} 顆 shadow zone 球\n")
    print(s.to_pandas().round(4).to_string(index=False))

    free = res["arms"]["free"]["b"]
    print(f"\n=== b 的後驗（自由估計）===")
    print(f"  平均 {free.mean():.4f}   95% 區間 "
          f"[{np.percentile(free, 2.5):.4f}, {np.percentile(free, 97.5):.4f}]")
    print(f"  P(b > 1) = {(free > 1).mean():.3f}"
          f"   →  {'區間不含 1，offset 的假設被資料否定' if np.percentile(free, 2.5) > 1 or np.percentile(free, 97.5) < 1 else '區間仍含 1'}")

    print(f"\n=== 釘死 b=1 之後，榜單動了多少 ===")
    print(f"  framing runs 相關  Pearson {c['runs_pearson']:.4f}  "
          f"Spearman {c['runs_spearman']:.4f}")
    print(f"  單一捕手最大位移   {c['runs_max_abs_diff']:.2f} runs"
          f"（平均 {c['runs_mean_abs_diff']:.2f}）")
    print(f"  榜單全距           自由 {c['spread_free']:.1f} → 固定 {c['spread_fixed']:.1f} runs")
    print(f"  最大名次變動       {c['max_rank_change']} 名，前十重疊 {c['top10_overlap']}/10")
    print(f"  τ 捕手位移         {c['tau_catcher_shift']:+.4f}"
          f"   τ 主審位移 {c['tau_umpire_shift']:+.4f}")

    print(f"\n=== shadow zone 門檻敏感度（事前承諾的三個）===")
    ts = threshold_summary(run_thresholds())
    print(ts.to_pandas().round(4).to_string(index=False))
    narrowest = ts.sort("mean_ci_width_runs")["lo"][0], ts.sort("mean_ci_width_runs")["hi"][0]
    print(f"\n  區間最窄的門檻是 {narrowest[0]}/{narrowest[1]}"
          f"；主文用的是 {REFERENCE[0]}/{REFERENCE[1]}。")
