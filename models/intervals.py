"""Step 5：把不確定性報出來。

v1 這一整塊是空的——`hierarchical.py` 把每位捕手的 effect_sd 算出來了，然後整個
repo 沒有任何一處用它，leaderboard、圖、README 全是點值。

這裡產出三種東西：

1. 每位捕手的 Δ（每球額外好球率）後驗與 95% 區間
2. 成對比較機率 P(A 的效果 > B 的效果)，直接從後驗樣本數
3. 換算成 framing runs 時**把區間一起傳遞**，不是只傳點值

Δ 的定義和 `sim/estimate.py` 一樣，所以 Step 6 檢驗的正是這裡要報的數字：

    Δ_c = mean_i [ P(strike | 捕手 c 接) − P(strike | 聯盟平均捕手接) ]

模擬告訴我們怎麼讀這些區間
--------------------------
- 整體 coverage 在無混淆時 94%，可信。
- **效果最大的三分之一，coverage 只有 87–89%**，不是 95%。收縮把極端往內拉，
  而榜單上大家會看的正是那幾位。兩端的區間比它看起來的更不可靠。
- 若存在與捕手相關的未觀測混淆（例如投手群），混淆到真效果一半以內時區間還撐得住，
  等大時 coverage 掉到 88%，兩倍時 66%。這不是能從資料裡查出來的，只能標明。

圖由 `make_figures_v2.py` 產生（與 v1 的 `make_figures.py` 同樣的分工：分析在
models/，作圖與雙語標籤在根目錄的 make_figures*.py）。

用法
----
    uv run python -m models.intervals
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl

from models.baseline_gam import ARTIFACT_DIR

RUN_VALUE = 0.125          # 與 v1、Savant 相同的固定值
CACHE = ARTIFACT_DIR / "delta_posterior_train.pkl"
WARMUP, SAMPLES, CHAINS = 1000, 1000, 4


def fit_delta_posterior(force: bool = False, n_draws: int = 1000) -> dict:
    """擬合合併 train pool 的 shadow zone，回傳每位捕手的 Δ 後驗樣本。"""
    if CACHE.exists() and not force:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    import jax

    from models.hierarchical_v2 import encode, run_nuts, shadow_frame

    df = shadow_frame()                      # 2021+2022
    d = encode(df)
    mcmc = run_nuts(d, num_warmup=WARMUP, num_samples=SAMPLES, chains=CHAINS, progress=False)
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

    out = {
        "draws": draws,                                     # (n_draws, n_catchers)
        "catcher_ids": d["catcher_levels"],
        "n_pitches": np.diff(bounds),
        "tau": {g: np.asarray(s[f"tau_{g}"]) for g in ("catcher", "umpire", "pitcher")},
        "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


def leaderboard(post: dict, run_value: float = RUN_VALUE) -> pl.DataFrame:
    """每位捕手的 Δ 與 framing runs，都帶 95% 區間。

    runs = Δ × 該捕手的 shadow zone 球數 × run_value，逐抽樣計算後才取分位數，
    所以區間是傳遞過來的，不是把點估計乘一乘再配一個區間。
    """
    d, n = post["draws"], post["n_pitches"]
    runs = d * n * run_value
    q = lambda x, p: np.percentile(x, p, axis=0)
    return pl.DataFrame({
        "catcher": post["catcher_ids"],
        "shadow_pitches": n,
        "delta": d.mean(0),
        "delta_lo": q(d, 2.5), "delta_hi": q(d, 97.5),
        "runs": runs.mean(0),
        "runs_lo": q(runs, 2.5), "runs_hi": q(runs, 97.5),
        "p_positive": (d > 0).mean(0),
    }).sort("delta", descending=True)


def pairwise(post: dict, ids: np.ndarray | None = None) -> np.ndarray:
    """P(捕手 i 的效果 > 捕手 j 的效果)，直接從後驗樣本數。"""
    d = post["draws"]
    if ids is not None:
        idx = np.searchsorted(post["catcher_ids"], ids)
        d = d[:, idx]
    return (d[:, :, None] > d[:, None, :]).mean(0)


def separability(post: dict, min_pitches: int = 0) -> dict:
    """有多少捕手真的分得出來——事前講好要照實報的那個數字。

    min_pitches 可以只看樣本夠大的捕手：148 位裡有不少只接了一兩百顆球，他們的
    估計幾乎完全被收縮掉，混在一起會低估「樣本夠時分不分得出來」。
    """
    lb = leaderboard(post).filter(pl.col("shadow_pitches") >= min_pitches)
    # 依效果排序重排後驗，否則下面的「前五名」抓到的是 ID 順序的前五個
    idx = np.searchsorted(post["catcher_ids"], lb["catcher"].to_numpy())
    d = post["draws"][:, idx]
    n = d.shape[1]
    pm = (d[:, :, None] > d[:, None, :]).mean(0)
    iu = np.triu_indices(n, 1)
    return {
        "n_catchers": n,
        "nonzero_95": int(((lb["delta_lo"] > 0) | (lb["delta_hi"] < 0)).sum()),
        "pairs_total": len(iu[0]),
        "pairs_resolved_95": int(((pm[iu] > 0.95) | (pm[iu] < 0.05)).sum()),
        "top5_vs_rest_min_p": float(pm[:5, 5:].min()),
        "rank1_vs_rank2_p": float(pm[0, 1]),
        "rank1_vs_rank10_p": float(pm[0, min(9, n - 1)]),
    }


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    post = fit_delta_posterior()
    lb = leaderboard(post)
    sep = separability(post)

    print(f"捕手 {sep['n_catchers']} 位，divergence {post['divergences']}")
    print(f"\n變異成分後驗平均：" + "  ".join(
        f"{g}={v.mean():.4f}" for g, v in post["tau"].items()))
    tc, tu = post["tau"]["catcher"], post["tau"]["umpire"]
    print(f"P(tau_umpire > tau_catcher) = {(tu > tc).mean():.3f}")

    print(f"\n前五名")
    print(lb.head(5).select("catcher", "shadow_pitches", "delta", "delta_lo", "delta_hi",
                            "runs", "runs_lo", "runs_hi").to_pandas().round(4).to_string(index=False))
    print(f"\n後五名")
    print(lb.tail(5).select("catcher", "shadow_pitches", "delta", "delta_lo", "delta_hi",
                            "runs", "runs_lo", "runs_hi").to_pandas().round(4).to_string(index=False))

    print(f"\n分得出來嗎")
    for mp in (0, 500, 1000):
        sp = separability(post, min_pitches=mp)
        tag = "全部" if mp == 0 else f"≥{mp} 球"
        print(f"  [{tag:>8}] {sp['n_catchers']:>3} 位   "
              f"區間不含 0：{sp['nonzero_95']:>3} ({sp['nonzero_95']/sp['n_catchers']:.0%})   "
              f"配對分得出勝負：{sp['pairs_resolved_95']/sp['pairs_total']:.1%}   "
              f"P(第1 > 第2)={sp['rank1_vs_rank2_p']:.2f}  "
              f"P(第1 > 第10)={sp['rank1_vs_rank10_p']:.2f}")

