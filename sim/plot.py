"""模擬結果作圖。

輸出到 figures/（gitignored），跟 notebook 的圖一樣可重新產生。README 要用的圖
之後走 make_figures.py 的雙語流程，這裡先不處理語言問題。

用法
----
    uv run python -m sim.plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = Path(__file__).resolve().parent.parent / "figures"
NOMINAL = 0.95

STYLE = {
    "hierarchical": dict(color="#1f77b4", marker="o", label="Hierarchical (NUTS)"),
    "residual_runs": dict(color="#d62728", marker="s", label="Residual runs (unadjusted)"),
}


def confound_curve(n_reps: int = 50) -> Path:
    """coverage 隨未觀測混淆強度的變化。"""
    res = pl.read_parquet(RESULTS / f"confound_sweep_{n_reps}reps.parquet")
    agg = (res.group_by("strength", "estimator")
           .agg(coverage=pl.col("coverage").mean(),
                se=pl.col("coverage").std() / np.sqrt(pl.col("coverage").len()))
           .sort("strength"))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhline(NOMINAL, color="#555", lw=1, ls="--", zorder=1)
    ax.text(0.02, NOMINAL + 0.012, "nominal 95%", fontsize=8.5, color="#555")

    for est, st in STYLE.items():
        d = agg.filter(pl.col("estimator") == est)
        x, y, se = d["strength"].to_numpy(), d["coverage"].to_numpy(), d["se"].to_numpy()
        ax.plot(x, y, lw=1.8, ms=6, zorder=3, **st)
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=st["color"], alpha=0.15, zorder=2)

    ax.set_xlabel("Unmeasured confounding, relative to the true catcher effect")
    ax.set_ylabel("95% interval coverage")
    ax.set_title("How much confounding before the intervals stop meaning anything",
                 fontsize=11.5, pad=11)
    ax.set_xticks([0, 0.25, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["none", "0.25×", "0.5×", "1× (equal)", "2×"])
    ax.set_ylim(0.35, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "sim_confound_sweep.png"
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def coverage_by_scenario(n_reps: int = 100) -> Path:
    """八個情境的 coverage，整體與尾端並列。"""
    from sim.generate import SCENARIOS

    res = pl.read_parquet(RESULTS / f"sim_all_{n_reps}reps.parquet")
    agg = (res.group_by("scenario", "estimator")
           .agg(coverage=pl.col("coverage").mean(),
                extreme=pl.col("coverage_extreme").mean()))
    order = {s: i for i, s in enumerate(SCENARIOS)}
    agg = agg.with_columns(o=pl.col("scenario").replace_strict(order, return_dtype=pl.Int32)).sort("o")

    # 用點圖不用長條：coverage 的差異全在 70–100% 之間，長條從 0 起會把它壓扁，
    # 而截斷長條的軸是會誤導人的。點圖沒有這個問題。
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    y = np.arange(len(SCENARIOS))
    off = 0.17

    ax.axvline(NOMINAL, color="#333", lw=1.2, ls="--", zorder=1)
    ax.text(NOMINAL + 0.004, -0.75, "nominal 95%", fontsize=8.5, color="#333")

    for k, (est, st) in enumerate(STYLE.items()):
        d = agg.filter(pl.col("estimator") == est)
        yy = y + (0.5 - k) * 2 * off
        cov, ext = d["coverage"].to_numpy(), d["extreme"].to_numpy()
        ax.hlines(yy, np.minimum(cov, ext), np.maximum(cov, ext),
                  color=st["color"], lw=1.2, alpha=0.45, zorder=2)
        ax.scatter(cov, yy, s=54, color=st["color"], zorder=4, label=st["label"])
        ax.scatter(ext, yy, s=30, facecolor="white", edgecolor=st["color"],
                   lw=1.4, zorder=3)

    for i in range(1, len(SCENARIOS)):
        ax.axhline(i - 0.5, color="#ddd", lw=0.6, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(SCENARIOS, fontsize=9.5)
    ax.set_ylim(len(SCENARIOS) - 0.5, -1.1)
    ax.set_xlim(0.68, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("95% interval coverage    ●  all catchers     ○  largest third of effects")
    ax.set_title("Coverage by scenario", fontsize=11.5, pad=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=9)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "sim_coverage_by_scenario.png"
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (confound_curve(), coverage_by_scenario()):
        print(f"已輸出 → {f}")
