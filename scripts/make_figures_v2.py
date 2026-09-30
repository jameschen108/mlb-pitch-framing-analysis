"""產生 v2 的 README 圖，英中各一套。

沿用 make_figures.py 的慣例：輸出到 docs/images/{en,zh}/，文字全部走 LABELS，
圖形邏輯只寫一次。

    uv run python -m scripts.make_figures_v2 --lang both
    uv run python -m scripts.make_figures_v2 --only abs_tau   # 只畫 ABS 後記那張
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CJK_FONTS = ["PingFang TC", "Arial Unicode MS", "Heiti TC"]
NOMINAL = 0.95

COLOR = {"hierarchical": "#1f77b4", "residual_runs": "#d62728", "abs": "#d95f02"}

LABELS = {
    "en": {
        "hier": "Hierarchical (NUTS)",
        "resid": "Residual runs (unadjusted)",
        "nominal": "nominal 95%",
        "sweep_title": "How much confounding before the intervals stop meaning anything",
        "sweep_x": "Unmeasured confounding, relative to the true catcher effect",
        "sweep_y": "95% interval coverage",
        "sweep_ticks": ["none", "0.25×", "0.5×", "1× (equal)", "2×"],
        "cov_title": "Interval coverage by simulated scenario",
        "cov_x": "95% interval coverage    ●  all catchers     ○  largest third of effects",
        "cat_title": "Catcher framing effect, 2023, with 95% credible intervals",
        "cat_x": "Catchers, ranked by estimated effect  ({k} of {n} have intervals excluding zero)",
        "cat_y": "Extra called strikes per 100 shadow-zone pitches",
        "cat_note": ("Simulation puts coverage at the extremes at 87–91%, not 95% — shrinkage pulls "
                     "the ends in, so the top and bottom are less firm than they look."),
        "abs_title": "Catcher-to-catcher variation in umpires' original calls, by season",
        "abs_y": "τ catcher (logit scale), 95% interval",
        "abs_groups": ("baseline period", "may have\nadjusted early", "ABS\nchallenges"),
        "abs_note": ("Umpire's call before any challenge. Strike zone from batter height in every season; "
                     "baseline refit within each season."),
        "scenarios": {
            "baseline": "baseline", "unequal": "unequal workloads",
            "umpire_confound": "umpire confounding", "battery": "battery pairing",
            "location_shared": "location effect (shared)", "location_mix": "location effect (mixed)",
            "heavy_tail": "heavy-tailed effects", "omitted_covariate": "omitted covariate",
        },
    },
    "zh": {
        "hier": "階層模型（NUTS）",
        "resid": "未調整殘差",
        "nominal": "名目 95%",
        "sweep_title": "混淆要多大，區間才會失去意義",
        "sweep_x": "未觀測混淆的大小（相對於真實捕手效果）",
        "sweep_y": "95% 區間的實際涵蓋率",
        "sweep_ticks": ["無", "0.25 倍", "0.5 倍", "1 倍（等大）", "2 倍"],
        "cov_title": "各模擬情境下的區間涵蓋率",
        "cov_x": "95% 區間涵蓋率    ●  全體捕手     ○  效果最大的三分之一",
        "cat_title": "2023 年捕手 framing 效果與 95% 可信區間",
        "cat_x": "捕手（依估計效果排序，{n} 位中有 {k} 位的區間不含零）",
        "cat_y": "每 100 顆 shadow zone 球的額外好球數",
        "cat_note": "模擬顯示兩端的涵蓋率只有 87–91%，而非 95%：收縮把極端往內拉，上下兩端比看起來的更不確定。",
        "abs_title": "主審原判中捕手之間的差異，逐季",
        "abs_y": "τ 捕手（logit 尺度），95% 區間",
        "abs_groups": ("基準期", "可能已提前調整", "ABS 挑戰上路"),
        "abs_note": "主審在挑戰之前的原判。每一季都用身高定義好球帶，基準模型在各季內重新擬合。",
        "scenarios": {
            "baseline": "基準", "unequal": "樣本不均",
            "umpire_confound": "主審混淆", "battery": "投捕綁定",
            "location_shared": "位置效果（球位共用）", "location_mix": "位置效果（球位因人而異）",
            "heavy_tail": "重尾效果", "omitted_covariate": "遺漏變數",
        },
    },
}


def use_style(lang: str) -> None:
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.unicode_minus"] = False
    if lang == "zh":
        plt.rcParams["font.sans-serif"] = CJK_FONTS


def _style(L):
    return {"hierarchical": dict(color=COLOR["hierarchical"], marker="o", label=L["hier"]),
            "residual_runs": dict(color=COLOR["residual_runs"], marker="s", label=L["resid"])}


def sweep(L, out: Path, n_reps: int = 50) -> None:
    res = pl.read_parquet(ROOT / "sim" / "results" / f"confound_sweep_{n_reps}reps.parquet")
    agg = (res.group_by("strength", "estimator")
           .agg(coverage=pl.col("coverage").mean(),
                se=pl.col("coverage").std() / np.sqrt(pl.col("coverage").len()))
           .sort("strength"))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhline(NOMINAL, color="#555", lw=1, ls="--", zorder=1)
    ax.text(0.02, NOMINAL + 0.012, L["nominal"], fontsize=8.5, color="#555")
    for est, st in _style(L).items():
        d = agg.filter(pl.col("estimator") == est)
        x, y, se = d["strength"].to_numpy(), d["coverage"].to_numpy(), d["se"].to_numpy()
        ax.plot(x, y, lw=1.8, ms=6, zorder=3, **st)
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=st["color"], alpha=0.15, zorder=2)
    ax.set_xlabel(L["sweep_x"]); ax.set_ylabel(L["sweep_y"])
    ax.set_title(L["sweep_title"], fontsize=11.5, pad=11)
    ax.set_xticks([0, 0.25, 0.5, 1.0, 2.0]); ax.set_xticklabels(L["sweep_ticks"])
    ax.set_ylim(0.35, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "sim_confound_sweep.png", dpi=170); plt.close(fig)


def coverage(L, out: Path, n_reps: int = 100) -> None:
    from sim.generate import SCENARIOS
    res = pl.read_parquet(ROOT / "sim" / "results" / f"sim_all_{n_reps}reps.parquet")
    agg = (res.group_by("scenario", "estimator")
           .agg(coverage=pl.col("coverage").mean(), extreme=pl.col("coverage_extreme").mean()))
    order = {s: i for i, s in enumerate(SCENARIOS)}
    agg = agg.with_columns(o=pl.col("scenario").replace_strict(order, return_dtype=pl.Int32)).sort("o")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    y, off = np.arange(len(SCENARIOS)), 0.17
    ax.axvline(NOMINAL, color="#333", lw=1.2, ls="--", zorder=1)
    ax.text(NOMINAL + 0.004, -0.75, L["nominal"], fontsize=8.5, color="#333")
    for k, (est, st) in enumerate(_style(L).items()):
        d = agg.filter(pl.col("estimator") == est)
        yy = y + (0.5 - k) * 2 * off
        cov, ext = d["coverage"].to_numpy(), d["extreme"].to_numpy()
        ax.hlines(yy, np.minimum(cov, ext), np.maximum(cov, ext), color=st["color"],
                  lw=1.2, alpha=0.45, zorder=2)
        ax.scatter(cov, yy, s=54, color=st["color"], zorder=4, label=st["label"])
        ax.scatter(ext, yy, s=30, facecolor="white", edgecolor=st["color"], lw=1.4, zorder=3)
    for i in range(1, len(SCENARIOS)):
        ax.axhline(i - 0.5, color="#ddd", lw=0.6, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([L["scenarios"][s] for s in SCENARIOS], fontsize=9.5)
    ax.set_ylim(len(SCENARIOS) - 0.5, -1.1); ax.set_xlim(0.68, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(L["cov_x"]); ax.set_title(L["cov_title"], fontsize=11.5, pad=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig(out / "sim_coverage_by_scenario.png", dpi=170); plt.close(fig)


def caterpillar(L, out: Path) -> None:
    from models.holdout import run
    from models.intervals import leaderboard
    lb = leaderboard(run())
    x = np.arange(lb.height)
    est = lb["delta"].to_numpy() * 100
    lo, hi = lb["delta_lo"].to_numpy() * 100, lb["delta_hi"].to_numpy() * 100
    sig = (lo > 0) | (hi < 0)

    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.axhline(0, color="#333", lw=1, zorder=3)
    for mask, col, alpha in ((~sig, "#9aa5b1", 0.75), (sig, COLOR["hierarchical"], 0.95)):
        ax.vlines(x[mask], lo[mask], hi[mask], color=col, lw=1.3, alpha=alpha, zorder=2)
        ax.scatter(x[mask], est[mask], s=8, color=col, zorder=4)
    ax.set_xlabel(L["cat_x"].format(k=int(sig.sum()), n=lb.height))
    ax.set_ylabel(L["cat_y"]); ax.set_title(L["cat_title"], fontsize=11.5, pad=11)
    ax.set_xlim(-2, lb.height + 1)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.text(0.5, 0.025, L["cat_note"], ha="center", va="bottom", fontsize=8.5, color="#666")
    fig.savefig(out / "caterpillar_2023.png", dpi=170); plt.close(fig)


def abs_tau(L, out: Path) -> None:
    """ABS 後記：各季 τ_catcher，原判、ABS 好球帶、逐季基準模型（models/abs_era.py）。"""
    from models.abs_era import ABS_SEASON, ANTICIPATION_SEASON, tau_table
    t = tau_table("abs").filter(pl.col("group") == "catcher").sort("season")
    s = t["season"].to_numpy()
    m, lo, hi = (t[c].to_numpy() for c in ("tau_mean", "tau_eti_lo", "tau_eti_hi"))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, season in enumerate(s):
        col = COLOR["abs"] if season == ABS_SEASON else COLOR["hierarchical"]
        hollow = season == ANTICIPATION_SEASON
        ax.vlines(season, lo[i], hi[i], color=col, lw=2, zorder=2)
        ax.scatter(season, m[i], s=64, zorder=3, color="white" if hollow else col,
                   edgecolor=col, linewidth=2)
    ax.annotate(f"{m[-1]:.3f}", (s[-1], m[-1]), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=9, color="#333")
    ymin = lo.min() - 0.035
    for (a, b), text in zip(((s[0], ANTICIPATION_SEASON - 1), (ANTICIPATION_SEASON,) * 2,
                             (ABS_SEASON,) * 2), L["abs_groups"]):
        ax.plot([a - 0.3, b + 0.3], [ymin + 0.012] * 2, color="#999", lw=1)
        ax.text((a + b) / 2, ymin, text, ha="center", va="top", fontsize=8.5, color="#555")
    ax.set_xticks(s); ax.set_xlim(s[0] - 0.6, s[-1] + 0.8)
    ax.set_ylim(ymin - 0.03, hi.max() + 0.02)
    ax.set_ylabel(L["abs_y"]); ax.set_title(L["abs_title"], fontsize=11.5, pad=11)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.02, L["abs_note"], ha="center", va="bottom", fontsize=8, color="#666")
    fig.savefig(out / "abs_tau_by_season.png", dpi=170); plt.close(fig)


FIGURES = {"sweep": sweep, "coverage": coverage, "caterpillar": caterpillar, "abs_tau": abs_tau}


def build(lang: str, only: list[str] | None = None) -> None:
    L = LABELS[lang]
    out = ROOT / "docs" / "images" / lang
    out.mkdir(parents=True, exist_ok=True)
    use_style(lang)
    print(f"[{lang}]")
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        fn(L, out)
        print(f"  {fn.__name__}")


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["en", "zh", "both"], default="both")
    p.add_argument("--only", nargs="+", choices=list(FIGURES), help="只畫這幾張")
    a = p.parse_args()
    for lang in (["en", "zh"] if a.lang == "both" else [a.lang]):
        build(lang, a.only)
