"""重生 README 用的圖，中英各一套。

notebook 裡的圖是分析過程的產物（標籤是中文）；README 是給外部讀者看的門面，
英文版需要英文標籤。與其在 notebook 裡複製繪圖碼，這裡從既有的快取產物
（processed parquet、baseline GAM pickle、階層模型 artifacts）重新畫一次。

    uv run python make_figures.py          # 中英兩套
    uv run python make_figures.py --lang en

輸出：docs/images/en/、docs/images/zh/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "models" / "artifacts"

SEASONS = (2021, 2022, 2023)
ZONE_HALF_WIDTH = 0.83  # 本壘板半寬 + 球半徑（英尺）

# ---------------------------------------------------------------- 文案

LABELS = {
    "en": {
        "plate_x": "plate_x (ft, catcher's view)",
        "plate_z": "plate_z (standardized: 0 = bottom, 1 = top)",
        "p_strike": "P(called strike)",
        "heat_title": "Called-strike rate, 2023",
        "count_diff_title": "Hitter's counts minus pitcher's counts\n(same location; red = zone expands)",
        "count_diff_cbar": "Δ P(called strike)",
        "surface_gam": "GAM surface with 50% contour",
        "surface_emp": "Empirical called-strike rate",
        "contour_title": "50% called-strike contour by count",
        "counts": {(0, 2): "0-2 (pitcher ahead)", (1, 1): "1-1 (even)", (3, 0): "3-0 (hitter ahead)"},
        "nominal_zone": "nominal zone",
        "official_x": "Official framing runs (Baseball Savant)",
        "official_y": "This project, unadjusted residual leaderboard",
        "official_title": "Unadjusted residual leaderboard vs official, 2023",
        "tau_title": "Variance components by group",
        "tau_y": "Random-effect SD, τ (logit scale)",
        "tau_groups": ["Catcher", "Umpire", "Pitcher"],
        "shrink_x": "Naive framing runs (residual sum)",
        "shrink_y": "Hierarchical framing runs (shrunk, adjusted)",
        "shrink_title": "Shrinkage pulls small samples toward zero",
        "shrink_cbar": "called pitches",
        "yoy_x": "2022 framing runs / 1000 pitches",
        "yoy_y": "2023 framing runs / 1000 pitches",
        "yoy_title": "Year-over-year persistence, 2022 to 2023",
        "matrix_title": "Framing-rate correlation across seasons",
        "pooled_x": "Framing runs, 2021-2023 combined",
        "pooled_title": "Best framers, 2021-2023 combined",
        "traj_x": "Season",
        "traj_y": "Framing runs / 1000 pitches",
        "traj_title": "Season-by-season framing, best and worst",
        "identity": "y = x",
    },
    "zh": {
        "plate_x": "plate_x（英尺，捕手視角）",
        "plate_z": "標準化 plate_z（0 = 下緣，1 = 上緣）",
        "p_strike": "P(判好球)",
        "heat_title": "2023 好球判定率",
        "count_diff_title": "打者領先球數 減 投手領先球數\n（同位置比較，紅 = 好球帶擴張）",
        "count_diff_cbar": "Δ P(判好球)",
        "surface_gam": "GAM 曲面與 50% 等高線",
        "surface_emp": "實證好球判定率",
        "contour_title": "50% 好球機率等高線 × 球數",
        "counts": {(0, 2): "0-2（投手領先）", (1, 1): "1-1（平）", (3, 0): "3-0（打者領先）"},
        "nominal_zone": "名義好球帶",
        "official_x": "官方 framing runs（Baseball Savant）",
        "official_y": "本專案・未調整殘差榜單",
        "official_title": "未調整殘差榜單 vs 官方，2023",
        "tau_title": "三組效應的變異成分",
        "tau_y": "隨機效應標準差 τ（logit 尺度）",
        "tau_groups": ["捕手", "主審", "投手"],
        "shrink_x": "天真版 framing runs（殘差加總）",
        "shrink_y": "階層版 framing runs（收縮並控制混淆）",
        "shrink_title": "Shrinkage 把小樣本拉向 0",
        "shrink_cbar": "判定球數",
        "yoy_x": "2022 framing runs / 1000 顆",
        "yoy_y": "2023 framing runs / 1000 顆",
        "yoy_title": "年度間持續性，2022 → 2023",
        "matrix_title": "framing 率的跨季相關",
        "pooled_x": "framing runs，2021–2023 合計",
        "pooled_title": "2021–2023 最佳 framer",
        "traj_x": "球季",
        "traj_y": "framing runs / 1000 顆",
        "traj_title": "名將三季軌跡：最佳與最差",
        "identity": "y = x",
    },
}

CJK_FONTS = ["PingFang TC", "Arial Unicode MS", "Heiti TC"]


def use_style(lang: str) -> None:
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.unicode_minus"] = False
    if lang == "zh":
        plt.rcParams["font.sans-serif"] = CJK_FONTS


# ---------------------------------------------------------------- 共用

def draw_zone(ax) -> None:
    ax.plot(
        [-ZONE_HALF_WIDTH, ZONE_HALF_WIDTH, ZONE_HALF_WIDTH, -ZONE_HALF_WIDTH, -ZONE_HALF_WIDTH],
        [0, 0, 1, 1, 0],
        "k--", lw=1.2, alpha=0.7,
    )


def empirical_grid(df, bins=50, min_count=20, x_range=(-2, 2), z_range=(-0.5, 1.5)):
    x = df["plate_x"].to_numpy()
    z = df["plate_z_std"].to_numpy()
    s = df["is_strike"].to_numpy()
    xe = np.linspace(*x_range, bins + 1)
    ze = np.linspace(*z_range, bins + 1)
    ssum, _, _ = np.histogram2d(x, z, bins=[xe, ze], weights=s)
    tot, _, _ = np.histogram2d(x, z, bins=[xe, ze])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = ssum / tot
    rate[tot < min_count] = np.nan
    return rate, xe, ze


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir.name}/{name}.png")


# ---------------------------------------------------------------- 各圖

def fig_count_diff(L, df, out):
    hitter = df.filter(pl.col("balls") > pl.col("strikes"))
    pitcher = df.filter(pl.col("strikes") > pl.col("balls"))
    rh, xe, ze = empirical_grid(hitter, bins=40, min_count=15)
    rp, _, _ = empirical_grid(pitcher, bins=40, min_count=15)

    fig, ax = plt.subplots(figsize=(6.5, 7))
    mesh = ax.pcolormesh(xe, ze, (rh - rp).T, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    draw_zone(ax)
    ax.set_xlabel(L["plate_x"])
    ax.set_ylabel(L["plate_z"])
    ax.set_title(L["count_diff_title"])
    fig.colorbar(mesh, ax=ax, label=L["count_diff_cbar"], shrink=0.8)
    save(fig, out, "strike_rate_count_diff_2023")


def fig_surface_vs_empirical(L, df, gam, out):
    from models.baseline_gam import predict_grid

    XX, ZZ, P = predict_grid(gam, context={"stand_i": 1, "p_throws_i": 1, "balls": 0, "strikes": 0})
    rate, xe, ze = empirical_grid(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    axes[0].pcolormesh(XX, ZZ, P, cmap="RdBu_r", vmin=0, vmax=1)
    cs = axes[0].contour(XX, ZZ, P, levels=[0.5], colors="white", linewidths=2)
    axes[0].clabel(cs, fmt="50%%", fontsize=9)
    axes[0].set_title(L["surface_gam"])
    mesh = axes[1].pcolormesh(xe, ze, rate.T, cmap="RdBu_r", vmin=0, vmax=1)
    axes[1].set_title(L["surface_emp"])
    for ax in axes:
        draw_zone(ax)
        ax.set_xlabel(L["plate_x"])
    axes[0].set_ylabel(L["plate_z"])
    fig.colorbar(mesh, ax=axes, label=L["p_strike"], shrink=0.7)
    save(fig, out, "gam_surface_vs_empirical_2023")


def fig_count_contours(L, df, out):
    from models.baseline_gam import fit_by_count, predict_grid

    counts = [(0, 2), (1, 1), (3, 0)]
    colors = {(0, 2): "#2166ac", (1, 1): "#4d4d4d", (3, 0): "#b2182b"}
    gams = fit_by_count(df, counts)

    fig, ax = plt.subplots(figsize=(6.5, 7))
    for c in counts:
        GX, GZ, GP = predict_grid(gams[c], tensor_only=True)
        ax.contour(GX, GZ, GP, levels=[0.5], colors=[colors[c]], linewidths=2.2)
        ax.plot([], [], color=colors[c], lw=2.2, label=L["counts"][c])
    ax.plot(
        [-ZONE_HALF_WIDTH, ZONE_HALF_WIDTH, ZONE_HALF_WIDTH, -ZONE_HALF_WIDTH, -ZONE_HALF_WIDTH],
        [0, 0, 1, 1, 0], "k--", lw=1.2, alpha=0.6, label=L["nominal_zone"],
    )
    ax.set_xlabel(L["plate_x"])
    ax.set_ylabel(L["plate_z"])
    ax.set_title(L["contour_title"])
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.4, 1.4)
    save(fig, out, "gam_count_contours_2023")


def fig_official(L, merged, out):
    from scipy.stats import pearsonr, spearmanr

    x = merged["official_runs"].to_numpy()
    y = merged["framing_runs"].to_numpy()
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7, 7))
    lim = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6, label=L["identity"])
    ax.scatter(x, y, s=28, alpha=0.75, color="#c0392b", edgecolor="white", linewidth=0.5)
    labelled = pl.concat([
        merged.sort("official_runs", descending=True).head(3),
        merged.sort("official_runs").head(2),
    ]).unique(subset="mlbam_id")
    for r in labelled.iter_rows(named=True):
        ax.annotate(r["name"], (r["official_runs"], r["framing_runs"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(L["official_x"])
    ax.set_ylabel(L["official_y"])
    ax.set_title(f"{L['official_title']}\nPearson r = {pr:.3f}, Spearman r = {sr:.3f}")
    ax.legend(loc="upper left")
    save(fig, out, "framing_vs_official_2023")


def fig_variance_components(L, tau, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = ["catcher", "umpire", "pitcher"]
    vals = [tau[k] for k in keys]
    ax.bar(L["tau_groups"], vals, color=["#c0392b", "#2c7fb8", "#7fbf7b"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylabel(L["tau_y"])
    ax.set_title(L["tau_title"])
    save(fig, out, "hier_variance_components_2023")


def fig_shrinkage(L, cmp_df, out):
    x = cmp_df["framing_runs"].to_numpy()
    y = cmp_df["framing_runs_model"].to_numpy()
    n = cmp_df["called_pitches"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    lim = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6, label=L["identity"])
    sc = ax.scatter(x, y, s=10 + (n / n.max()) * 180, c=n, cmap="viridis",
                    alpha=0.8, edgecolor="white", linewidth=0.4)
    fig.colorbar(sc, ax=ax, label=L["shrink_cbar"], shrink=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(L["shrink_x"])
    ax.set_ylabel(L["shrink_y"])
    ax.set_title(L["shrink_title"])
    ax.legend(loc="upper left")
    save(fig, out, "hier_shrinkage_2023")


def fig_yoy(L, merged, out):
    from scipy.stats import pearsonr

    x = merged["rate_2022"].to_numpy()
    y = merged["rate_2023"].to_numpy()
    pr, _ = pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(7, 7))
    lim = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label=L["identity"])
    ax.scatter(x, y, s=32, alpha=0.75, color="#2c7fb8", edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    for r in pl.concat([
        merged.sort("rate_2023", descending=True).head(3),
        merged.sort("rate_2023").head(2),
    ]).unique(subset="mlbam_id").iter_rows(named=True):
        ax.annotate(r["name"], (r["rate_2022"], r["rate_2023"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(L["yoy_x"])
    ax.set_ylabel(L["yoy_y"])
    ax.set_title(f"{L['yoy_title']}  (n={merged.height}, r = {pr:.3f})")
    ax.legend(loc="upper left")
    save(fig, out, "reliability_year_over_year")


def fig_matrix(L, years, R, N, out):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(R, cmap="RdYlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(len(years)), years)
    for i in range(len(years)):
        for j in range(len(years)):
            txt = f"{R[i, j]:.3f}" + ("" if i == j else f"\n(n={N[i, j]})")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title(L["matrix_title"])
    save(fig, out, "persistence_matrix_2021_2023")


def fig_pooled(L, pooled, out):
    top = pooled.head(15).sort("framing_runs")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(top["name"], top["framing_runs"],
            color=["#c0392b" if v > 0 else "#2c7fb8" for v in top["framing_runs"]])
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(L["pooled_x"])
    ax.set_title(L["pooled_title"])
    save(fig, out, "pooled_leaderboard_2021_2023")


def fig_trajectories(L, traj, top_names, bottom_names, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for nm in top_names + bottom_names:
        row = traj.filter(pl.col("name") == nm)
        if row.height == 0:
            continue
        color = "#c0392b" if nm in top_names else "#2c7fb8"
        ax.plot(SEASONS, [row["r21"][0], row["r22"][0], row["r23"][0]],
                "o-", color=color, alpha=0.8, label=nm)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(list(SEASONS))
    ax.set_xlabel(L["traj_x"])
    ax.set_ylabel(L["traj_y"])
    ax.set_title(L["traj_title"])
    ax.legend(fontsize=8, ncol=2)
    save(fig, out, "catcher_trajectories_2021_2023")


# ---------------------------------------------------------------- 主流程

def load_everything():
    """一次載入所有圖需要的資料（兩種語言共用，只畫兩次）。"""
    from data.official import fetch_official_framing
    from models.baseline_gam import fit_or_load, load_modeling_frame
    from models.framing_runs import add_names, framing_leaderboard, merge_official
    from models.hierarchical import load_cached
    from models.reliability import (
        pairwise_year_correlations,
        pooled_leaderboard,
        season_rate_leaderboard,
    )

    seasons = {y: pl.read_parquet(PROCESSED / f"statcast_{y}_baseline.parquet") for y in SEASONS}
    df23 = load_modeling_frame(2023)
    gam = fit_or_load(2023)

    simple = add_names(framing_leaderboard(seasons[2023]))
    official = merge_official(simple, fetch_official_framing(2023))

    _, hier, meta = load_cached(2023)
    shrink = simple.select(["mlbam_id", "called_pitches", "framing_runs"]).join(
        hier.select(["mlbam_id", "framing_runs_model"]), on="mlbam_id", how="inner"
    )

    rates = {
        y: season_rate_leaderboard(d, 1000).select("mlbam_id", pl.col("rate").alias(f"r{str(y)[2:]}"))
        for y, d in seasons.items()
    }
    yoy = add_names(
        rates[2022].join(rates[2023], on="mlbam_id", how="inner")
        .rename({"r22": "rate_2022", "r23": "rate_2023"})
    )
    years, R, N = pairwise_year_correlations(seasons, 1000)
    pooled = add_names(pooled_leaderboard(seasons, min_pitches=3000))

    traj = add_names(
        rates[2021].join(rates[2022], on="mlbam_id", how="inner")
        .join(rates[2023], on="mlbam_id", how="inner")
    )
    in_traj = pooled.filter(pl.col("mlbam_id").is_in(traj["mlbam_id"].to_list()))
    top_names = in_traj.sort("framing_runs", descending=True).head(4)["name"].to_list()
    bottom_names = in_traj.sort("framing_runs").head(3)["name"].to_list()

    return dict(
        df23=df23, gam=gam, official=official, tau=meta["tau"], shrink=shrink,
        yoy=yoy, years=years, R=R, N=N, pooled=pooled, traj=traj,
        top_names=top_names, bottom_names=bottom_names,
    )


def build(lang: str, data: dict) -> None:
    L = LABELS[lang]
    out = ROOT / "docs" / "images" / lang
    use_style(lang)
    print(f"[{lang}]")
    fig_count_contours(L, data["df23"], out)
    fig_count_diff(L, data["df23"], out)
    fig_surface_vs_empirical(L, data["df23"], data["gam"], out)
    fig_official(L, data["official"], out)
    fig_variance_components(L, data["tau"], out)
    fig_shrinkage(L, data["shrink"], out)
    fig_yoy(L, data["yoy"], out)
    fig_matrix(L, data["years"], data["R"], data["N"], out)
    fig_pooled(L, data["pooled"], out)
    fig_trajectories(L, data["traj"], data["top_names"], data["bottom_names"], out)


def main() -> None:
    p = argparse.ArgumentParser(description="重生 README 用圖（中英兩套）")
    p.add_argument("--lang", choices=["en", "zh", "both"], default="both")
    args = p.parse_args()

    print("載入資料與模型…")
    data = load_everything()
    for lang in (["en", "zh"] if args.lang == "both" else [args.lang]):
        build(lang, data)


if __name__ == "__main__":
    main()
