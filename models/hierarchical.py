"""第二層（核心版）：階層 logistic 模型，分離捕手真實技術與運氣。

兩階段法（單季 35 萬列、三季 104 萬列，跑完整貝葉斯不實際）：
1. 凍結第一層 GAM 的預測 logit 當 offset/covariate。
2. 對殘差建交叉隨機截距模型，同時放捕手/主審/投手三組效應互相控制：

    logit P(strike) = β0 + β1·logit_base + θ_catcher + φ_umpire + ψ_pitcher
    θ_catcher ~ N(0, τ_c²),  φ_umpire ~ N(0, τ_u²),  ψ_pitcher ~ N(0, τ_p²)

引擎：statsmodels BinomialBayesMixedGLM（變分貝葉斯）。θ_catcher 的後驗均值即
**收縮後**的捕手效應——樣本少的替補捕手被拉向 0。

捕手 framing runs（控制主審/投手後）用「每球邊際貢獻」換算：
對該捕手接的每顆球，比較含 θ_catcher 與不含 θ_catcher 的預測機率差，加總 × RUN_VALUE。

計算取捨：投手 <MIN_PITCHER_PITCHES 顆併成一組（低樣本投手效應本就收縮到 0），
把 level 數壓下來。即使如此，VB 仍會跑到預設迭代上限而未完全收斂——單季數分鐘、
三季合併約一小時。固定效應與 τ 在不同子樣本/種子下穩定，合成資料也能正確還原
（見 tests/test_random_effects.py），但更紮實的做法是改用 R glmmTMB 的 MLE。

注意：variance_components() 依賴 statsmodels 的 vcp_mean 依 vc dict key 順序排列，
extract_effects() 則靠正則解析 vc_names。兩者都無明文 API 保證，故有測試守著。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from data.umpires import attach_umpires

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

RUN_VALUE = 0.125
MIN_PITCHER_PITCHES = 100  # 投手池化門檻
POOLED_PITCHER_ID = -1

_VC_RE = re.compile(r"C\((\w+)\)\[(-?\d+)\]")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_frame(season: int = 2023) -> pl.DataFrame:
    """基準資料 + 主審，計算 logit_base、池化投手。"""
    df = pl.read_parquet(DATA_PROCESSED / f"statcast_{season}_baseline.parquet")
    df = attach_umpires(df, season).drop_nulls("umpire_id")
    df = df.with_columns(logit_base=pl.Series(_logit(df["baseline_strike_prob"].to_numpy())))
    keep = set(
        df.group_by("pitcher").agg(pl.len().alias("n")).filter(pl.col("n") >= MIN_PITCHER_PITCHES)["pitcher"].to_list()
    )
    df = df.with_columns(
        pitcher_g=pl.when(pl.col("pitcher").is_in(list(keep)))
        .then(pl.col("pitcher"))
        .otherwise(POOLED_PITCHER_ID)
    )
    return df


def load_frame_multi(seasons=(2021, 2022, 2023), min_pitcher_pitches: int = 150) -> pl.DataFrame:
    """三季合併：各季自己的基準 + 主審堆疊，投手依跨季總數池化。

    每季已各自 fit 基準 GAM，logit_base 已吸收各季好球帶差異，故不需季固定效應；
    捕手/主審/投手效應跨三季共享，同一捕手三季得到單一 pooled 估計（最穩）。
    """
    frames = []
    for s in seasons:
        df = pl.read_parquet(DATA_PROCESSED / f"statcast_{s}_baseline.parquet")
        df = attach_umpires(df, s).drop_nulls("umpire_id")
        df = df.with_columns(
            logit_base=pl.Series(_logit(df["baseline_strike_prob"].to_numpy())),
            season=pl.lit(s),
        )
        frames.append(df)
    df = pl.concat(frames, how="diagonal_relaxed")
    keep = set(
        df.group_by("pitcher").agg(pl.len().alias("n")).filter(pl.col("n") >= min_pitcher_pitches)["pitcher"].to_list()
    )
    df = df.with_columns(
        pitcher_g=pl.when(pl.col("pitcher").is_in(list(keep)))
        .then(pl.col("pitcher"))
        .otherwise(POOLED_PITCHER_ID)
    )
    return df


def _to_pandas(df: pl.DataFrame):
    pdf = df.select(
        ["is_strike", "logit_base", "fielder_2", "umpire_id", "pitcher_g"]
    ).to_pandas()
    pdf["catcher"] = pdf["fielder_2"].astype(str)
    pdf["umpire"] = pdf["umpire_id"].astype("int64").astype(str)
    pdf["pitcher_s"] = pdf["pitcher_g"].astype("int64").astype(str)
    return pdf


def fit(df: pl.DataFrame):
    """Fit 交叉隨機截距模型，回傳 (model, result)。"""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    pdf = _to_pandas(df)
    vc = {
        "catcher": "0 + C(catcher)",
        "umpire": "0 + C(umpire)",
        "pitcher": "0 + C(pitcher_s)",
    }
    model = BinomialBayesMixedGLM.from_formula("is_strike ~ logit_base", vc, pdf)
    result = model.fit_vb()
    return model, result


def extract_effects(model, result) -> pl.DataFrame:
    """把 vc_mean/vc_sd 對應回 [group, mlbam_id, effect_mean, effect_sd]。"""
    rows = []
    for name, mean, sd in zip(model.vc_names, result.vc_mean, result.vc_sd):
        m = _VC_RE.match(name)
        if not m:
            continue
        group = {"catcher": "catcher", "umpire": "umpire", "pitcher_s": "pitcher"}[m.group(1)]
        rows.append({"group": group, "mlbam_id": int(m.group(2)), "effect_mean": float(mean), "effect_sd": float(sd)})
    return pl.DataFrame(rows)


def variance_components(model, result) -> dict:
    """回傳各組隨機效應標準差 τ（vcp 為 log-sd 的後驗均值，依 vc dict key 順序）。"""
    names = ["catcher", "umpire", "pitcher"]
    return {names[i]: float(np.exp(m)) for i, m in enumerate(result.vcp_mean)}


def framing_runs(df: pl.DataFrame, effects: pl.DataFrame, result, run_value: float = RUN_VALUE) -> pl.DataFrame:
    """用每球邊際捕手貢獻換算 framing runs（控制主審/投手/位置）。"""
    intercept, slope = float(result.fe_mean[0]), float(result.fe_mean[1])

    def eff(group):
        return effects.filter(pl.col("group") == group).select(
            pl.col("mlbam_id"), pl.col("effect_mean").alias(f"{group}_eff")
        )

    d = (
        df.join(eff("catcher"), left_on="fielder_2", right_on="mlbam_id", how="left")
        .join(eff("umpire"), left_on="umpire_id", right_on="mlbam_id", how="left")
        .join(eff("pitcher"), left_on="pitcher_g", right_on="mlbam_id", how="left")
        .with_columns([pl.col(c).fill_null(0.0) for c in ["catcher_eff", "umpire_eff", "pitcher_eff"]])
    )
    eta = (
        intercept
        + slope * pl.col("logit_base")
        + pl.col("catcher_eff")
        + pl.col("umpire_eff")
        + pl.col("pitcher_eff")
    )
    inv = lambda x: 1.0 / (1.0 + (-x).exp())  # noqa: E731
    d = d.with_columns(
        delta_p=(inv(eta) - inv(eta - pl.col("catcher_eff"))),
    )
    lb = (
        d.group_by("fielder_2")
        .agg(
            called_pitches=pl.len(),
            extra_strikes_model=pl.col("delta_p").sum(),
            catcher_effect=pl.col("catcher_eff").first(),
        )
        .with_columns(framing_runs_model=pl.col("extra_strikes_model") * run_value)
        .rename({"fielder_2": "mlbam_id"})
        .sort("framing_runs_model", descending=True)
    )
    return lb


# ---- 快取：一次 fit，存衍生結果供 notebook 載入 ----

def _paths(key):
    return (
        ARTIFACT_DIR / f"hierarchical_effects_{key}.parquet",
        ARTIFACT_DIR / f"hierarchical_framing_{key}.parquet",
        ARTIFACT_DIR / f"hierarchical_meta_{key}.json",
    )


def _fit_cache(df: pl.DataFrame, key) -> tuple:
    print(f"[{key}] fit 中… n={df.height:,}，catcher/umpire/pitcher levels="
          f"{df['fielder_2'].n_unique()}/{df['umpire_id'].n_unique()}/{df['pitcher_g'].n_unique()}", flush=True)
    model, result = fit(df)
    effects = extract_effects(model, result)
    lb = framing_runs(df, effects, result)
    tau = variance_components(model, result)
    meta = {
        "intercept": float(result.fe_mean[0]),
        "slope": float(result.fe_mean[1]),
        "tau": tau,
        "n": df.height,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ep, fp, mp = _paths(key)
    effects.write_parquet(ep)
    lb.write_parquet(fp)
    mp.write_text(json.dumps(meta, indent=2))
    print("fe (intercept, slope):", round(meta["intercept"], 3), round(meta["slope"], 3))
    print("τ (random-effect sd):", {k: round(v, 3) for k, v in tau.items()})
    print(f"已快取 → {ep.name}, {fp.name}, {mp.name}")
    return effects, lb, meta


def fit_and_cache(season: int = 2023):
    return _fit_cache(load_frame(season), season)


def fit_and_cache_multi(seasons=(2021, 2022, 2023)):
    key = f"{min(seasons)}_{max(seasons)}"
    return _fit_cache(load_frame_multi(seasons), key)


def load_cached(key=2023):
    ep, fp, mp = _paths(key)
    if not fp.exists():
        return fit_and_cache(key) if isinstance(key, int) else fit_and_cache_multi()
    return pl.read_parquet(ep), pl.read_parquet(fp), json.loads(mp.read_text())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        fit_and_cache_multi()
    else:
        fit_and_cache(2023)
