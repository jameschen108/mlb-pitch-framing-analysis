"""第一層：好球判定基準模型（GAM）。

建模 P(判好球 | 位置與情境)，**不含捕手身分**——這是 framing 的反事實基準：
「這顆球換一個平均捕手來接，會被判好球的機率」。

模型
----
主力基準模型（供後續 framing runs 使用）：

    logit P(strike) = te(plate_x, plate_z_std) + f(stand) + f(p_throws) + f(balls) + f(strikes)

- `te(...)`：plate_x 與標準化 plate_z 的二維張量光滑面，捕捉好球帶形狀。
- `f(...)`：打者側、投手側、球數的 factor 項（加性位移）。

**取捨**：球數對好球帶「形狀」的影響（而非只是整體高低）是位置×球數的交互，
pyGAM 的 categorical-by 交互支援有限。此處基準模型用加性球數項；
球數改變好球帶形狀的現象，改由 `fit_by_count()`（各球數分別 fit tensor 面）
的 50% 等高線診斷來呈現，也連到後續更完整的階層模型動機。

快取
----
全量 fit 約兩分鐘，故 fit 後 pickle 到 models/artifacts/（gitignored），
notebook 以 fit_or_load() 載入，避免每次重跑。

用法
----
    uv run python -m models.baseline_gam    # fit 全量、存快取、印指標
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
from pygam import LogisticGAM, f, te

MODELS_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = MODELS_DIR / "artifacts"
DATA_PROCESSED = MODELS_DIR.parent / "data" / "processed"

# 基準模型的特徵順序（餵給 pyGAM 的欄位順序）
FEATURES = ["plate_x", "plate_z_std", "stand_i", "p_throws_i", "balls", "strikes"]

# 剔除極端座標，避免 GAM 邊界外插不穩（企劃書風險備案）
X_ABS_MAX = 2.5
Z_STD_MIN, Z_STD_MAX = -1.0, 2.0


def load_modeling_frame(season: int = 2023) -> pl.DataFrame:
    """載入 processed 季檔，剔除極端座標，加上編碼欄位。"""
    path = DATA_PROCESSED / f"statcast_{season}.parquet"
    df = pl.read_parquet(path)
    df = df.filter(
        (pl.col("plate_x").abs() <= X_ABS_MAX)
        & (pl.col("plate_z_std") >= Z_STD_MIN)
        & (pl.col("plate_z_std") <= Z_STD_MAX)
    )
    df = df.with_columns(
        stand_i=(pl.col("stand") == "R").cast(pl.Int64),  # R=1, L=0
        p_throws_i=(pl.col("p_throws") == "R").cast(pl.Int64),
    )
    return df


def _matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(FEATURES).to_numpy()


def fit_baseline(df: pl.DataFrame, n_splines: int = 20, verbose: bool = True) -> LogisticGAM:
    """Fit 主力基準模型。"""
    X = _matrix(df)
    y = df["is_strike"].to_numpy()
    terms = te(0, 1, n_splines=[n_splines, n_splines]) + f(2) + f(3) + f(4) + f(5)
    t0 = time.time()
    gam = LogisticGAM(terms).fit(X, y)
    if verbose:
        print(f"基準模型 fit 完成：{time.time() - t0:.0f}s，n={len(y):,}")
    return gam


def model_path(season: int) -> Path:
    return ARTIFACT_DIR / f"baseline_gam_{season}.pkl"


def fit_or_load(season: int = 2023, force: bool = False, n_splines: int = 20) -> LogisticGAM:
    """載入快取模型；不存在或 force 時重新 fit 並存檔。"""
    path = model_path(season)
    if path.exists() and not force:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    df = load_modeling_frame(season)
    gam = fit_baseline(df, n_splines=n_splines)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(gam, fh)
    print(f"已存模型 → {path}")
    return gam


def fit_by_count(
    df: pl.DataFrame,
    counts: list[tuple[int, int]],
    n_splines: int = 12,
) -> dict[tuple[int, int], LogisticGAM]:
    """對指定球數各自 fit 一個純 tensor 好球帶面（供 50% 等高線診斷）。"""
    out: dict[tuple[int, int], LogisticGAM] = {}
    for balls, strikes in counts:
        sub = df.filter((pl.col("balls") == balls) & (pl.col("strikes") == strikes))
        X = sub.select(["plate_x", "plate_z_std"]).to_numpy()
        y = sub["is_strike"].to_numpy()
        out[(balls, strikes)] = LogisticGAM(
            te(0, 1, n_splines=[n_splines, n_splines])
        ).fit(X, y)
    return out


def predict_grid(
    gam: LogisticGAM,
    x_range=(-2.0, 2.0),
    z_range=(-0.5, 1.5),
    n: int = 120,
    context: dict | None = None,
    tensor_only: bool = False,
):
    """在位置網格上預測好球機率，回傳 (XX, ZZ, P)。

    context: 給主力模型固定的情境值（stand_i, p_throws_i, balls, strikes）。
    tensor_only: 若模型只有 te(0,1) 兩個特徵（fit_by_count 產物）設 True。
    """
    xs = np.linspace(*x_range, n)
    zs = np.linspace(*z_range, n)
    XX, ZZ = np.meshgrid(xs, zs)
    flat = np.column_stack([XX.ravel(), ZZ.ravel()])

    if not tensor_only:
        ctx = {"stand_i": 1, "p_throws_i": 1, "balls": 0, "strikes": 0}
        if context:
            ctx.update(context)
        n_pts = flat.shape[0]
        cols = [
            flat[:, 0],
            flat[:, 1],
            np.full(n_pts, ctx["stand_i"]),
            np.full(n_pts, ctx["p_throws_i"]),
            np.full(n_pts, ctx["balls"]),
            np.full(n_pts, ctx["strikes"]),
        ]
        flat = np.column_stack(cols)

    P = gam.predict_proba(flat).reshape(XX.shape)
    return XX, ZZ, P


def add_baseline_prob(df: pl.DataFrame, gam: LogisticGAM) -> pl.DataFrame:
    """把基準模型預測機率加回資料（供第二層 framing runs）。"""
    p = gam.predict_proba(_matrix(df))
    return df.with_columns(baseline_strike_prob=pl.Series(p))


def build_baseline_parquet(season: int = 2023, force: bool = False) -> pl.DataFrame:
    """對某季 fit（或載入）基準 GAM，輸出含 baseline_strike_prob 的 parquet。

    等同 notebooks/02 的輸出步驟，抽成函式方便跨季重用（如年度間相關要的 2022）。
    """
    out = DATA_PROCESSED / f"statcast_{season}_baseline.parquet"
    if out.exists() and not force:
        return pl.read_parquet(out)
    df = load_modeling_frame(season)
    gam = fit_or_load(season)
    df = add_baseline_prob(df, gam)
    df.write_parquet(out)
    print(f"已輸出基準機率 → {out.name}（{df.height:,} 列）")
    return df


if __name__ == "__main__":
    from sklearn.metrics import log_loss, roc_auc_score

    df = load_modeling_frame(2023)
    gam = fit_or_load(2023, force=True)
    p = gam.predict_proba(_matrix(df))
    y = df["is_strike"].to_numpy()
    print(f"acc={gam.accuracy(_matrix(df), y):.4f}  "
          f"auc={roc_auc_score(y, p):.4f}  logloss={log_loss(y, p):.4f}")
