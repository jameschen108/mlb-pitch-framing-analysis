"""v2 的基準模型：和 v1 同一個模型形式，但 fit 與評估分開。

v1 在整季資料上 fit，然後在同一批資料上報 AUC 0.98 —— 那是 in-sample。這裡沿用
完全相同的模型形式（te(plate_x, plate_z_std) + f(stand) + f(p_throws) + f(balls)
+ f(strikes)），只換掉評估方式，所以兩者的差就是「in-sample 樂觀了多少」，沒有
被模型變動污染。

指標只看 log loss，不看 accuracy 也不看 AUC。AUC 只在乎排序，對「模型說 70% 的
那批球是不是真的 70%」完全沒有資訊量，而 framing 的整套算法建立在機率值本身
（殘差 = 實際 − 預測）上，校準錯了整條線都錯。

用法
----
    uv run python -m models.baseline_v2
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl

from models.baseline_gam import ARTIFACT_DIR, _matrix, fit_baseline
from models.splits import describe, train_val

MODEL_PATH = ARTIFACT_DIR / "baseline_v2_train.pkl"


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(gam, df: pl.DataFrame) -> dict:
    """log loss、Brier，以及只猜基礎率的 log loss 當參考點。"""
    y = df["is_strike"].to_numpy().astype(float)
    p = gam.predict_proba(_matrix(df))
    base = float(y.mean())
    return {
        "n": df.height,
        "log_loss": _log_loss(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss_base_rate": _log_loss(y, np.full_like(y, base)),
    }


def calibration(gam, df: pl.DataFrame, bins: int = 10) -> pl.DataFrame:
    """等寬分箱的校準表：模型說幾成、實際幾成。"""
    p = gam.predict_proba(_matrix(df))
    out = pl.DataFrame({"p": p, "y": df["is_strike"].cast(pl.Float64)})
    return (
        out.with_columns(bin=(pl.col("p") * bins).floor().clip(0, bins - 1).cast(pl.Int32))
        .group_by("bin")
        .agg(n=pl.len(), predicted=pl.col("p").mean(), actual=pl.col("y").mean())
        .sort("bin")
    )


def fit_or_load(force: bool = False, n_splines: int = 20):
    """在 train 上 fit（或載入快取）。**只吃 train，不碰 val、不碰 holdout。**"""
    if MODEL_PATH.exists() and not force:
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh)
    tr, _ = train_val()
    print(f"fit on train: {describe(tr)}")
    t0 = time.time()
    gam = fit_baseline(tr, n_splines=n_splines, verbose=False)
    print(f"fit 完成：{time.time() - t0:.0f}s")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(gam, fh)
    return gam


if __name__ == "__main__":
    tr, va = train_val()
    gam = fit_or_load()

    print()
    for name, d in (("train (in-sample)", tr), ("val   (out-of-sample)", va)):
        m = evaluate(gam, d)
        print(f"{name}: n={m['n']:>8,}  log loss={m['log_loss']:.5f}  "
              f"brier={m['brier']:.5f}  (只猜基礎率={m['log_loss_base_rate']:.5f})")

    print("\nval 校準表：")
    print(calibration(gam, va).to_pandas().round(4).to_string(index=False))
