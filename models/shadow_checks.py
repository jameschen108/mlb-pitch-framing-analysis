"""METHODS §2.3 與 §4.1 的兩個檢查，原本是一次性的計算，搬進來才能照重現指令核對。

1. `shadow_information()`：shadow zone 佔多少球、佔多少 Fisher information。捕手效果
   的資訊量正比於 p(1−p)，所以只看球數會高估丟掉的東西。用的是 v1 在 2023 上的
   樣本內基準（`statcast_2023_baseline.parquet`），因為 METHODS 的那張表當初就是
   用它畫的帶狀區。
2. `baseline_calibration()` 與 `isotonic_shift()`：2021–22 out-of-fold 基準在 shadow
   zone 內的校準分箱，以及用 isotonic 重新校準之後，每位捕手的未調整 runs 會移動
   多少。isotonic 擬合在整個 train pool 上（y 對 out-of-fold p̂）。

用法
----
    uv run python -m models.shadow_checks
"""

from __future__ import annotations

import numpy as np
import polars as pl

from models.baseline_gam import ARTIFACT_DIR, DATA_PROCESSED
from models.hierarchical_v2 import SHADOW_HI, SHADOW_LO
from models.intervals import RUN_VALUE

INFO_PATH = ARTIFACT_DIR / "shadow_information.parquet"
CALIB_PATH = ARTIFACT_DIR / "baseline_calibration.parquet"
ISO_PATH = ARTIFACT_DIR / "isotonic_shift.parquet"
MIN_CALLED = 1000          # 算 SE 放大倍數時，捕手至少要有的判定球數
MIN_SHADOW = 300           # isotonic 位移只看 shadow zone ≥300 球的捕手


def shadow_information(force: bool = False) -> pl.DataFrame:
    """2023、v1 基準：shadow zone 的球數佔比、資訊佔比，以及每位捕手 SE 放大多少。

    SE ∝ 1/√I，所以只用 shadow zone 時，捕手效果的標準誤放大 √(I_全部 / I_shadow)。
    `se_factor_from_counts` 是只看球數會得到的倍數，拿來對照。
    """
    if INFO_PATH.exists() and not force:
        return pl.read_parquet(INFO_PATH)
    df = pl.read_parquet(DATA_PROCESSED / "statcast_2023_baseline.parquet")
    p = df["baseline_strike_prob"].to_numpy()
    w, sh = p * (1 - p), (p > SHADOW_LO) & (p < SHADOW_HI)
    t = (df.with_columns(w=pl.Series(w), sh=pl.Series(sh))
         .group_by("fielder_2")
         .agg(n=pl.len(), ns=pl.col("sh").sum(),
              info=pl.col("w").sum(), info_s=(pl.col("w") * pl.col("sh")).sum())
         .filter(pl.col("n") >= MIN_CALLED))
    out = pl.DataFrame([{
        "season": 2023, "baseline": "v1_in_sample", "n_called": len(p),
        "shadow_share_pitches": float(sh.mean()),
        "shadow_share_information": float(w[sh].sum() / w.sum()),
        "min_called": MIN_CALLED, "n_catchers": t.height,
        "se_factor_mean": float(np.sqrt(t["info"] / t["info_s"]).mean()),
        "se_factor_from_counts": float(np.sqrt(t["n"] / t["ns"]).mean()),
    }])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(INFO_PATH)
    return out


def _oof_pool():
    from models.crossfit import load_or_compute
    pool = load_or_compute()
    return (pool, pool["baseline_prob_oof"].to_numpy(), pool["is_strike"].to_numpy().astype(float))


def baseline_calibration(force: bool = False) -> pl.DataFrame:
    """2021–22 out-of-fold 基準在 shadow zone 內的校準，每 0.1 一箱。"""
    if CALIB_PATH.exists() and not force:
        return pl.read_parquet(CALIB_PATH)
    _, p, y = _oof_pool()
    edges = np.round(np.arange(SHADOW_LO, SHADOW_HI + 1e-9, 0.1), 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
                     "predicted": float(p[m].mean()), "actual": float(y[m].mean()),
                     "predicted_minus_actual": float(p[m].mean() - y[m].mean())})
    out = pl.DataFrame(rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(CALIB_PATH)
    return out


def isotonic_shift(force: bool = False) -> pl.DataFrame:
    """isotonic 重新校準之後，每位捕手的未調整 shadow-zone runs 移動多少。

    和 runs 本身的全距（最高減最低）並排：位移的全距小到跟它不成比例，就表示
    S 形校準偏差對每位捕手的影響幾乎一樣，不會改變排名。
    """
    if ISO_PATH.exists() and not force:
        return pl.read_parquet(ISO_PATH)
    from sklearn.isotonic import IsotonicRegression

    pool, p, y = _oof_pool()
    p_iso = IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(p)
    sh = (p > SHADOW_LO) & (p < SHADOW_HI)
    t = (pl.DataFrame({"catcher": pool["fielder_2"].to_numpy()[sh],
                       "r": (y - p)[sh], "r_iso": (y - p_iso)[sh]})
         .group_by("catcher")
         .agg(n=pl.len(), runs=pl.col("r").sum() * RUN_VALUE, runs_iso=pl.col("r_iso").sum() * RUN_VALUE)
         .filter(pl.col("n") >= MIN_SHADOW))
    d = (t["runs_iso"] - t["runs"]).to_numpy()
    out = pl.DataFrame([{
        "min_shadow": MIN_SHADOW, "n_catchers": t.height,
        "shift_mean": float(d.mean()), "shift_sd": float(d.std()),
        "shift_range": float(d.max() - d.min()),
        "runs_range": float(t["runs"].max() - t["runs"].min()),
    }])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(ISO_PATH)
    return out


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
    print(shadow_information(), baseline_calibration(), isotonic_shift(), sep="\n")
