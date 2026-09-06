"""Train pool 的 out-of-fold 基準機率。

shadow zone 是用基準模型的預測定義的，而 framing 的訊號就是 y − p̂。如果 p̂ 是
in-sample 的，殘差會被 fit 自動拉平到零，等於把要測的東西先抹掉一部分。所以
train pool 裡每一顆球的 p̂ 都要來自「沒看過這顆球」的模型。

依場次分 K folds（不是依單顆球，理由同 models/splits）。K 次 fit 各約四分鐘，
結果快取在 models/artifacts/（gitignored），只有第一次慢。

用法
----
    uv run python -m models.crossfit
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from models.baseline_gam import ARTIFACT_DIR, _matrix, fit_baseline
from models.splits import SPLIT_SEED, load_train_pool

K_FOLDS = 5
OOF_PATH = ARTIFACT_DIR / "baseline_oof_train.npy"


def fold_assignment(games: np.ndarray, k: int = K_FOLDS, seed: int = SPLIT_SEED) -> dict:
    """場次 → fold。獨立成函式，確保快取與重算用的是同一組分配。"""
    rng = np.random.default_rng(seed)
    return dict(zip(games, rng.integers(0, k, len(games))))


def compute_oof(pool: pl.DataFrame, k: int = K_FOLDS, verbose: bool = True) -> np.ndarray:
    games = pool["game_pk"].unique().sort().to_numpy()
    fold = pool["game_pk"].replace_strict(fold_assignment(games, k), return_dtype=pl.Int32)
    pool = pool.with_columns(fold=fold)

    oof = np.full(pool.height, np.nan)
    for i in range(k):
        tr = pool.filter(pl.col("fold") != i)
        te_mask = (pool["fold"] == i).to_numpy()
        t0 = time.time()
        gam = fit_baseline(tr, n_splines=20, verbose=False)
        oof[te_mask] = gam.predict_proba(_matrix(pool.filter(pl.col("fold") == i)))
        if verbose:
            print(f"  fold {i}: fit n={tr.height:,} → 預測 n={te_mask.sum():,} ({time.time()-t0:.0f}s)", flush=True)
    assert not np.isnan(oof).any()
    return oof


def load_or_compute(force: bool = False) -> pl.DataFrame:
    """回傳 train pool，多一欄 baseline_prob_oof。"""
    pool = load_train_pool()
    if OOF_PATH.exists() and not force:
        oof = np.load(OOF_PATH)
        if len(oof) != pool.height:
            raise ValueError(f"快取長度 {len(oof)} 對不上 pool {pool.height}，用 force=True 重算")
    else:
        oof = compute_oof(pool)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(OOF_PATH, oof)
        print(f"已存 → {OOF_PATH.name}")
    return pool.with_columns(baseline_prob_oof=pl.Series(oof))


if __name__ == "__main__":
    df = load_or_compute()
    p = df["baseline_prob_oof"]
    print(f"{df.height:,} 列，shadow zone (0.2–0.8) {((p > 0.2) & (p < 0.8)).sum():,} 顆")
