"""v2 的資料切分。

v1 把整季資料同時拿來 fit 和評估，所以 README 報的 AUC 0.98 與 log loss 都是
in-sample。v2 把切分固定下來：

    2021–2022  train pool  ──┬── 80% train（fit 用）
                             └── 20% val（選模用）
    2023       holdout       只用一次，留給最後報結果

**依 game_pk 切，不依單顆球切。** 同一場比賽的球共享主審、天氣、球場與當天的
好球帶鬆緊；隨機切單顆球會讓同一場的球同時出現在 train 和 val，val 的 log loss
會偏樂觀。切在場次上比較貴（val 的比例沒辦法剛好 20%），但那個數字才能用。

種子固定在 SPLIT_SEED，任何時候重跑都要拿到同一組切分。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from models.baseline_gam import load_modeling_frame

TRAIN_SEASONS = (2021, 2022)
HOLDOUT_SEASON = 2023
VAL_FRACTION = 0.2
SPLIT_SEED = 20260906


def load_train_pool(seasons=TRAIN_SEASONS) -> pl.DataFrame:
    """載入 train pool（預設 2021+2022），欄位與 v1 的 modeling frame 相同。"""
    frames = [load_modeling_frame(s).with_columns(season=pl.lit(s, pl.Int32)) for s in seasons]
    return pl.concat(frames, how="vertical")


def assign_split(
    df: pl.DataFrame,
    val_fraction: float = VAL_FRACTION,
    seed: int = SPLIT_SEED,
) -> pl.DataFrame:
    """加一欄 split ∈ {train, val}，以 game_pk 為單位隨機分配。"""
    games = df["game_pk"].unique().sort().to_numpy()
    rng = np.random.default_rng(seed)
    val_games = rng.choice(games, size=int(round(len(games) * val_fraction)), replace=False)
    return df.with_columns(
        split=pl.when(pl.col("game_pk").is_in(pl.Series(val_games)))
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
    )


def train_val(df: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """回傳 (train, val)。df 為 None 時自己載入 train pool。"""
    if df is None:
        df = load_train_pool()
    df = assign_split(df)
    return df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "val")


def load_holdout() -> pl.DataFrame:
    """載入 2023 holdout。

    這份只能用一次，在所有選模決定都定案之後。每次呼叫都在 NOTES.md 記一筆，
    寫清楚為什麼碰它——事後才發現用了兩次，比事前知道要糟得多。
    """
    return load_modeling_frame(HOLDOUT_SEASON).with_columns(
        season=pl.lit(HOLDOUT_SEASON, pl.Int32)
    )


def describe(df: pl.DataFrame) -> str:
    """一行摘要，方便印在 log 和 NOTES 裡。"""
    return f"{df.height:,} 列 / {df['game_pk'].n_unique():,} 場 / strike rate {df['is_strike'].mean():.4f}"
