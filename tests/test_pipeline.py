"""管線關鍵不變量的快速單元測試（不需下載資料）。

    uv run pytest -q
"""

import numpy as np
import polars as pl

from data.fetch import clean
from models.reliability import spearman_brown, _catcher_rate
from models.hierarchical import _logit


def _synthetic_raw() -> pl.DataFrame:
    """模擬 Statcast 原始欄位（含春訓列與非判定球）。"""
    return pl.DataFrame(
        {
            "game_date": ["2023-04-01"] * 5,
            "game_type": ["R", "R", "S", "R", "R"],  # 一列春訓
            "game_pk": [1, 1, 2, 1, 1],
            "pitcher": [10, 10, 10, 11, 11],
            "batter": [20, 21, 22, 23, 24],
            "fielder_2": [30, 30, 30, 31, 31],
            "stand": ["R", "L", "R", "R", "L"],
            "p_throws": ["R", "R", "R", "L", "L"],
            "balls": [0, 1, 0, 2, 3],
            "strikes": [0, 2, 1, 1, 0],
            "plate_x": [0.0, 0.5, 0.1, -0.3, 2.0],
            "plate_z": [2.5, 1.6, 3.0, 2.0, 2.2],
            "sz_top": [3.5, 3.5, 3.4, 3.3, 3.5],
            "sz_bot": [1.5, 1.5, 1.6, 1.5, 1.5],
            # 第 3 列 foul（非判定），其餘 called
            "description": ["called_strike", "ball", "foul", "ball", "called_strike"],
        }
    )


def test_clean_keeps_only_regular_season_called_pitches():
    out = clean(_synthetic_raw())
    # 春訓(S)與 foul 都被剔除 → 剩 4 顆例行賽判定球
    assert out.height == 4
    assert set(out["game_type"].unique().to_list()) == {"R"}
    assert set(out["description"].unique().to_list()) <= {"called_strike", "ball"}


def test_standardization_formula():
    out = clean(_synthetic_raw())
    # plate_z_std = (plate_z - sz_bot) / (sz_top - sz_bot)
    row = out.filter((pl.col("plate_x") == 0.0)).row(0, named=True)
    expected = (2.5 - 1.5) / (3.5 - 1.5)
    assert abs(row["plate_z_std"] - expected) < 1e-9
    # 好球帶正中央 (z_std=0.5) 應落在 [0,1] 內
    assert 0.0 <= row["plate_z_std"] <= 1.0


def test_is_strike_label():
    out = clean(_synthetic_raw())
    strikes = out.filter(pl.col("description") == "called_strike")["is_strike"].to_list()
    balls = out.filter(pl.col("description") == "ball")["is_strike"].to_list()
    assert all(s == 1 for s in strikes)
    assert all(b == 0 for b in balls)


def test_logit_is_bounded_and_monotonic():
    p = np.array([0.0, 0.01, 0.5, 0.99, 1.0])
    z = _logit(p)
    assert np.all(np.isfinite(z))  # 邊界 0/1 被 clip，不應 inf
    assert np.all(np.diff(z) > 0)  # 單調遞增
    assert abs(z[2]) < 1e-9  # logit(0.5) == 0


def test_spearman_brown_increases_reliability():
    # 半量相關 r，全量信度 R = 2r/(1+r) 應 >= r（0<r<1）
    for r in [0.3, 0.5, 0.8]:
        R = spearman_brown(r)
        assert R >= r
        assert R <= 1.0


def test_catcher_rate_matches_manual_framing():
    # 一位捕手 2 顆球：實際 [1,0]，基準 [0.5,0.5] → extra=0，rate=0
    df = pl.DataFrame(
        {"fielder_2": [30, 30], "is_strike": [1, 0], "baseline_strike_prob": [0.5, 0.5]}
    )
    r = _catcher_rate(df, group_extra="")
    assert abs(r["extra"][0]) < 1e-9
    assert abs(r["rate"][0]) < 1e-9
