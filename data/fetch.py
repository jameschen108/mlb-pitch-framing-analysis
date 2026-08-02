"""Statcast 逐球資料抓取與清理管線（第一階段：2023 單季開發用）。

設計重點
--------
- **按月迴圈**抓取，避開 pybaseball.statcast() 一次抓太長的逾時問題。
- **重試機制**：每月抓取失敗時指數退避重試。
- **快照**：每月原始資料存 data/raw/，已抓過的月份重跑時直接略過，可安全中斷續跑。
- **清理**：只留主審判定球（called_strike / ball），標準化 plate_z，選欄位。
- 清理後合併成單一季檔存 data/processed/。

用法
----
    uv run python -m data.fetch            # 抓 2023 整季
    uv run python -m data.fetch --season 2023 --start-month 4 --end-month 4  # 只抓 4 月（快速測試）
"""

from __future__ import annotations

import argparse
import calendar
import time
from pathlib import Path

import polars as pl

# pybaseball 抓取回傳 pandas，這裡延後 import 以加快非抓取路徑

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 只保留主審實際判定的球（framing 樣本），排除揮棒、界外、擊球等
CALLED_DESCRIPTIONS = ["called_strike", "ball"]

# 只留例行賽：排除春訓(S)、季後賽(D/F/L/W)、明星賽(A)等。
# 春訓用小聯盟主審、分隊陣容、且部分球場測試 ABS；官方 framing 榜單也是純例行賽。
REGULAR_SEASON = "R"

# 清理後保留的欄位：建模需要的位置/情境/身分，加上少數 ID 與日期
KEEP_COLUMNS = [
    "game_date",
    "game_type",
    "game_pk",
    "pitcher",
    "batter",
    "fielder_2",  # 捕手
    "stand",  # 打者左右打
    "p_throws",  # 投手左右投
    "balls",
    "strikes",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "description",
]


def month_bounds(season: int, month: int) -> tuple[str, str]:
    """回傳該月第一天與最後一天的 YYYY-MM-DD 字串。"""
    last_day = calendar.monthrange(season, month)[1]
    return f"{season}-{month:02d}-01", f"{season}-{month:02d}-{last_day:02d}"


def fetch_month(
    season: int,
    month: int,
    max_retries: int = 3,
    backoff: float = 5.0,
):
    """抓單月 Statcast 原始資料（pandas DataFrame），附指數退避重試。"""
    from pybaseball import statcast

    start, end = month_bounds(season, month)
    for attempt in range(1, max_retries + 1):
        try:
            df = statcast(start_dt=start, end_dt=end, verbose=False)
            return df
        except Exception as exc:  # noqa: BLE001 - 抓取端各種網路/解析錯誤都想重試
            if attempt == max_retries:
                raise
            wait = backoff * attempt
            print(f"  [{start}~{end}] 第 {attempt} 次失敗：{exc!r}，{wait:.0f}s 後重試")
            time.sleep(wait)


def raw_month_path(season: int, month: int) -> Path:
    return RAW_DIR / f"statcast_{season}_{month:02d}.parquet"


def get_or_fetch_month(season: int, month: int, force: bool = False) -> pl.DataFrame:
    """取得單月原始資料：若快照存在則讀取，否則抓取後存快照。"""
    path = raw_month_path(season, month)
    if path.exists() and not force:
        print(f"  [{season}-{month:02d}] 使用既有快照 {path.name}")
        return pl.read_parquet(path)

    print(f"  [{season}-{month:02d}] 抓取中…")
    pdf = fetch_month(season, month)
    if pdf is None or len(pdf) == 0:
        print(f"  [{season}-{month:02d}] 無資料（可能是季外月份），跳過")
        return pl.DataFrame()

    df = pl.from_pandas(pdf)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"  [{season}-{month:02d}] 已存 {path.name}（{len(df):,} 列原始）")
    return df


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """過濾主審判定球、標準化 plate_z、選欄位。"""
    if df.is_empty():
        return df

    # 只留清單內欄位（原始資料有近百欄）
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df.select(cols)

    # 只留例行賽的主審判定球
    if "game_type" in df.columns:
        df = df.filter(pl.col("game_type") == REGULAR_SEASON)
    df = df.filter(pl.col("description").is_in(CALLED_DESCRIPTIONS))

    # 位置/好球帶欄位缺失無法建模，剔除
    df = df.drop_nulls(["plate_x", "plate_z", "sz_top", "sz_bot"])

    # 標準化 plate_z：0 = 好球帶下緣、1 = 上緣，控制打者身高差異
    df = df.with_columns(
        plate_z_std=(pl.col("plate_z") - pl.col("sz_bot"))
        / (pl.col("sz_top") - pl.col("sz_bot"))
    )

    # is_strike 標籤方便後續建模與 EDA
    df = df.with_columns(
        is_strike=(pl.col("description") == "called_strike").cast(pl.Int8)
    )

    return df


def build_season(
    season: int = 2023,
    start_month: int = 3,
    end_month: int = 10,
    force: bool = False,
) -> pl.DataFrame:
    """抓整季、清理、合併，存 processed parquet 並回傳。"""
    print(f"=== 建立 {season} 球季資料（{start_month}~{end_month} 月）===")
    frames: list[pl.DataFrame] = []
    for month in range(start_month, end_month + 1):
        raw = get_or_fetch_month(season, month, force=force)
        cleaned = clean(raw)
        if not cleaned.is_empty():
            frames.append(cleaned)
            print(f"  [{season}-{month:02d}] 清理後保留 {len(cleaned):,} 顆判定球")

    if not frames:
        raise RuntimeError("沒有抓到任何資料")

    season_df = pl.concat(frames, how="vertical")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"statcast_{season}.parquet"
    season_df.write_parquet(out)
    print(f"=== 完成：{len(season_df):,} 顆判定球 → {out} ===")
    return season_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="抓取並清理單季 Statcast 資料")
    p.add_argument("--season", type=int, default=2023)
    p.add_argument("--start-month", type=int, default=3)
    p.add_argument("--end-month", type=int, default=10)
    p.add_argument("--force", action="store_true", help="忽略既有快照重新抓取")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_season(
        season=args.season,
        start_month=args.start_month,
        end_month=args.end_month,
        force=args.force,
    )
