"""打者身高，用來把每一季都換成 ABS 的好球帶定義。

2026 起 Statcast 的 `sz_top` / `sz_bot` 改成身高的 53.5% / 27%（ABS 的好球帶），
之前是逐球量的打擊姿勢。兩種定義下的 `plate_z_std` 不是同一個東西，要跨季比較，
每一季都得用同一把尺。只能往身高制統一：2026 已經沒有姿勢制的上下緣了。

身高來源
--------
- 2026 有上場的打者：`sz_top / 0.535`。這是 ABS 用的實測身高，同一位打者整季
  固定（差異只在四捨五入的 0.0005 英尺以內）。
- 其餘打者：MLB Stats API 的登錄身高。實測與登錄通常差不到半英寸。

快取在 data/raw/batter_heights.parquet，欄位 batter、height_ft、source。

用法
----
    uv run python -m data.heights
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HEIGHTS_PATH = RAW_DIR / "batter_heights.parquet"

ABS_TOP, ABS_BOT = 0.535, 0.27          # 好球帶上下緣佔身高的比例
ABS_SEASON = 2026
SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)

_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"


def _parse_height(s: str) -> float:
    """'6\\' 2"' → 6.1667 英尺。"""
    ft, inch = s.replace('"', "").split("' ")
    return int(ft) + int(inch) / 12


def _listed_heights(ids: list[int], batch: int = 150) -> pl.DataFrame:
    rows = []
    for i in range(0, len(ids), batch):
        url = _PEOPLE_URL.format(ids=",".join(map(str, ids[i:i + batch])))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        for p in json.loads(urllib.request.urlopen(req, timeout=60).read())["people"]:
            if p.get("height"):
                rows.append({"batter": p["id"], "height_ft": _parse_height(p["height"])})
    return pl.DataFrame(rows, schema={"batter": pl.Int64, "height_ft": pl.Float64})


def _abs_heights() -> pl.DataFrame:
    """2026 的實測身高：每位打者 sz_top 的中位數 / 0.535。"""
    files = sorted(RAW_DIR.glob(f"statcast_{ABS_SEASON}_*.parquet"))
    df = pl.concat([pl.read_parquet(f, columns=["batter", "sz_top", "game_type"]) for f in files])
    return (df.filter(pl.col("game_type") == "R").drop_nulls("sz_top")
            .group_by("batter").agg(height_ft=pl.col("sz_top").median() / ABS_TOP))


def build_heights(force: bool = False) -> pl.DataFrame:
    if HEIGHTS_PATH.exists() and not force:
        return pl.read_parquet(HEIGHTS_PATH)
    batters = set()
    for s in SEASONS[:-1]:
        batters |= set(pl.read_parquet(PROCESSED_DIR / f"statcast_{s}.parquet", columns=["batter"])
                       ["batter"].unique().to_list())
    measured = _abs_heights().with_columns(source=pl.lit("abs_2026"))
    batters |= set(measured["batter"].to_list())
    rest = sorted(batters - set(measured["batter"].to_list()))
    listed = _listed_heights(rest).with_columns(source=pl.lit("listed"))
    out = pl.concat([measured, listed]).sort("batter")
    missing = batters - set(out["batter"].to_list())
    if missing:
        raise RuntimeError(f"{len(missing)} 位打者查不到身高：{sorted(missing)[:10]}…")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(HEIGHTS_PATH)
    print(f"身高：{out.height} 位打者（實測 {measured.height}、登錄 {listed.height}）→ {HEIGHTS_PATH.name}")
    return out


def attach_abs_zone(df: pl.DataFrame) -> pl.DataFrame:
    """加上 ABS 定義的上下緣與 `plate_z_abs`（0 = 下緣、1 = 上緣）。原本的欄位不動。"""
    h = pl.read_parquet(HEIGHTS_PATH).select("batter", "height_ft")
    out = df.join(h, on="batter", how="left")
    n_missing = out["height_ft"].null_count()
    if n_missing:
        raise RuntimeError(f"{n_missing} 顆球的打者沒有身高，重跑 build_heights(force=True)")
    return out.with_columns(
        abs_top=pl.col("height_ft") * ABS_TOP, abs_bot=pl.col("height_ft") * ABS_BOT,
    ).with_columns(
        plate_z_abs=(pl.col("plate_z") - pl.col("abs_bot")) / (pl.col("abs_top") - pl.col("abs_bot"))
    )


if __name__ == "__main__":
    build_heights()
