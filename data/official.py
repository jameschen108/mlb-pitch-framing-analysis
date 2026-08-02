"""抓取 Baseball Savant 官方 catcher framing 榜單（驗證用）。

pybaseball.statcast_catcher_framing() 目前壞掉（savant 端點改版，回傳 HTML），
且舊 CSV 端點會忽略 year 參數。改直接解析 savant catcher_framing 頁面中
嵌入的 `data = [...]` JSON——這才是年度正確的資料。

欄位（保留）：id（MLBAM，對應 fielder_2）、name（"Last, First"）、
pitches、rv_tot（官方 framing run value）。
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"

_PAGE_URL = (
    "https://baseballsavant.mlb.com/catcher_framing"
    "?year={year}&team=&min=q&type=catcher"
    "&sortColumn=diff_calls&sortDirection=desc"
)
_DATA_RE = re.compile(r"data = (\[.*?\]);", re.DOTALL)


def fetch_official_framing(year: int = 2023, force: bool = False) -> pl.DataFrame:
    """抓官方 framing 榜單，快取到 data/raw/official_framing_{year}.parquet。"""
    path = RAW_DIR / f"official_framing_{year}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)

    url = _PAGE_URL.format(year=year)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    m = _DATA_RE.search(html)
    if not m:
        raise RuntimeError("找不到嵌入的 framing 資料，savant 頁面結構可能又改了")

    records = json.loads(m.group(1))
    df = pl.DataFrame(records)
    keep = [c for c in ["id", "name", "pitches", "rv_tot"] if c in df.columns]
    df = df.select(keep).with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col("rv_tot").cast(pl.Float64),
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"官方 {year} framing 榜單：{len(df)} 位捕手 → {path.name}")
    return df


if __name__ == "__main__":
    df = fetch_official_framing(2023)
    print(df.sort("rv_tot", descending=True).head(10))
