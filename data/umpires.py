"""抓每場的主審（home plate umpire），供階層模型控制主審效應。

Statcast 不含主審欄位，但我們每顆球都有 game_pk。用 MLB Stats API 的 boxscore
端點依 game_pk 查 officials，取 officialType == "Home Plate" 的主審。
主審 id 是 MLBAM，與 fielder_2 / pitcher 同一套 id 系統。

按 game_pk 逐場抓、快取到 data/raw/umpires_{season}.parquet，可中斷續跑。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

_BOX_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"


def _hp_umpire(game_pk: int, max_retries: int = 3, backoff: float = 2.0):
    """回傳 (umpire_id, umpire_name) 或 (None, None)。"""
    url = _BOX_URL.format(game_pk=game_pk)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, max_retries + 1):
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=30).read())
            for o in data.get("officials", []):
                if o.get("officialType") == "Home Plate":
                    off = o["official"]
                    return off["id"], off["fullName"]
            return None, None  # 該場無 officials 資料
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == max_retries:
                print(f"  game_pk={game_pk} 抓取失敗：{exc!r}")
                return None, None
            time.sleep(backoff * attempt)


def fetch_umpires(season: int = 2023, force: bool = False) -> pl.DataFrame:
    """抓該季所有場次的主審，回傳 [game_pk, umpire_id, umpire_name]。

    game_pk 清單取自 processed baseline 資料。已快取的場次會沿用（續跑）。
    """
    out_path = RAW_DIR / f"umpires_{season}.parquet"
    game_pks = _season_game_pks(season)

    cached = pl.DataFrame(schema={"game_pk": pl.Int64, "umpire_id": pl.Int64, "umpire_name": pl.Utf8})
    if out_path.exists() and not force:
        cached = pl.read_parquet(out_path)

    done = set(cached["game_pk"].to_list())
    todo = [g for g in game_pks if g not in done]
    print(f"主審抓取：共 {len(game_pks)} 場，已快取 {len(done)}，待抓 {len(todo)}", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (gp, (uid, uname)) in enumerate(
            zip(todo, ex.map(_hp_umpire, todo)), 1
        ):
            rows.append({"game_pk": gp, "umpire_id": uid, "umpire_name": uname})
            if i % 400 == 0:
                print(f"  …{i}/{len(todo)}", flush=True)
                _flush(cached, rows, out_path)

    result = _flush(cached, rows, out_path)
    n_missing = result.filter(pl.col("umpire_id").is_null()).height
    print(f"完成：{result.height} 場，其中 {n_missing} 場查無主審 → {out_path.name}")
    return result


def _flush(cached: pl.DataFrame, rows: list[dict], out_path: Path) -> pl.DataFrame:
    if rows:
        cached = pl.concat([cached, pl.DataFrame(rows, schema=cached.schema)], how="vertical")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_parquet(out_path)
    return cached


def _season_game_pks(season: int) -> list[int]:
    path = PROCESSED_DIR / f"statcast_{season}_baseline.parquet"
    if not path.exists():
        path = PROCESSED_DIR / f"statcast_{season}.parquet"
    return (
        pl.read_parquet(path, columns=["game_pk"])
        .unique()
        .sort("game_pk")["game_pk"]
        .to_list()
    )


def attach_umpires(df: pl.DataFrame, season: int = 2023) -> pl.DataFrame:
    """把主審 id 併入逐球資料（依 game_pk join）。"""
    ump = pl.read_parquet(RAW_DIR / f"umpires_{season}.parquet")
    return df.join(ump.select(["game_pk", "umpire_id"]), on="game_pk", how="left")


if __name__ == "__main__":
    fetch_umpires(2023)
