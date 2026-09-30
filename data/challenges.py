"""2026 起的 ABS 挑戰紀錄，用來還原主審的**原判**。

為什麼需要
----------
Statcast 的 `description` 記的是挑戰之後的最終判決：打者挑戰成功的球記成 `ball`，
捕手挑戰成功的記成 `called_strike`，而且沒有任何欄位標記這件事。framing 量的是主審
被捕手影響的程度，要的是原判，所以得從別處把挑戰找回來。

MLB Stats API 的 `playByPlay` 在每一顆被挑戰的球上有 `reviewDetails`：
`reviewType`、`isOverturned`、挑戰的球員與球隊。ABS 挑戰是 `reviewType == "MJ"`、
而且落在判定球上（call code `B` / `C`）；其他代碼是一般的 replay review（觸殺、
觸身球等），不碰判定。`MJ` 在 play 層級會重複出現一次，只取 pitch 層級。

`playByPlay` 回傳的 call 也是最終判決，所以原判 = 被推翻的反過來。

快取
----
- `data/raw/challenges_{season}.parquet`：每一次 ABS 挑戰一列
- `data/raw/challenge_games_{season}.parquet`：已經抓過的場次（含沒有挑戰的場次），
  用來分辨「這場沒挑戰」和「這場還沒抓」

用法
----
    uv run python -m data.challenges 2026
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

ABS_FIRST_SEASON = 2026
ABS_REVIEW_TYPE = "MJ"
CALLED_CODES = {"B": 0, "C": 1}          # call code → 最終判決是不是好球

_PBP_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"

SCHEMA = {
    "game_pk": pl.Int64, "at_bat_number": pl.Int64, "pitch_number": pl.Int64,
    "final_strike": pl.Int8, "overturned": pl.Boolean,
    "challenger_id": pl.Int64, "challenge_team_id": pl.Int64,
}


def challenges_path(season: int) -> Path:
    return RAW_DIR / f"challenges_{season}.parquet"


def games_path(season: int) -> Path:
    return RAW_DIR / f"challenge_games_{season}.parquet"


def _game_challenges(game_pk: int, max_retries: int = 3, backoff: float = 2.0):
    """回傳 (成功與否, 該場 ABS 挑戰的列)。"""
    url = _PBP_URL.format(game_pk=game_pk)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, max_retries + 1):
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=60).read())
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == max_retries:
                print(f"  game_pk={game_pk} 抓取失敗：{exc!r}")
                return False, []
            time.sleep(backoff * attempt)

    rows = []
    for play in data.get("allPlays", []):
        at_bat = play["about"]["atBatIndex"] + 1          # Statcast 的 at_bat_number 從 1 起算
        for ev in play.get("playEvents", []):
            rd = ev.get("reviewDetails")
            code = ev.get("details", {}).get("call", {}).get("code")
            if not rd or not ev.get("isPitch") or rd.get("reviewType") != ABS_REVIEW_TYPE:
                continue
            if code not in CALLED_CODES:
                raise ValueError(f"game_pk={game_pk} 有 ABS 挑戰落在非判定球上（call code {code!r}）")
            rows.append({
                "game_pk": game_pk, "at_bat_number": at_bat, "pitch_number": ev.get("pitchNumber"),
                "final_strike": CALLED_CODES[code], "overturned": bool(rd.get("isOverturned")),
                "challenger_id": rd.get("player", {}).get("id"),
                "challenge_team_id": rd.get("challengeTeamId"),
            })
    return True, rows


def season_game_pks(season: int) -> list[int]:
    """例行賽場次，取自已經抓下來的 Statcast 月檔。"""
    files = sorted(RAW_DIR.glob(f"statcast_{season}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"沒有 {season} 的 Statcast 月檔，先抓逐球資料")
    frames = [pl.read_parquet(f, columns=["game_pk", "game_type"]) for f in files]
    return (pl.concat(frames).filter(pl.col("game_type") == "R")["game_pk"]
            .unique().sort().to_list())


def fetch_challenges(season: int = ABS_FIRST_SEASON, force: bool = False) -> pl.DataFrame:
    """抓該季每一場的 ABS 挑戰，可中斷續跑。"""
    game_pks = season_game_pks(season)
    done_games = pl.DataFrame(schema={"game_pk": pl.Int64})
    rows = pl.DataFrame(schema=SCHEMA)
    if games_path(season).exists() and not force:
        done_games = pl.read_parquet(games_path(season))
        rows = pl.read_parquet(challenges_path(season))

    done = set(done_games["game_pk"].to_list())
    todo = [g for g in game_pks if g not in done]
    print(f"ABS 挑戰：共 {len(game_pks)} 場，已抓 {len(done)}，待抓 {len(todo)}", flush=True)

    new_games, new_rows, failed = [], [], []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (gp, (ok, r)) in enumerate(zip(todo, ex.map(_game_challenges, todo)), 1):
            (new_games if ok else failed).append(gp)
            new_rows += r
            if i % 400 == 0:
                print(f"  …{i}/{len(todo)}", flush=True)

    done_games = pl.concat([done_games, pl.DataFrame({"game_pk": new_games}, schema={"game_pk": pl.Int64})])
    rows = pl.concat([rows, pl.DataFrame(new_rows, schema=SCHEMA)])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    done_games.write_parquet(games_path(season))
    rows.write_parquet(challenges_path(season))
    print(f"完成：{done_games.height} 場、{rows.height} 次挑戰、推翻 {rows['overturned'].sum()} 次"
          + (f"；{len(failed)} 場失敗，重跑即可續抓" if failed else ""))
    return rows


def _explain_unmatched(unmatched: pl.DataFrame, season: int) -> None:
    """對不上樣本的挑戰，必須在原始資料裡確認它不是有球位的判定球，才可以略過。

    例如落在 `automatic_ball`（投球計時違規，沒有球位）上的挑戰，或 Statcast 與
    playByPlay 對同一球記法不同（界外 vs 壞球）。這些球本來就不在 framing 樣本裡。
    解釋不了的就停下來。
    """
    keys = ["game_pk", "at_bat_number", "pitch_number"]
    files = sorted(RAW_DIR.glob(f"statcast_{season}_*.parquet"))
    raw = (pl.concat([pl.read_parquet(f, columns=[*keys, "description", "plate_x"]) for f in files])
           if files else pl.DataFrame(schema={**{k: pl.Int64 for k in keys},
                                              "description": pl.Utf8, "plate_x": pl.Float64}))
    j = unmatched.join(raw, on=keys, how="left")
    outside = j.filter(pl.col("description").is_not_null()
                       & (~pl.col("description").is_in(["called_strike", "ball"])
                          | pl.col("plate_x").is_null()))
    if outside.height != unmatched.height:
        raise RuntimeError(f"{unmatched.height - outside.height} 次挑戰對得上 Statcast 的判定球卻不在樣本裡，"
                           "或在原始資料裡找不到")
    kinds = dict(outside["description"].value_counts().iter_rows())
    print(f"  {outside.height} 次挑戰落在不進樣本的球上，略過：{kinds}")


def attach_original_call(df: pl.DataFrame, season: int) -> pl.DataFrame:
    """把 `is_strike` 換成主審原判，最終判決留在 `is_strike_final`。

    2026 以前沒有挑戰，原判 = 最終判決，直接複製。2026 起必須有完整的挑戰快取——
    缺場次就停下來，不靜默當成「那場沒有挑戰」。每一次挑戰都必須對得上 Statcast 的
    一顆判定球，而且最終判決要一致，否則也停下來。
    """
    df = df.with_columns(is_strike_final=pl.col("is_strike"))
    if season < ABS_FIRST_SEASON:
        return df.with_columns(challenged=pl.lit(False), overturned=pl.lit(False))

    if not games_path(season).exists():
        raise FileNotFoundError(f"{season} 有 ABS 挑戰，先跑 `python -m data.challenges {season}`")
    done = set(pl.read_parquet(games_path(season))["game_pk"].to_list())
    missing = set(df["game_pk"].unique().to_list()) - done
    if missing:
        raise RuntimeError(f"{len(missing)} 場還沒抓挑戰紀錄，重跑 `python -m data.challenges {season}`")

    ch = pl.read_parquet(challenges_path(season))
    keys = ["game_pk", "at_bat_number", "pitch_number"]
    out = df.join(ch.select(*keys, "final_strike", "overturned"), on=keys, how="left")

    matched = out.filter(pl.col("final_strike").is_not_null())
    if matched.height != ch.height:
        _explain_unmatched(ch.join(matched.select(keys), on=keys, how="anti"), season)
    bad = matched.filter(pl.col("final_strike") != pl.col("is_strike_final"))
    if bad.height:
        raise RuntimeError(f"{bad.height} 次挑戰的最終判決與 Statcast 不一致")

    overturned = pl.col("overturned").fill_null(False)
    return out.with_columns(
        is_strike=pl.when(overturned).then(1 - pl.col("is_strike_final"))
        .otherwise(pl.col("is_strike_final")).cast(pl.Int8),
        challenged=pl.col("final_strike").is_not_null(),
        overturned=overturned,
    ).drop("final_strike")


if __name__ == "__main__":
    import sys

    for season in map(int, sys.argv[1:] or [str(ABS_FIRST_SEASON)]):
        fetch_challenges(season)
