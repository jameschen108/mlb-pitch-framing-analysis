"""ABS 挑戰還原原判的不變量。全部用合成資料，不需下載。

這一步錯了不會噴錯：翻錯方向的話，2026 的每一顆被推翻的球都會帶著相反的標籤進模型，
而那些球全落在 shadow zone——正是 framing 要量的地方。
"""

import polars as pl
import pytest

import data.challenges as ch


def _pitches() -> pl.DataFrame:
    """三顆判定球，is_strike 是 Statcast 的最終判決。"""
    return pl.DataFrame({
        "game_pk": [1, 1, 1],
        "at_bat_number": [1, 1, 2],
        "pitch_number": [1, 2, 1],
        "is_strike": pl.Series([0, 1, 1], dtype=pl.Int8),
    })


def _write(tmp_path, rows, games=(1,)):
    pl.DataFrame({"game_pk": list(games)}, schema={"game_pk": pl.Int64}).write_parquet(
        tmp_path / "challenge_games_2026.parquet")
    pl.DataFrame(rows, schema=ch.SCHEMA).write_parquet(tmp_path / "challenges_2026.parquet")


def _challenge(ab, pitch, final_strike, overturned):
    return {"game_pk": 1, "at_bat_number": ab, "pitch_number": pitch, "final_strike": final_strike,
            "overturned": overturned, "challenger_id": 9, "challenge_team_id": 9}


def test_overturned_calls_are_flipped_back_to_the_umpires_call(tmp_path, monkeypatch):
    """打者挑戰成功（最終 ball）→ 原判好球；捕手挑戰成功（最終 strike）→ 原判壞球。"""
    monkeypatch.setattr(ch, "RAW_DIR", tmp_path)
    _write(tmp_path, [_challenge(1, 1, 0, True),     # 打者挑戰成功
                      _challenge(1, 2, 1, True),     # 捕手挑戰成功
                      _challenge(2, 1, 1, False)])   # 挑戰失敗，維持原判
    out = ch.attach_original_call(_pitches(), 2026).sort("at_bat_number", "pitch_number")
    assert out["is_strike_final"].to_list() == [0, 1, 1]
    assert out["is_strike"].to_list() == [1, 0, 1]
    assert out["challenged"].to_list() == [True, True, True]
    assert out["overturned"].to_list() == [True, True, False]


def test_unchallenged_pitches_keep_their_call(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "RAW_DIR", tmp_path)
    _write(tmp_path, [])
    out = ch.attach_original_call(_pitches(), 2026)
    assert out["is_strike"].to_list() == out["is_strike_final"].to_list()
    assert not out["challenged"].any()


def test_missing_games_stop_the_pipeline(tmp_path, monkeypatch):
    """沒抓到的場次不能被當成「那場沒有挑戰」。"""
    monkeypatch.setattr(ch, "RAW_DIR", tmp_path)
    _write(tmp_path, [], games=())
    with pytest.raises(RuntimeError, match="還沒抓"):
        ch.attach_original_call(_pitches(), 2026)


def test_a_challenge_that_matches_no_pitch_stops_the_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "RAW_DIR", tmp_path)
    _write(tmp_path, [_challenge(7, 7, 1, True)])
    with pytest.raises(RuntimeError, match="找不到"):
        ch.attach_original_call(_pitches(), 2026)


def test_a_disagreeing_final_call_stops_the_pipeline(tmp_path, monkeypatch):
    """playByPlay 與 Statcast 的最終判決不一致，表示兩邊對錯球了。"""
    monkeypatch.setattr(ch, "RAW_DIR", tmp_path)
    _write(tmp_path, [_challenge(1, 1, 1, True)])      # Statcast 這顆是 ball
    with pytest.raises(RuntimeError, match="不一致"):
        ch.attach_original_call(_pitches(), 2026)


def test_seasons_before_abs_are_untouched():
    out = ch.attach_original_call(_pitches(), 2025)
    assert out["is_strike"].to_list() == out["is_strike_final"].to_list()
    assert not out["overturned"].any()
