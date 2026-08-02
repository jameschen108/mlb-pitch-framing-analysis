"""第二層（簡單版）：捕手 framing runs。

用第一層基準模型輸出的每球好球機率，量化每位捕手「多偷/多丟」了幾個好球：

    extra_strikes(捕手) = Σ (實際判定 − 基準機率)
    framing_runs(捕手)  = extra_strikes × RUN_VALUE

正值 = 比平均捕手多偷好球（好 framer）。RUN_VALUE 為 ball→strike 的期望失分價值
（0.125 runs，與 Savant 相同）。

這是**未調整版**：只控制了進壘位置與球數，沒有控制主審/投手（那是階層模型的
工作）。主審與投手造成的殘差會一併記到捕手頭上，所以它誇大的是捕手之間的
「離散度」而非單向灌水——階層模型把兩端的極端值都往 0 拉回（實測離散度收斂到
本版的 82%）。

關於與官方榜單的高相關（r≈0.99）：Savant 說明其 framing runs 含**球場與投手
調整**，完整模型未公開，對主審的處理也無文件說明；本版兩種調整都沒做。因此這
份吻合說明的是位置模型校準正確、換算合理，並顯示那些調整不太改變榜單順序，
但不能佐證任何一方分離出了捕手的真實技術。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# ball→strike 的期望失分價值（企劃書）。設為常數是簡化；嚴謹版會隨球數變化。
RUN_VALUE = 0.125


def load_baseline(season: int = 2023) -> pl.DataFrame:
    """載入含 baseline_strike_prob 的資料（由 02_gam_baseline notebook 產生）。"""
    path = DATA_PROCESSED / f"statcast_{season}_baseline.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在；請先跑 notebooks/02_gam_baseline.ipynb 產生基準機率"
        )
    return pl.read_parquet(path)


def framing_leaderboard(df: pl.DataFrame, run_value: float = RUN_VALUE) -> pl.DataFrame:
    """按捕手加總 framing runs，回傳排行榜。"""
    lb = (
        df.group_by("fielder_2")
        .agg(
            called_pitches=pl.len(),
            actual_strikes=pl.col("is_strike").sum(),
            expected_strikes=pl.col("baseline_strike_prob").sum(),
        )
        .with_columns(
            extra_strikes=pl.col("actual_strikes") - pl.col("expected_strikes"),
        )
        .with_columns(
            framing_runs=pl.col("extra_strikes") * run_value,
            # 每 1000 顆判定球的 framing runs，便於跨樣本量比較
            framing_runs_per_1000=(pl.col("extra_strikes") * run_value)
            / pl.col("called_pitches")
            * 1000,
        )
        .rename({"fielder_2": "mlbam_id"})
        .sort("framing_runs", descending=True)
    )
    return lb


def add_names(lb: pl.DataFrame) -> pl.DataFrame:
    """用 Chadwick 反查加上捕手姓名。"""
    from pybaseball import playerid_reverse_lookup

    ids = lb["mlbam_id"].to_list()
    look = playerid_reverse_lookup(ids, key_type="mlbam")
    names = pl.from_pandas(look[["key_mlbam", "name_first", "name_last"]]).with_columns(
        name=(pl.col("name_last").str.to_titlecase() + ", " + pl.col("name_first").str.to_titlecase())
    ).select(pl.col("key_mlbam").alias("mlbam_id"), "name")
    return lb.join(names, on="mlbam_id", how="left")


def merge_official(lb: pl.DataFrame, official: pl.DataFrame) -> pl.DataFrame:
    """把我方榜單與官方榜單 inner join（只留兩邊都有的合格捕手）。"""
    off = official.select(
        pl.col("id").alias("mlbam_id"),
        pl.col("rv_tot").alias("official_runs"),
        pl.col("pitches").alias("official_pitches"),
    )
    return lb.join(off, on="mlbam_id", how="inner")


if __name__ == "__main__":
    from data.official import fetch_official_framing

    df = load_baseline(2023)
    lb = add_names(framing_leaderboard(df))
    print("=== 我方 framing runs 前 10 ===")
    print(lb.select(["name", "called_pitches", "extra_strikes", "framing_runs"]).head(10))

    merged = merge_official(lb, fetch_official_framing(2023))
    corr = merged.select(pl.corr("framing_runs", "official_runs")).item()
    print(f"\n合格捕手 n={len(merged)}，與官方 rv_tot 的 Pearson r = {corr:.3f}")
