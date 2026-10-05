"""Post-hoc calibration checks from existing fits; no NUTS fitting or downloads.

Run: uv run python -m models.recalibration
Writes the corrected 2023 Savant correlation comparison and per-pitch drift checks
to artifacts; scripts.make_results exports those tables to results/.
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl

from data.official import RAW_DIR
from data.umpires import attach_umpires
from models.baseline_gam import ARTIFACT_DIR, _matrix, load_modeling_frame
from models.baseline_v2 import MODEL_PATH
from models.hierarchical_v2 import SHADOW_HI, SHADOW_LO, _logit
from models.holdout import HOLDOUT_SEASONS, _recalibrate, cache_path
from models.intervals import RUN_VALUE
from models.validate import MIN_SHADOW, N_BOOT, paired_bootstrap_diff

CORRECTED_PATH = ARTIFACT_DIR / "corrected_external_checks.parquet"
PER_PITCH_PATH = ARTIFACT_DIR / "per_pitch_recalibration.parquet"


def run() -> tuple[pl.DataFrame, pl.DataFrame]:
    with open(MODEL_PATH, "rb") as fh:
        gam = pickle.load(fh)
    frames, shifts = {}, {}
    for season in HOLDOUT_SEASONS:
        df = load_modeling_frame(season)
        p = np.concatenate([gam.predict_proba(_matrix(df.slice(i, 12000)))
                            for i in range(0, df.height, 12000)])
        sf = (attach_umpires(df.with_columns(p=pl.Series(p)), season).drop_nulls("umpire_id")
              .filter((pl.col("p") > SHADOW_LO) & (pl.col("p") < SHADOW_HI)))
        p, y = sf["p"].to_numpy(), sf["is_strike"].to_numpy()
        mask = np.ones(len(y), dtype=bool)
        qi, _, _ = _recalibrate(_logit(p), y, mask, slope=False)
        qs, _, _ = _recalibrate(_logit(p), y, mask, slope=True)
        frames[season] = (sf.with_columns(raw=pl.Series(y - p),
                                         intercept=pl.Series(y - qi),
                                         intercept_slope=pl.Series(y - qs))
                          .group_by("fielder_2").agg(n=pl.len(),
                                                    raw=pl.col("raw").mean(),
                                                    intercept=pl.col("intercept").mean(),
                                                    intercept_slope=pl.col("intercept_slope").mean())
                          .rename({"fielder_2": "catcher"}).sort("catcher"))
        t = frames[season]
        shift = (t["intercept"] - t["raw"]).to_numpy()
        shifts[season] = {"season": season, "intercept_shift_min": float(shift.min()),
                          "intercept_shift_max": float(shift.max())}

    with open(cache_path(2023), "rb") as fh:
        post = pickle.load(fh)
    hier = pl.DataFrame({"catcher": post["catcher_ids"],
                         "hier_runs": post["draws"].mean(0) * post["n_pitches"] * RUN_VALUE})
    off = pl.read_parquet(RAW_DIR / "official_framing_2023.parquet").rename({"id": "catcher"})
    j = frames[2023].join(hier, on="catcher").join(off, on="catcher").sort("catcher")
    hr, sv = j["hier_runs"].to_numpy(), j["rv_tot"].to_numpy()
    rr = j["intercept"].to_numpy() * j["n"].to_numpy() * RUN_VALUE
    corrected = pl.DataFrame([{"season": 2023, "pitches": "shadow",
                               "baseline": "intercept", "r_hierarchical": float(np.corrcoef(hr, sv)[0, 1]),
                               "r_residual_recalibrated": float(np.corrcoef(rr, sv)[0, 1]),
                               **paired_bootstrap_diff((hr, sv), (rr, sv), n_boot=N_BOOT)}])
    for a, b in zip(HOLDOUT_SEASONS[:-1], HOLDOUT_SEASONS[1:]):
        j = frames[a].join(frames[b], on="catcher", suffix="_b").filter(
            (pl.col("n") >= MIN_SHADOW) & (pl.col("n_b") >= MIN_SHADOW))
        shifts[a] |= {"next_season": b, "n_catchers_pair": j.height,
                      **{f"r_yoy_{col}": float(np.corrcoef(j[col], j[f"{col}_b"])[0, 1])
                         for col in ("raw", "intercept", "intercept_slope")}}
    per_pitch = pl.DataFrame(list(shifts.values()), infer_schema_length=None)
    corrected.write_parquet(CORRECTED_PATH)
    per_pitch.write_parquet(PER_PITCH_PATH)
    return corrected, per_pitch


if __name__ == "__main__":
    for table in run():
        print(table)
