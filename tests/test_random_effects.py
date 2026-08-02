"""用合成資料檢查隨機效應的抽取是否正確對應到組別。

`variance_components()` 假設 statsmodels 的 `vcp_mean` 依 vc dict 的 key 順序
排列（catcher, umpire, pitcher），`extract_effects()` 則靠正則從 `vc_names`
解析組別與 id。這兩個假設都沒有明文 API 保證——statsmodels 若改變排列方式，
τ 會被靜默錯標到別組，而結論「主審變異大於捕手」正是建立在這個標籤上。

這裡造三組真實 SD 差距明顯的資料（umpire ≫ pitcher ≫ catcher），
檢查還原後的順序與數值都對得上。約 2 秒。
"""

import numpy as np
import pandas as pd
import pytest

from models.hierarchical import extract_effects, variance_components

N_CATCHER, N_UMPIRE, N_PITCHER = 12, 10, 14
SD_CATCHER, SD_UMPIRE, SD_PITCHER = 0.15, 0.80, 0.40


@pytest.fixture(scope="module")
def synthetic_fit():
    """在已知變異成分的資料上 fit，回傳 (model, result, 實際抽樣 SD)。"""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    rng = np.random.default_rng(7)
    ce = rng.normal(0, SD_CATCHER, N_CATCHER)
    ue = rng.normal(0, SD_UMPIRE, N_UMPIRE)
    pe = rng.normal(0, SD_PITCHER, N_PITCHER)

    n = 12_000
    ci = rng.integers(0, N_CATCHER, n)
    ui = rng.integers(0, N_UMPIRE, n)
    pi = rng.integers(0, N_PITCHER, n)
    x = rng.normal(0, 1, n)
    eta = 0.2 + x + ce[ci] + ue[ui] + pe[pi]
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))

    df = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "catcher": ci.astype(str),
            "umpire": ui.astype(str),
            "pitcher_s": pi.astype(str),
        }
    )
    vc = {
        "catcher": "0 + C(catcher)",
        "umpire": "0 + C(umpire)",
        "pitcher": "0 + C(pitcher_s)",
    }
    model = BinomialBayesMixedGLM.from_formula("y ~ x", vc, df)
    result = model.fit_vb()
    # 比對對象是「這次抽出來的」SD，不是母體 SD——level 數少時兩者可以差很多
    realized = {"catcher": ce.std(), "umpire": ue.std(), "pitcher": pe.std()}
    return model, result, realized


def test_variance_components_labels_match_groups(synthetic_fit):
    model, result, realized = synthetic_fit
    tau = variance_components(model, result)

    # 順序必須還原：umpire 最大、catcher 最小
    assert tau["umpire"] > tau["pitcher"] > tau["catcher"], tau

    # 每組數值也要落在實際抽樣 SD 附近（VB 近似，容忍度放寬）
    for group, truth in realized.items():
        assert abs(tau[group] - truth) < 0.15, (group, tau[group], truth)


def test_extract_effects_maps_group_and_id(synthetic_fit):
    model, result, _ = synthetic_fit
    eff = extract_effects(model, result)

    counts = {
        r["group"]: r["len"]
        for r in eff.group_by("group").len().iter_rows(named=True)
    }
    assert counts == {"catcher": N_CATCHER, "umpire": N_UMPIRE, "pitcher": N_PITCHER}

    # id 應解析回 0..n-1 的 level 標籤
    cat_ids = sorted(eff.filter(eff["group"] == "catcher")["mlbam_id"].to_list())
    assert cat_ids == list(range(N_CATCHER))
