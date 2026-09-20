"""v2 估計式的三個不變量。全部用合成資料，不需下載。

現有的 tests/ 守的是資料清理、簡單轉換，以及 v1 的隨機效應標籤對應。那些都不碰
v2 真正產出結論的那條路徑：out-of-fold 基準機率 → shadow zone → NUTS → Δ → 榜單。
這個檔案補上那條路徑上最容易安靜壞掉的三個地方。

    uv run pytest -q
"""

import numpy as np
import polars as pl
import pytest

from models.crossfit import fold_assignment
from models.splits import assign_split


# ---------------------------------------------------------------- 1. 場次不洩漏

def _synthetic_pool(n_games: int = 60, pitches_per_game: int = 25) -> pl.DataFrame:
    """每場 pitches_per_game 顆球的假 pool，只有測試用得到的欄位。"""
    return pl.DataFrame({
        "game_pk": np.repeat(np.arange(1000, 1000 + n_games), pitches_per_game),
        "is_strike": np.tile([0, 1], n_games * pitches_per_game // 2),
    })


def test_crossfit_folds_never_split_a_game():
    """同一場的每一顆球必須落在同一個 fold。

    這是 out-of-fold 基準機率的全部意義所在。若切分是按球而非按場，同一場的球會
    同時出現在訓練與測試側——而同場的球共享主審、球場與當天的好球帶，OOF 預測因此
    被污染，殘差被壓平，而被壓平的正是要測的 framing 訊號。

    壞掉的話不會噴錯，只會讓每個數字都稍微偏小，所以要有測試守著。
    """
    pool = _synthetic_pool()
    games = pool["game_pk"].unique().sort().to_numpy()
    mapping = fold_assignment(games, k=5)

    pool = pool.with_columns(
        fold=pool["game_pk"].replace_strict(mapping, return_dtype=pl.Int32)
    )

    # 每場只能對到一個 fold
    per_game = pool.group_by("game_pk").agg(pl.col("fold").n_unique().alias("n"))
    assert per_game["n"].max() == 1

    # 任一 fold 的場次集合，與其餘 fold 的訓練集不得相交
    for i in range(5):
        test_games = set(pool.filter(pl.col("fold") == i)["game_pk"].unique().to_list())
        train_games = set(pool.filter(pl.col("fold") != i)["game_pk"].unique().to_list())
        assert not (test_games & train_games)

    # 每場都要被分配到，且 fold 不能全空
    assert len(mapping) == len(games)
    assert pool["fold"].n_unique() == 5


def test_train_val_split_never_splits_a_game():
    """train/val 切分同理：以場次為單位，不以球為單位。"""
    pool = _synthetic_pool()
    out = assign_split(pool, val_fraction=0.2)

    per_game = out.group_by("game_pk").agg(pl.col("split").n_unique().alias("n"))
    assert per_game["n"].max() == 1

    train_games = set(out.filter(pl.col("split") == "train")["game_pk"].to_list())
    val_games = set(out.filter(pl.col("split") == "val")["game_pk"].to_list())
    assert not (train_games & val_games)
    assert val_games                                  # 不能切出空的驗證集


def test_fold_assignment_is_deterministic():
    """同一組場次、同一個種子，必須得到同一組分配。

    `fold_assignment` 被抽成獨立函式就是為了這件事：OOF 快取存的是一組預測值，
    重算時若分配變了，快取與資料的對應會靜默錯開。
    """
    games = np.arange(1000, 1200)
    assert fold_assignment(games, k=5) == fold_assignment(games, k=5)


# ---------------------------------------------------------------- 2. Δ 的計算

def _one_draw(u_catcher, n_catchers, catcher_idx, n=None):
    """組出 delta_draws() 要的 samples/data，單一抽樣、其他效應為零。"""
    n = len(catcher_idx) if n is None else n
    zeros = np.zeros((1, 1))
    samples = {
        "a": np.array([0.0]),
        "b": np.array([1.0]),
        "u_catcher": np.asarray(u_catcher, dtype=float).reshape(1, n_catchers),
        "u_umpire": zeros,
        "u_pitcher": zeros,
    }
    data = {
        "catcher_idx": np.asarray(catcher_idx, dtype=np.int32),
        "umpire_idx": np.zeros(n, dtype=np.int32),
        "pitcher_idx": np.zeros(n, dtype=np.int32),
        "catcher_levels": np.arange(n_catchers),
    }
    return samples, data


def test_delta_matches_hand_computation():
    """Δ 要等於「含捕手效應」與「不含」兩個預測機率的差，逐球平均。

        Δ_c = mean_i [ σ(η_i) − σ(η_i − u_c) ]

    手算：logit_base = 0、a = 0、b = 1、u_catcher = 0.5，其餘為零
        σ(0.5) − σ(0) = 0.62245933 − 0.5 = 0.12245933
    """
    from models.hierarchical_v2 import delta_draws

    samples, data = _one_draw([0.5], n_catchers=1, catcher_idx=[0, 0, 0])
    draws, n_pitches = delta_draws(samples, data, np.zeros(3), n_draws=1)

    expected = 1 / (1 + np.exp(-0.5)) - 0.5
    assert draws.shape == (1, 1)
    assert abs(draws[0, 0] - expected) < 1e-12
    assert n_pitches.tolist() == [3]


def test_delta_is_zero_when_catcher_effect_is_zero():
    """u_catcher = 0 的捕手，Δ 必須恰好是零——不是接近零。"""
    from models.hierarchical_v2 import delta_draws

    samples, data = _one_draw([0.0], n_catchers=1, catcher_idx=[0, 0])
    draws, _ = delta_draws(samples, data, np.array([1.3, -0.7]), n_draws=1)
    assert abs(draws[0, 0]) < 1e-15


def test_delta_assigns_each_catcher_its_own_effect():
    """多位捕手時，Δ 不能錯位。

    `delta_draws` 內部用 argsort + searchsorted 依捕手分組再逐組取平均。那段是整個
    榜單的來源，而錯位不會噴錯，只會讓每個人拿到別人的數字。這裡刻意讓球數不等、
    效應正負相反，錯位就會被抓到。
    """
    from models.hierarchical_v2 import delta_draws

    # 捕手 0 接 4 顆（效應 +0.8），捕手 1 接 2 顆（效應 −0.8），交錯排列
    catcher_idx = [0, 1, 0, 0, 1, 0]
    samples, data = _one_draw([0.8, -0.8], n_catchers=2, catcher_idx=catcher_idx)
    draws, n_pitches = delta_draws(samples, data, np.zeros(6), n_draws=1)

    assert n_pitches.tolist() == [4, 2]
    assert draws[0, 0] > 0 and draws[0, 1] < 0
    # 對稱效應、logit_base 全零 → 兩個 Δ 大小相等方向相反
    assert abs(draws[0, 0] + draws[0, 1]) < 1e-12


# ---------------------------------------------------------------- 3. NUTS 還原

@pytest.fixture(scope="module")
def recovery_fit():
    """在已知參數的合成資料上跑一次小型 NUTS。固定種子，約 20 秒。"""
    from models.hierarchical_v2 import run_nuts

    rng = np.random.default_rng(20260920)
    n, n_c, n_u, n_p = 6000, 10, 8, 6
    true_tau_c = 0.45

    ce = rng.normal(0, true_tau_c, n_c)
    ue = rng.normal(0, 0.20, n_u)
    pe = rng.normal(0, 0.20, n_p)

    ci = rng.integers(0, n_c, n)
    ui = rng.integers(0, n_u, n)
    pi = rng.integers(0, n_p, n)
    logit_base = rng.normal(0, 1.2, n)

    eta = 0.0 + 1.0 * logit_base + ce[ci] + ue[ui] + pe[pi]
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))

    data = {
        "y": y.astype(np.float32),
        "logit_base": logit_base.astype(np.float32),
        "catcher_idx": ci.astype(np.int32),
        "umpire_idx": ui.astype(np.int32),
        "pitcher_idx": pi.astype(np.int32),
        "catcher_levels": np.arange(n_c),
        "umpire_levels": np.arange(n_u),
        "pitcher_levels": np.arange(n_p),
    }
    mcmc = run_nuts(data, num_warmup=300, num_samples=300, chains=1,
                    seed=0, progress=False)
    return mcmc, data, ce, true_tau_c


def test_nuts_recovers_known_parameters(recovery_fit):
    """已知真值的合成資料上，NUTS 要還原斜率、τ 與每位捕手的效應。

    這是唯一一個會在 v2 模型本身被改壞時亮紅燈的測試——改錯先驗、把非中心參數化
    寫回中心化、把 logit_base 接錯，都會在這裡顯現，而其他測試全都看不到。
    """
    mcmc, _, ce, true_tau_c = recovery_fit
    s = mcmc.get_samples()

    # 斜率：真值 1.0
    b = np.asarray(s["b"])
    assert 0.85 < b.mean() < 1.15

    # τ_catcher：89% 區間要蓋住「這次實際抽出來的」SD，不是母體 SD
    tau = np.asarray(s["tau_catcher"])
    realized = ce.std()
    assert np.percentile(tau, 5.5) < realized < np.percentile(tau, 94.5)

    # 捕手效應：收縮後仍應與真值高度相關
    u = np.asarray(s["u_catcher"]).mean(0)
    assert np.corrcoef(u, ce)[0, 1] > 0.9

    # τ_catcher 明顯大於 τ_umpire（真值 0.45 對 0.20）
    assert np.asarray(s["tau_catcher"]).mean() > np.asarray(s["tau_umpire"]).mean()


def test_nuts_runs_without_divergences(recovery_fit):
    """非中心參數化下 divergence 應為零或接近零。

    τ 小的時候中心化寫法的後驗是漏斗形，NUTS 會噴大量 divergence。若有人把模型改回
    中心化，這個測試會抓到；症狀看起來像「跑不動」，實際上是幾何問題。
    """
    mcmc, *_ = recovery_fit
    divergences = int(np.sum(mcmc.get_extra_fields()["diverging"]))
    assert divergences <= 3


def test_fixing_slope_pins_b_exactly(recovery_fit):
    """slope=1.0 時 b 必須恰好是 1——offset 對照組靠這個前提成立。"""
    from models.hierarchical_v2 import run_nuts

    _, data, _, _ = recovery_fit
    mcmc = run_nuts(data, num_warmup=100, num_samples=100, chains=1,
                    seed=0, progress=False, slope=1.0)
    b = np.asarray(mcmc.get_samples()["b"])
    assert np.all(b == 1.0)
