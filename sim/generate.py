"""合成資料生成：真值是我設的，所以可以直接檢驗估計式。

Step 7 的外部驗證都是拿我的估計去比另一個我也不知道真值的東西（Savant 的榜單、
去年的自己）。模擬是唯一能問「這個估計式偏不偏、區間準不準」的地方。

規模刻意設小
------------
真實規模是每季 5 萬顆 shadow zone 球、380 個參數，一次 NUTS 要一兩分鐘。五個情境
各跑上百次重複就是幾十小時，跑不完。所以合成資料用 30 位捕手 × 每人約 500 球，
一次一萬多列。

**這是計算成本的取捨，不是統計上的選擇。** 樣本量會影響 coverage，小樣本下的結論
不能直接外推到真實規模——這句要寫進 METHODS。

哪些東西從真實資料抽
--------------------
球位機率分布和接球數的不均勻程度都從 2022 的 shadow zone 抽，不自己編一個好看的
分布。現實中主戰捕手和替補差一個數量級，那個不均勻正是要檢驗的東西之一。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

N_CATCHERS = 30
N_UMPIRES = 20
N_PITCHERS = 40
PITCHES_PER_CATCHER = 500

TAU_CATCHER = 0.15
TAU_UMPIRE = 0.20
TAU_PITCHER = 0.17

HEAVY_TAIL_DF = 3          # t 分布自由度，越小尾巴越重
OMITTED_BETA = 0.6         # 遺漏變數對 logit 的效果大小
LOCATION_TILT = 1.5        # 投手球位偏好的強度

SCENARIOS = (
    "baseline",
    "unequal",
    "umpire_confound",
    "battery",
    "location_shared",     # 位置交互，但球位分布與捕手無關 → 實際上不構成設錯，見下
    "location_mix",        # 位置交互 + 各捕手面對不同的球位分布
    "heavy_tail",          # 捕手效果重尾，違反 N(0, τ²) 先驗
    "omitted_covariate",   # 有一個影響判決、與捕手相關、但模型看不到的變數
)

# 前四個情境操弄的是資料結構，後四個操弄的是「模型假設對不對」。
#
# location_shared 是一個失敗的設計，刻意留著：它讓捕手效果隨球位變化，看起來像
# 模型設錯，但球位是 iid 抽的、與捕手無關，所以每位捕手面對的球位分布幾乎相同
# （各捕手平均球位之間的 sd 只有 0.017）。位置交互項平均之後退化成常數
# γ_c·E[centred]，也就是另一個常數捕手效果——常數截距模型完全抓得到。
#
# 教訓是：要讓「效果隨 x 變化」真的構成設錯，各單位面對的 x 分布必須不同。
# location_mix 就是補上這一點。


@dataclass
class Truth:
    """一次模擬的真值。effect_delta 才是估計目標，不是 u_catcher。"""
    u_catcher: np.ndarray
    u_umpire: np.ndarray
    u_pitcher: np.ndarray
    effect_delta: np.ndarray = field(default=None)


def _real_shadow_pool(season: int = 2022) -> tuple[np.ndarray, np.ndarray]:
    """真實 shadow zone 的 (基準機率分布, 每位捕手球數分布)。"""
    from models.hierarchical_v2 import shadow_frame

    df = shadow_frame(seasons=[season])
    p = df["baseline_prob_oof"].to_numpy()
    counts = df.group_by("fielder_2").agg(n=pl.len())["n"].to_numpy()
    return p, np.sort(counts)[::-1]


def _workloads(rng, scenario: str, n_catchers: int, per_catcher: int,
               real_counts: np.ndarray) -> np.ndarray:
    """每位捕手的球數。unequal 情境保留現實中的極度不均。"""
    total = n_catchers * per_catcher
    if scenario == "unequal":
        w = np.sort(rng.choice(real_counts, n_catchers, replace=True))[::-1].astype(float)
        return np.maximum((w / w.sum() * total).round().astype(int), 30)
    return np.full(n_catchers, per_catcher)


def simulate(
    scenario: str = "baseline",
    seed: int = 0,
    n_catchers: int = N_CATCHERS,
    per_catcher: int = PITCHES_PER_CATCHER,
    pool: tuple[np.ndarray, np.ndarray] | None = None,
    confound_strength: float = 0.5,
) -> tuple[pl.DataFrame, Truth]:
    """生成一份合成資料。

    confound_strength 只對 omitted_covariate 有意義：遺漏變數的捕手層級成分相對於
    捕手效果本身的大小。0 等於沒有混淆，1 等於混淆和真效果一樣大。掃這個參數比
    挑單一數值有用——它回答「混淆要多大，區間才會失去意義」，而不是「在我挑的
    這個強度下會不會壞」。
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"未知情境 {scenario}；可用：{SCENARIOS}")
    rng = np.random.default_rng(seed)
    p_pool, real_counts = pool if pool is not None else _real_shadow_pool()

    # 捕手效果置中：估計目標是「相對於聯盟平均捕手」，而模型只能把捕手效果估到
    # 相差一個常數（全域水準被截距吸收）。真值如果用未置中的 u_c 定義，兩者就差
    # 一個常數，會在每個情境上表現成同一個假偏誤。這裡直接讓真值本身置中。
    if scenario == "heavy_tail":
        # t(3) 標準化到同樣的 sd，總變異相同但會出現遠離中心的離群捕手。
        # 模型的 N(0, τ²) 先驗會把這些人過度收縮，區間跟不上。
        u_c = rng.standard_t(HEAVY_TAIL_DF, n_catchers) * (TAU_CATCHER / np.sqrt(HEAVY_TAIL_DF / (HEAVY_TAIL_DF - 2)))
    else:
        u_c = rng.normal(0, TAU_CATCHER, n_catchers)
    u_c = u_c - u_c.mean()
    u_u = rng.normal(0, TAU_UMPIRE, N_UMPIRES)
    u_p = rng.normal(0, TAU_PITCHER, N_PITCHERS)

    counts = _workloads(rng, scenario, n_catchers, per_catcher, real_counts)
    catcher = np.repeat(np.arange(n_catchers), counts)
    n = len(catcher)

    # 主審指派：baseline 隨機；umpire_confound 讓部分捕手系統性遇到寬鬆主審
    if scenario == "umpire_confound":
        lenient = np.argsort(u_u)[-N_UMPIRES // 3:]          # 最寬鬆的三分之一
        favoured = np.arange(n_catchers) < n_catchers // 3    # 前三分之一捕手受惠
        umpire = np.where(
            favoured[catcher] & (rng.random(n) < 0.7),
            rng.choice(lenient, n),
            rng.integers(0, N_UMPIRES, n),
        )
    else:
        umpire = rng.integers(0, N_UMPIRES, n)

    # 投捕配對：battery 情境提高集中度（每位捕手只配少數投手）
    if scenario in ("battery", "location_mix", "omitted_covariate"):
        per_c = max(2, N_PITCHERS // n_catchers)
        staff = np.array([rng.choice(N_PITCHERS, per_c, replace=False) for _ in range(n_catchers)])
        pitcher = staff[catcher, rng.integers(0, per_c, n)]
    else:
        pitcher = rng.integers(0, N_PITCHERS, n)

    if scenario == "location_mix":
        # 每位投手有自己的球位偏好，捕手又綁定投手群，所以各捕手面對的球位
        # 分布真的不同——這是 location_shared 缺的那一塊。
        tilt = rng.uniform(-LOCATION_TILT, LOCATION_TILT, N_PITCHERS)
        bins = np.quantile(p_pool, np.linspace(0, 1, 41))
        centres = (bins[:-1] + bins[1:]) / 2
        w = np.exp(np.outer(tilt, (centres - 0.5) * 2))
        w /= w.sum(1, keepdims=True)
        p_base = np.array([rng.choice(centres, p=w[q]) for q in pitcher])
    else:
        p_base = rng.choice(p_pool, n)
    logit_base = np.log(p_base / (1 - p_base))

    # misspecified：捕手效果隨球位變化，但擬合時仍當成常數截距
    if scenario in ("location_shared", "location_mix"):
        gamma = rng.normal(0, TAU_CATCHER, n_catchers)      # 位置交互，強度同主效果
        gamma = gamma - gamma.mean()
        centred = (p_base - 0.5) * 2                        # −1..1，低機率端到高機率端
        u_c_i = u_c[catcher] + gamma[catcher] * centred
    else:
        u_c_i = u_c[catcher]

    eta = logit_base + u_c_i + u_u[umpire] + u_p[pitcher]

    if scenario == "omitted_covariate":
        # 一個真的影響判決、與捕手相關、但模型看不到的變數。這正是 v1 的
        # Limitations 列的那一串：球種、球速、位移、球場。
        #
        # 關鍵在真值怎麼定：Δ 只算 u_c 的貢獻，**不含 z**。也就是問「這個估計有沒有
        # 分離出捕手自己的效果」，而不是「它有沒有準確估到捕手加上他配到的投手群」。
        # 前者才是整個專案在問的問題，而估計式會把兩者混在一起。
        #
        # z 必須有**捕手層級的成分**才會咬。試過兩個沒用的版本：
        #   投手層級 → 被投手隨機效果整包吸收（階層 coverage 96.3%）
        #   逐球雜訊 → 與捕手無關，只是多一點未解釋變異（94.7%）
        # 任何「控制投手、主審、球位之後仍與捕手相關」的東西，模型只能把它算進
        # u_c。所以這裡給 z 一個捕手層級成分，而真值不把那份算給捕手。
        #
        # 這有點套套邏輯——把資料分不開的東西從真值裡拿掉，coverage 當然會垮。
        # 但這就是重點：它量出「捕手的技術」和「他碰巧配到的投手群」混在一起時
        # 會差多少，也說明其他情境的 coverage 漂亮**不等於**可以做因果解讀。
        zeta = rng.normal(0, confound_strength * TAU_CATCHER / OMITTED_BETA, n_catchers)
        zeta = zeta - zeta.mean()
        z = zeta[catcher] + rng.normal(0, 1, n)
        eta = eta + OMITTED_BETA * z
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))

    # 估計目標：每位捕手在他自己那批球上的平均額外好球率
    p_with = 1 / (1 + np.exp(-eta))
    p_without = 1 / (1 + np.exp(-(eta - u_c_i)))
    # 註：omitted_covariate 的 eta 已含 z，但 p_with − p_without 只差 u_c_i，
    # 所以 Δ 算的就是捕手自己的貢獻，z 不在裡面。這正是要的。
    delta = np.zeros(n_catchers)
    for c in range(n_catchers):
        m = catcher == c
        delta[c] = (p_with[m] - p_without[m]).mean()

    df = pl.DataFrame({
        "is_strike": y.astype(np.int8),
        "logit_base": logit_base,
        "baseline_prob": p_base,
        "fielder_2": catcher.astype(np.int64),
        "umpire_id": umpire.astype(np.int64),
        "pitcher_g": pitcher.astype(np.int64),
    })
    return df, Truth(u_c, u_u, u_p, delta)


if __name__ == "__main__":
    pool = _real_shadow_pool()
    for sc in SCENARIOS:
        df, truth = simulate(sc, seed=0, pool=pool)
        counts = df.group_by("fielder_2").agg(n=pl.len())["n"]
        print(f"{sc:16s} n={df.height:>7,}  球數 {counts.min():>4}–{counts.max():>5}  "
              f"strike rate {df['is_strike'].mean():.3f}  "
              f"true delta sd {truth.effect_delta.std():.4f}")
