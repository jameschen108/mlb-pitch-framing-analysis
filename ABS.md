# Postscript: framing in the ABS era

**English** | [繁體中文](ABS.zh-TW.md)

In 2026 the automated ball-strike (ABS) challenge system went live in the regular season. A batter, pitcher or catcher can ask for a called pitch to be checked against the tracked zone, and the call is overturned if the tracking disagrees. This document asks whether umpires' calls still respond to who is catching once that is possible. It is a separate analysis from v2, done after v2 was finished. The v2 pipeline, caches and published numbers are unchanged, and because the design below differs from v2's, the same season can carry a different τ here than in the [README](README.md).

The design was written down before any 2026 model was fit, and the section on it below is that plan. The short answer is in section 4: 2026 has the lowest catcher variation of the six seasons, but by the rule set in advance the change is not detectable, and the decline started before ABS did.

---

## 1. Three questions, one answered here

| | Question | Calls used |
|---|---|---|
| **Q1** | Do umpires' calls still vary with the catcher? | The umpire's **original** call |
| Q2 | How much framing value survives the challenges? | Final call against original call |
| Q3 | Is challenging a new catcher skill, and does it go with framing? | Challenge records |

This round answers Q1 only. Q2 is partly mechanical, since overturned stolen strikes can only lower net value, and Q3 is descriptive. Both wait for Q1 to settle.

## 2. Data

### 2.1 Statcast records the final call

The 2026 regular season ran from March 25 to September 27: 2,429 games and 356,498 called pitches. In 2026, Statcast's `description` is the call **after** any challenge. A pitch a batter successfully challenged is recorded as `ball`; one a catcher successfully challenged, as `called_strike`. No field marks either. Framing is about the umpire's call, so the original has to be recovered from elsewhere.

The MLB Stats API's `playByPlay` feed carries `reviewDetails` on every challenged pitch: the review type, whether it was overturned, and who challenged. ABS challenges are review type `MJ` on called pitches; the other review types are ordinary replay reviews and never touch a ball-strike call. The feed also shows the final call, so the original is the final call flipped wherever the challenge succeeded ([`data/challenges.py`](data/challenges.py)).

| | |
|---|--:|
| ABS challenges | 7,896 |
| overturned | 4,431 (56%) |
| on pitches outside the called-pitch sample (3 `automatic_ball`, 1 `automatic_strike`, 1 recorded as a foul) | 5 |
| matched to a called pitch | 7,891 |
| of those, final call agrees between the feed and Statcast | 7,891 (100%) |

Catchers made 4,292 of the matched challenges (55% overturned), batters 3,466 (58%) and pitchers 133 (40%). Of the 4,016 team-games with a challenge, 4,006 had at most two that failed and 10 had three. Recovering the originals moves the season's called-strike rate from 32.44% (final) to 32.32% (original): successful catcher challenges slightly outnumber successful batter ones.

The pipeline refuses to build the 2026 season without complete challenge records, and it stops if any challenge fails to match a called pitch without an explanation or disagrees with Statcast on the final call. Six tests guard the reconstruction ([`tests/test_challenges.py`](tests/test_challenges.py)).

### 2.2 One strike zone for every season

From 2026, Statcast's `sz_top` and `sz_bot` are fixed fractions of the batter's height, 53.5% and 27%: the ABS zone. The ratio is identical on every row, and from April to June, the months checked, a batter's value moved by no more than 0.0005 ft. Before 2026 they were measured pitch by pitch from the batter's stance. A standardized height built from one is not comparable with one built from the other, and 2026 no longer has the stance version, so every season is converted to the height-based zone ([`data/heights.py`](data/heights.py)).

Heights come from 2026's `sz_top / 0.535` for the 662 batters who appeared in 2026, and from the Stats API's listed height for the other 986. For 300 of the measured batters checked against their listed height, the two differ by 0.02 inches on average and by at most half an inch. The ABS zone's top edge sits lower than the stance zone's: about 3.22 ft on average in 2023 and 2025, against 3.36 and 3.44 for the stance version.

One thing not checked: at what depth over the plate `plate_z` is measured, against the ABS zone's reference plane.

## 3. Design

Everything in this section was fixed before any 2026 model was fit. At that point I had looked at one day of 2026 challenge counts and no estimates.

- **Calls**: the umpire's original call in 2026. Other seasons have no challenges.
- **Zone**: the height-based ABS zone in every season.
- **Baseline model**: the same form as v2, fit **within each season** by five-fold cross-fitting on `game_pk`. v2 fixed one model on 2021–2022, and README §7 shows it overpredicting 2025's strike rate by 2.6 points; a comparison across seasons cannot carry that. The per-season baselines match each season's strike rate to within 0.002 points ([`results/abs/baseline_check.csv`](results/abs/baseline_check.csv)).
- **Shadow zone**: 0.2 < p̂ < 0.8 under each season's own baseline.
- **Hierarchical model**: the same as v2, NUTS with 4 chains × 1,000 warm-up + 1,000 draws, one fit per season ([`models/abs_era.py`](models/abs_era.py)).
- **Baseline period**: 2021–2024. 2025 is reported but kept out of the decision, because ABS was tested in spring training that year and the called zone narrowed during it (README §7), so umpires may already have been adjusting.

**Decision rule.** For each baseline season *s*, compute P(τ_2026 < τ_s) by pairing posterior draws from the two independent fits.

- All four above 0.95: evidence of a decrease.
- All four below 0.05: evidence of an increase.
- Otherwise: no detectable change.

The ratio τ_2026 / τ_s is reported with its interval whatever the outcome.

**Expectation, written in advance**: τ_2026 lower than the baseline period, but not detectably, because only a few calls per game can be challenged, so umpires have little reason to change much, and τ had already been falling since 2023.

**A disclosure.** A background job meant to wait for the 2021–2025 fits was triggered by a progress line and ran the report while 2025 was still fitting. So 2026 against 2021–2024 was seen after the design was fixed but before every season had finished. Nothing in the design or the rule changed as a result.

## 4. Results

All eleven fits (six seasons on the ABS zone, five on the stance zone) returned zero divergences. The largest R-hat is 1.002–1.007 in 2021–2025 and 1.018 in 2026; by the plan, nothing was refit.

<p align="center">
  <img src="docs/images/en/abs_tau_by_season.png" width="620">
</p>

| | 2021 | 2022 | 2023 | 2024 | 2025 | **2026** |
|---|--:|--:|--:|--:|--:|--:|
| τ catcher | 0.180 | 0.224 | 0.226 | 0.206 | 0.169 | **0.155** |
| 95% interval | [0.145, 0.221] | [0.187, 0.268] | [0.186, 0.275] | [0.167, 0.251] | [0.135, 0.211] | [0.121, 0.193] |
| shadow-zone pitches | 52,316 | 50,387 | 50,136 | 49,128 | 48,090 | 42,902 |

| 2026 against | P(τ_2026 lower) | τ_2026 / τ_s, median [95%] |
|---|--:|---|
| 2021 | 0.869 | 0.86 [0.65, 1.11] |
| 2022 | 0.998 | 0.69 [0.53, 0.90] |
| 2023 | 0.997 | 0.69 [0.52, 0.90] |
| 2024 | 0.983 | 0.75 [0.57, 0.98] |
| 2025 (outside the rule) | 0.735 | 0.92 [0.68, 1.21] |

**By the rule: no detectable change.** Three of the four baseline seasons clear 0.95; 2021 does not. The expectation was right on both counts.

The rule's verdict is not the most important thing in the table. From 2023, τ was already falling: 0.226, 0.206, 0.169, then 0.155. 2025 cannot be told apart from 2026 (P = 0.735). Even if 2021 had cleared the bar, this design could not separate an effect of ABS from a continuation of the decline that began in 2023. That is the limit of a before-and-after comparison with no control group, and it would stand however the rule had come out.

**Persistence.** The correlation between a catcher's effect in consecutive seasons (catchers with ≥300 shadow-zone pitches in both):

| 2021→22 | 2022→23 | 2023→24 | 2024→25 | 2025→26 |
|--:|--:|--:|--:|--:|
| 0.66 | 0.65 | 0.64 | 0.57 | **0.40** |

2025→26 is the lowest, as expected, but the drop began a year earlier. Each pair has 43 to 47 catchers, so a correlation of 0.5 carries an interval of roughly ±0.25.

**Sensitivity to the zone definition.** On the stance zone, τ for 2021–2025 is 0.165, 0.204, 0.201, 0.183 and 0.171. That is 0.016 to 0.025 below the ABS-zone figures in 2021–2024 and about equal in 2025. The shape is the same, higher in 2022–2023 and falling after, but neighboring seasons swap places: 2022 and 2023, and 2021 and 2025. The definition moves the level by up to about a tenth. The main comparison uses the ABS zone throughout, since 2026 has no other, so it does not depend on that shift.

**Fewer doubtful calls.** Each season's shadow zone is drawn by its own baseline, so the number of pitches in it measures how many calls the model finds uncertain. 2026 has 10.8% fewer than 2025 and 15% fewer than the 2021–2024 average. That too continues a decline from 2023.

All tables are in [`results/abs/`](results/abs/).

## 5. What this can and cannot say

- **It cannot attribute anything to ABS.** Every team switched at once, so there is no control group, and the decline predates the rule. Anything else that changed in 2026, such as umpire turnover or pitch mix, is mixed in.
- **τ does not say which side changed.** It measures how much umpires' calls vary with the catcher. A smaller τ fits umpires responding less to framing, catchers framing less, or catchers becoming more alike, and these data cannot tell them apart.
- **Knowing a call can be challenged is part of what is measured.** If umpires call the zone differently because they may be overruled, that shows up here, and it cannot be separated from anything else that changed that season.
- **Precision.** A single season pins τ to roughly ±20%. A change of a tenth would not be visible.
- **One season.** 2026 is the only ABS season so far.

## 6. Not done yet

Q2, how much value survives the challenges, and Q3, whether challenging is a catcher skill, wait for Q1 to settle. The exploratory check registered in advance, whether umpires respond differently once the batting team has no challenges left, has not been run.

## 7. Reproduce

```bash
# 2026: pitches, challenges, then the season file (refuses to build without challenges)
uv run python -c "from data.fetch import get_or_fetch_month; [get_or_fetch_month(2026, m) for m in range(3, 11)]"
uv run python -m data.challenges 2026
uv run python -m data.fetch --season 2026
uv run python -m data.umpires 2026
uv run python -m data.heights

# one fit per season (~15–25 min each, mostly the cross-fitting), then the report
uv run python -m models.abs_era fit --zone abs
uv run python -m models.abs_era fit --zone stance
uv run python -m models.abs_era report

uv run python make_figures_v2.py --only abs_tau
uv run python make_results.py
```
