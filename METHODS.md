# Methods

**English** | [繁體中文](METHODS.zh-TW.md)

Technical companion to [`README.md`](README.md). It covers what is being estimated and how, what this design can and cannot identify, and where the numbers are weaker than they look.

---

## 1. What is being estimated, and how

These are two separate questions. v1 did not keep them apart, which made it hard to check.

**The estimand.** For catcher *c*, the average change in called-strike probability over the pitches he actually received, from replacing a league-average receiver with him:

```
Δ_c = mean over c's pitches of [ P(strike | c receives) − P(strike | average receiver) ]
```

Δ is an average over pitches. It is reported per 100 shadow-zone pitches, or summed and multiplied by 0.125 runs for the leaderboard.

Three properties of Δ are worth noting. It is measured against the league average, because a random-effects model identifies catcher effects only up to a constant; the global level goes into the intercept. It is averaged over that catcher's own pitches, so two catchers who faced different pitch mixes are each scored on what they actually saw. And it is on the probability scale, not the logit scale.

**The estimators.** There are two, both targeting Δ:

- **Residual runs** (the v1 approach): `Δ̂_c = mean(actual − baseline predicted)` over that catcher's pitches. No adjustment for umpire or pitcher. Intervals come from a binomial standard error.
- **Hierarchical model**: crossed random intercepts for catcher, umpire and pitcher on the residual, with the baseline logit entering as a calibration covariate whose coefficient is estimated rather than fixed at 1. Δ is computed per posterior draw and summarized.

Δ is used instead of the logit-scale coefficient `u_catcher` on purpose. Residual runs has no `u_catcher`, so comparing the two estimators on `u` would compare different quantities. Δ is what both estimate and what the leaderboard reports, so the simulation in section 6 tests the same number that gets published.

---

## 2. Data and design

2021–2023 regular seasons, called pitches only: 1.06M, or 1.04M after dropping pitches with |plate_x| > 2.5 ft or standardized height outside [−1, 2] to keep the spline from extrapolating. 2024 and 2025 (0.71M called pitches, 0.69M after the same trim) were added after v2 was locked, as evaluation seasons only (§2.4). Statcast via `pybaseball`; home-plate umpires from the MLB Stats API, joined on `game_pk`. Heights are standardized as `(plate_z − sz_bot) / (sz_top − sz_bot)`.

### 2.1 Out-of-sample baseline probabilities

The whole measure rests on the residual `actual − predicted`. A baseline model fit on the pitches it scores flattens those residuals. Every baseline probability in this project comes from a model that never saw the pitch:

- **2021–2022**: five-fold cross-fitting, with folds assigned by `game_pk`. Five GAM fits, each predicting the fold it did not see.
- **2023–2025**: predicted by a model fit on the 2021–2022 training split only. That model is never refit, which by 2025 has a cost (§4.1).

The two schemes differ in form, but both are out of sample.

On this measure, skipping the step matters less than it first appeared. On 2023, the same unadjusted estimator correlates with Savant at 0.990 using an in-sample baseline and 0.958 using an out-of-sample one. Re-estimating only the out-of-sample baseline's intercept on 2023 brings it to 0.989 ([`results/savant_decomposition.csv`](results/savant_decomposition.csv)). So most of the gap is the season's overall strike rate, which a baseline from other seasons does not have, and which residual runs, lacking an intercept, turn into a term proportional to playing time: before the correction, a catcher's runs correlate −0.38 with the number of pitches he received, and after it −0.16, close to Savant's −0.14. The remaining difference is small, but it is not a measurement of flattened residuals: the 0.990 and the 0.989 come from different pipelines. Re-estimating one parameter on 2023 is itself in-sample, but a single intercept cannot overfit 350,000 pitches. The case for out-of-sample baselines rests on the argument at the top of this section, not on this number. Cross-fitting within 2021–2022 has much less of this problem, because every fold's model is fit on the same two seasons it predicts. It is not zero: the pooled fit sits between the two seasons, under-predicting 2021's overall strike rate by 0.24 points and over-predicting 2022's by 0.24. The train-only model applied to 2023–2025 is off by 0.6 to 2.6 points (§4.1).

### 2.2 Splitting by game, not by pitch

The train/validation split used to evaluate the baseline is on `game_pk`, not on individual pitches. Pitches within a game share an umpire, a park, and that day's zone. Splitting by pitch would put correlated observations on both sides and make the validation loss optimistic, which defeats the purpose of the split. The cost is that the split cannot land on exactly 20%; it landed on 19.93%.

A bootstrap over games on the validation split's shadow-zone residual sums showed that this clustering widens standard errors by a factor of **1.21 to 1.29**. That is less than the 1.5 to 1.8 I had guessed, but not negligible.

One comparison uses a different unit. When two estimators are compared against the same external target (README §7), the quantities being correlated are already one number per catcher, and the game-level clustering is absorbed in the step that produced them. There the resampling unit is the catcher, and the resample is paired across the two estimators (`paired_bootstrap_diff` in [`models/validate.py`](models/validate.py)). Resampling the two estimators independently would count the variation in which catchers are sampled twice and widen the interval, which here would have made it easier to reach the conclusion I already expected.

### 2.3 The shadow zone

Statcast's Shadow Zone is geometric: a band one ball-width either side of the rule-book strike zone. The zone used in this project is defined by the model instead: the pitches the fitted baseline puts at 0.2 < p̂ < 0.8. The two overlap, but they are not interchangeable. In particular they do not share a denominator, so any figure here that sits beside a Savant figure is computed over a different set of pitches. Below, "shadow zone" always means the model-defined band.

The analysis is restricted to this band for two reasons, in order of importance.

**Substantive**: framing can only operate where the call is in doubt. A pitch down the middle is a strike regardless of who catches it.

**Statistical**: the discarded pitches carry almost no information. Fisher information about a catcher's effect scales with p(1−p), which is near zero at the extremes. Summing over 2023, with the band drawn by v1's in-sample baseline (the v2 baseline puts 15% of 2023 pitches in it; [`results/shadow_information.csv`](results/shadow_information.csv)):

| | Share of pitches | Share of information |
|---|--:|--:|
| Shadow zone (0.2–0.8) | 14.5% | **60.8%** |

Across the 75 catchers with at least 1,000 called pitches, the standard error of the catcher effect grows by a factor of 1.28 on average, not the 2.6 that the raw pitch counts would suggest.

**Computational cost** is a third, weaker reason. At matched iteration counts, NUTS costs 11.4× more per iteration on a full season than on the shadow zone, which would put a full-data fit at roughly an hour. That is slow but feasible. What the restriction really buys is cheap refitting: a pooled two-season fit takes about seven minutes, and the threshold checks, the engine comparison and the simulation all need many fits.

**Circularity.** The shadow zone is defined by the baseline model, so the baseline has to be fit on all taken pitches first, and the subset taken afterwards.

**Sensitivity.** All three thresholds were committed to in advance in a private working plan, along with a commitment not to pick whichever gave the narrowest intervals. The plan's timing cannot be verified from outside, but the outcome can. Refitting the train pool at each threshold ([`models/sensitivity.py`](models/sensitivity.py), [`results/sensitivity_threshold.csv`](results/sensitivity_threshold.csv)):

| Threshold | Pitches | Catchers | τ catcher | τ umpire | P(τ_u > τ_c) | Intervals excluding zero | Mean interval width | r vs 0.2/0.8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.15 / 0.85 | 130,159 | 149 | 0.1859 | 0.1860 | 0.50 | 37 (24.8%) | 5.61 runs | 0.991 |
| **0.20 / 0.80** | 103,615 | 148 | 0.1883 | 0.1857 | 0.46 | 34 (23.0%) | 5.08 runs | — |
| 0.25 / 0.75 | 81,971 | 148 | 0.1802 | 0.1880 | 0.63 | 24 (16.2%) | 4.41 runs | 0.987 |

The leaderboard is stable across thresholds: per-catcher framing runs correlate at 0.991 and 0.987 with the reported band. The variance-component ordering is not. P(τ_umpire > τ_catcher) is 0.50, 0.46 and 0.63 across the three. This is a second reason, separate from the season-to-season flip, not to state that ordering as a fact about baseball. Both the season and the threshold move it, and neither moves it far enough to settle it.

The reported band is also not the convenient one. The narrowest intervals are at 0.25/0.75 (4.41 runs), not at the reported 0.20/0.80 (5.08), and the band that separates the most catchers from zero is 0.15/0.85 (24.8% against 23.0%). Interval width in runs scales with the number of pitches in the band, so a narrower band gives narrower intervals and smaller run totals at the same time. The better measure is the share of catchers separated from zero, and on that the reported band is in the middle of the three.

### 2.4 Locking 2023

2023 was locked for the whole of v2's development. Model form, shadow-zone threshold, inference engine, estimand and the list of reported quantities were all settled on 2021–2022. The hierarchical model was fit on 2023 once, at the end, so the new figures could be set against v1's published 2023 table.

This is not a holdout in the strict sense: v1 had analyzed 2023 and published a leaderboard on it, and comparing against that table is the reason this round uses the season. What the lock does guarantee is that no v2 decision (the threshold, the engine, the estimand, the reported quantities) was tuned on 2023; it is a locked evaluation set, not unseen data.

The lock had a cost: with 2023 reserved, year-over-year stability rested on a single season pair. 2024 and 2025 paid most of it back.

**2024 and 2025.** Both seasons were fetched after every decision above had been made, so none of those decisions could have been tuned on them. Neither was seen by v1 either, which makes them closer to a true holdout than 2023 is. Each goes through the same path as 2023: the train-only baseline, the same threshold and sampler settings, one fit. With 2021–2023 they give ten pairs of seasons for year-over-year stability instead of one (README §8). The training data did not grow.

Adding them meant touching 2023 again. The year-over-year pairs that include 2023 read its cached posterior, and the baseline's calibration was scored on it (§4.1) and diagnosed on it. None of this refit the 2023 model or informed a choice.

---

## 3. Identification

Random effects handle small-sample instability. They do not handle confounding.

### 3.1 Catcher and umpire separate cleanly

Umpires are assigned to games close enough to randomly, and the schedule mixes enough, that the two effects separate well. On 2023 shadow-zone pitches:

- 102 catchers × 94 umpires = 9,588 possible pairs; **3,811 observed (39.7%)**
- Among the 66 catchers with ≥300 shadow-zone pitches, the median catcher worked with **50 distinct umpires**; the least-exposed worked with 27

That is dense crossing, so `u_catcher` and `u_umpire` are identified.

### 3.2 Catcher and pitcher do not separate cleanly

The pairing is lopsided:

- A catcher's pitches coming from his single most frequent pitcher: median **14.9%**, 90th percentile 22.4%. Catchers see many pitchers.
- A pitcher's pitches caught by his single most frequent catcher: median **60.5%**, 90th percentile **84.8%**.

A pitcher is mostly caught by one catcher, so the data have little room to tell that pitcher's effect from his catcher's. A pitcher's constant effect is in the model, and the simulation confirms that the pitcher term absorbs it (§6.2). What the concentration leaves is the problem of separating two terms the model already has.

This shows up directly in the posterior. For each catcher with at least 300 shadow-zone pitches, take the pitcher he caught most, and correlate their effects across posterior draws (66 pairs, pooled group excluded):

| | ρ |
|---|--:|
| Median | **−0.221** |
| 10th percentile | −0.304 |
| Most negative | −0.374 |
| Share below −0.2 | 63.6% |

The negative correlation is what partial non-identification looks like: within the posterior, credit for the same calls trades off between the catcher and his battery-mate. The model cannot tell whose it is, so it splits the difference and widens both intervals.

So credit between catcher and pitcher is only partly separable. In the one battery scenario simulated (section 6), where pairings are concentrated on purpose, the hierarchical intervals still cover 94%, so at that strength the concentration shows up as wider intervals rather than lost coverage. That is one scenario at one strength with 30 catchers; it does not show that the estimate is unbiased under every pairing structure. What it cannot separate is anything about a pitcher that changes with the catcher he throws to. The pitcher term does not capture that, so it lands on the catcher. This is one reason the catcher term should be read as variation associated with the catcher under this specification, not as a measure of skill.

### 3.3 What shrinkage does and does not do

Partial pooling deals with the fact that a catcher with 150 shadow-zone pitches has a noisy raw estimate. It does not deal with the fact that his pitches may have come mostly from two pitchers. Shrinkage makes small-sample estimates less extreme; it does nothing about a systematic source of bias. The two problems can look similar on a leaderboard, but they are unrelated.

---

## 4. Estimation

### 4.1 Baseline

```
logit P(strike) = te(plate_x, plate_z_std) + f(stand) + f(p_throws) + f(balls) + f(strikes)
```

Logistic GAM (pyGAM), tensor spline with 20 splines per margin. The model is judged on validation log loss, not accuracy or AUC. The framing metric is built on the predicted probabilities themselves, so calibration is what matters, and AUC is insensitive to it.

In-sample and out-of-sample log loss differ by 0.002 (0.17346 vs 0.17149). With about 410 basis functions, 550,000 rows and a penalty term, this split shows no overall gap between in-sample and out-of-sample log loss, so v1's in-sample reporting was a methodological flaw that did not visibly inflate its fit statistics. That says nothing directly about v1's other estimates.

The later seasons show the limits of comparing log loss across seasons. On 2023, 2024 and 2025 the train-only baseline scores 0.1691, 0.1695 and 0.1693, close to the validation split's 0.1715, yet it overpredicts the overall strike rate by 0.8, 0.6 and 2.6 points; in 2025 it predicts 0.358 against an actual 0.332 ([`results/baseline_transport.csv`](results/baseline_transport.csv)). Log loss does respond to the shift. Re-estimating only the intercept within each season lowers it by 0.0006 in 2023, 0.0004 in 2024 and 0.0060 in 2025, and within the 2025 shadow zone from 0.609 to 0.581. What hid the 2025 drift is that seasons differ: after the correction 2025 scores 0.1633, below 2023's 0.1686, so its pitches were easier to predict, and that offset the cost of the drift. The 2025 shift has two unrelated sources. In raw `plate_x`, the called zone is about 0.06 ft narrower than in 2024 for both batter sides; this data cannot tell whether the umpires or the tracking changed. And Statcast's `sz_top` has risen from 3.36 ft in 2023 to 3.44 ft, while the umpires' top edge in raw feet has stayed between 3.46 and 3.49 ([`results/baseline_drift.csv`](results/baseline_drift.csv)). As a result, the mean shadow-zone residual in 2025 is −0.109, against −0.03 in 2023 and 2024. The hierarchical model absorbs a season-wide shift through its intercept `a`. The unadjusted estimator has no intercept, and README §7 shows what that costs it. To check whether a fixed model still fits a later season, compare predicted and actual strike rates, or log loss before and after an intercept correction within that season. Raw log loss compared across seasons will not show it.

Calibration is the weaker part. Out of fold, the fitted surface over-predicts strikes below p̂ = 0.5 and under-predicts above it, by 0.7 to 1.6 points. On the 103,615 out-of-fold shadow-zone pitches, all three bins below 0.5 over-predict and all three above under-predict ([`results/baseline_calibration.csv`](results/baseline_calibration.csv)). The S-shape is real, most likely from the spline over-smoothing the transition band. Recalibrating isotonically (fit on the whole train pool) shifts per-catcher runs by amounts spanning **0.32 runs**, against a spread of 31.5 runs across the same 84 catchers (≥300 shadow-zone pitches, 2021–2022; [`results/isotonic_shift.csv`](results/isotonic_shift.csv)). That is too small to change any conclusion.

### 4.2 Hierarchical model

```
logit P(strike) = a + b·logit_base + u_catcher + u_umpire + u_pitcher
u_g = τ_g · z_g,    z_g ~ N(0, 1),    τ_g ~ HalfNormal(0.5)
a ~ N(0, 2),        b ~ N(1, 1)
```

The model has two stages, but it is not an offset model. The baseline logit enters the second stage as a covariate whose coefficient `b` is estimated under a N(1, 1) prior; an offset would hold that coefficient at 1. The difference matters. With an offset, the second stage takes the first stage's predictions exactly as they are. A free `b` can rescale them, absorbing the slope part of the §4.1 miscalibration before the random effects see the residual. Fit freely on the train pool, `b` has posterior mean **1.089**, 95% interval **[1.071, 1.107]**, with P(b > 1) = 1.000. The interval excludes 1, so the data reject the offset assumption. (v1's VB fit gives 1.031 on full-season 2023 and 1.026 on the pooled three seasons, [`results/v1_fit_summary.csv`](results/v1_fit_summary.csv). Those use a different sample and a different engine, so they do not contradict this.)

Fixing `b` at 1 barely changes the results. Refitting the same pitches with the offset model leaves per-catcher framing runs correlated with the free fit at r = 0.9997 (Spearman 0.9991). The largest single-catcher shift is **0.40 runs against a leaderboard spread of 28.9**, the mean shift is 0.06, the top ten are the same ten, and the largest rank change among 148 catchers is 12 places. The catcher and umpire components move by less than 0.005 and the pitcher component by 0.006, and P(τ_umpire > τ_catcher) stays at 0.46. The offset assumption is wrong, but the estimates do not depend on it. Both fits are in [`models/sensitivity.py`](models/sensitivity.py), summarized in [`results/sensitivity_slope.csv`](results/sensitivity_slope.csv).

This check was run on the 2021–2022 train pool rather than on 2023, which had already been fit once (§2.4). Refitting 2023 for a robustness check would use it a second time.

Pitchers with fewer than 100 shadow-zone pitches in the data being fitted share a single pooled effect: 749 of 1,123 pitchers, carrying 25% of the pitches, on the train pool; 669 of 835, carrying 48%, on 2023; 659 of 825 (47%) on 2024; and 692 of 850 (49%) on 2025. Their individual effects would be heavily shrunk anyway, but for those pitches the pitcher term is a single shared intercept that says nothing about who actually threw the pitch.

The random effects are written in non-centered form. When τ is small, the centered form tends to produce a funnel-shaped posterior that NUTS explores badly, with divergent transitions. A centered version was never tried here; the choice follows standard advice. All fits returned zero or near-zero divergences (2 across 800,000 draws in the simulation study).

Switching engines did not change the estimates. On identical data (2022, shadow zone), variational Bayes and NUTS agree on per-catcher effects at r = 0.9999 (Spearman 0.9998), and VB's posterior standard deviation is 0.95× NUTS's. Variational inference is known to understate posterior variance, and here the understatement is about 5%. What NUTS adds is the joint posterior. The statsmodels VB fit is mean-field, an independent normal approximation for each parameter, so it drops the posterior correlations that Δ, the pairwise probabilities and P(τ_umpire > τ_catcher) depend on, including the catcher–pitcher correlation in §3.2.

---

## 5. Uncertainty

**Δ posterior.** For each posterior draw, η is reconstructed for every pitch, Δ is computed as `σ(η) − σ(η − u_catcher)`, and the result is averaged within catcher. This gives a posterior distribution over Δ for each catcher; the interval is its 2.5/97.5 percentiles.

**Runs.** `runs = Δ × shadow-zone pitches × 0.125`. The pitch count and run value are fixed for each catcher, so the interval on runs is the interval on Δ rescaled. It is computed per draw anyway, which would matter only for totals across catchers, whose draws are correlated.

**Pairwise comparisons.** `P(Δ_A > Δ_B)` is counted directly from the joint posterior draws, so the comparison accounts for the correlation between the two estimates.

**Separability.** On 2023, 27 of 102 catchers (26%) have intervals excluding zero, and in 27% of pairs one catcher is ahead of the other with posterior probability above 0.95. Restricting to catchers with ≥1,000 shadow-zone pitches raises these to 53% and 55%. Neighbors in the ranking rarely separate: of the 101 adjacent pairs, 99 have P below 0.6, with a median of 0.52. The top two separate on 2023 (P = 0.98) but not on 2021–2022 (0.77). On 2024 and 2025, 20% and 16% of catchers have intervals excluding zero and 23% and 18% of pairs are ordered at 0.95; the top two do not separate in either (0.65 and 0.70), which leaves 2023 the only fit where they do ([`results/separability.csv`](results/separability.csv)).

**Coverage at the extremes.** In simulation, coverage for the largest third of effects runs 4 to 7 points below overall coverage in every scenario, including the unconfounded baseline, where it is 89.0% against 94.7%. Partial pooling gets its stability by pulling the tails in, so some shortfall at the extremes should be expected wherever it is used; the size measured here belongs to these simulated settings (30 catchers, about 500 pitches each). The caterpillar plot carries this caveat in its caption.

---

## 6. Simulation design

Simulation is the only check in this project where the truth is set rather than inferred. Its results are in the README; this section covers how it is built and what it does not establish.

### 6.1 Scale

Synthetic datasets use **30 catchers × ~500 pitches each**, about 15,000 rows, against a real analysis set of 50,000–100,000. The size is a compute trade-off: a single fit takes 9 seconds at this size, which makes 800 fits feasible. Coverage depends on sample size, so these figures should not be read as exact for full-season samples.

Pitch-location probabilities and the inequality of catcher workloads are drawn from the real 2022 shadow zone rather than from a convenient parametric distribution. The real inequality, with starters catching an order of magnitude more than backups, is one of the things being tested.

### 6.2 Scenarios

Eight scenarios in two groups. The first four change the data structure; the last four attack the model's assumptions.

| Scenario | Manipulation |
|---|---|
| baseline | Umpires assigned at random |
| unequal | Workloads from 30 to 1,561 pitches |
| umpire_confound | Some catchers systematically draw permissive umpires |
| battery | Catcher–pitcher pairings concentrated |
| location_shared | Catcher effect varies with location; locations independent of catcher |
| location_mix | As above, but catchers face different location distributions |
| heavy_tail | Catcher effects from t(3), violating the normal prior |
| omitted_covariate | A variable affecting calls, correlated with catcher, absent from the model; half the size of the catcher effect (swept in §6.4) |

`location_shared` looks like a misspecification but is not one. When locations are drawn independently of catcher, every catcher faces the same distribution, so the interaction term averages to the constant `γ_c · E[centred]`, where `centred = 2(p̂ − 0.5)` puts the baseline probability on a −1 to 1 scale. A constant-intercept model recovers that exactly. The catcher-level spread in mean location is 0.017 under this scenario, against 0.078 under `location_mix`. It stays in the table as a control.

Two further scenario designs failed before `omitted_covariate` worked, both for the same reason: the fitted model absorbed what was meant to break it. A covariate defined at the pitcher level is taken up whole by the pitcher random effect. Per-pitch noise is uncorrelated with the catcher and only adds unexplained variance. Only a covariate with a catcher-level component, with the truth defined to exclude that component, affects coverage.

That last construction is close to tautological: removing from the truth something the data cannot separate guarantees that coverage fails once the removed part is large enough. At the strength used in the table it is not yet large enough, and the hierarchical intervals cover 92.9%. It is included because the size of the failure is informative, and because it shows that good coverage elsewhere does not justify a causal reading.

### 6.3 Metrics

Bias, RMSE, 95% interval coverage, and rank recovery (Spearman against the true ordering), for both estimators, 100 replications per scenario. Monte Carlo error on coverage is ±1.1%.

Coverage is also reported by effect size. Under heavy tails only one or two of thirty catchers are outliers. Missing both in every replication would cost aggregate coverage under seven points, and missing them only some of the time barely registers, so a failure at the tail can hide in the aggregate.

### 6.4 The confounding sweep

Rather than choose one confounder size and report whether it broke the intervals, the strength of the catcher-correlated confounder is swept from zero to twice the true catcher effect, at 50 replications per point. The sweep runs on the `omitted_covariate` generator, which also concentrates battery pairings, so the unadjusted estimator starts at 78% even at zero strength. The sweep turns "does unmeasured confounding matter?", whose answer depends on an arbitrary choice of size, into "how large would it have to be?".

### 6.5 What the simulation does not establish

In five of the eight scenarios the data are generated from the model's own functional form, so good coverage there is close to guaranteed and should not be read as validation. The three scenarios built to break the model did not break it at the strengths used. That limits the claim: the intervals survive the misspecifications tested here, at those strengths. The sweep in §6.4 shows one of them breaking the intervals once it is made larger.

---

## 7. What the numbers here bound

The limitations section of [`README.md`](README.md) lists the scope limits, such as unmodeled pitch characteristics, a flat run value and the coordinate trim. The four that can be quantified are here.

**Catcher and pitcher are partly inseparable.** A pitcher's pitches are caught by his most frequent catcher a median 60.5% of the time, and 84.8% at the 90th percentile. The posterior correlation between a catcher's effect and his most-caught pitcher's effect has a median of ρ = −0.221, with 63.6% of pairs below −0.2. In the one battery scenario simulated, the intervals held at 94%, so at that strength the cost was precision rather than coverage; that is not a general guarantee. The catcher term also still picks up anything about a pitcher that changes with the catcher he throws to.

**Coverage at the extremes is 87–91%, not 95%**, in every simulated scenario, including the unconfounded baseline. Shrinkage is the cause, so some shortfall should be expected on other data too, but the 87–91% figure belongs to these simulated settings. It affects the top and bottom of the leaderboard.

**Unmeasured catcher-correlated confounding has a measurable cost:** 94% coverage with none, 88% when it is as large as the catcher effect, and 66% at twice that. This is a different problem from the pairing concentration in §3.2: a variable that follows the catcher and is never observed. Nothing in the data says where on that curve the real analysis sits, and better inference would not change that, because it is a property of the design rather than of the estimation.

**The baseline is fit once and drifts.** Fit on 2021–2022 and never refit, it overpredicts the overall strike rate by 0.6 to 0.8 points in 2023 and 2024 and by 2.6 points in 2025 (§4.1). The hierarchical estimates absorb this through their intercept. The unadjusted estimator cannot: in 2025 it agrees with Savant at 0.647 against the hierarchical model's 0.955, and re-estimating two calibration parameters on the season brings it back to 0.964. Any comparison across seasons that keeps one baseline fixed has to measure this drift first.

---

## 8. Pitfalls encountered

Recorded because each cost time, and each would have produced a wrong number if it had gone unnoticed.

**One validation split was too small to detect a bias of this size.** A 971-game split showed a shadow-zone residual bias significant at p ≈ 0.03 at all three thresholds. Five-fold cross-fitting over all 4,856 games put it at z = +0.30, covering zero. The agreement across thresholds looked like three pieces of evidence but was only one: the bands share most of their pitches, so the three tests are nearly the same test, and a single p ≈ 0.03 on a single split is weak.

**Timing JAX without blocking.** `mcmc.run()` returns while the computation is still running, so a timer stopped right after it measures dispatch rather than the fit. Call `jax.block_until_ready(mcmc.get_samples())` before stopping the clock.

**`numpyro.set_host_device_count` is ignored once XLA has initialized.** Calling it inside each fit means a later 4-chain run falls back to sequential execution, with only one warning on stderr and twice the wall time. The device count has to be set through `XLA_FLAGS` before JAX is imported.

**Effects relative to zero vs relative to the league mean.** The two differ by a constant, which shows up in a simulation as a small bias that is the same in every scenario. When five structurally different scenarios return nearly the same bias (+0.0029 to +0.0030), the cause is arithmetic, not statistics.

**Comparing log loss across seasons hid a 2.6-point calibration shift.** Scored on 2025, the fixed baseline's log loss was no worse than on its own validation split, because 2025's pitches happened to be easier to predict. Within the season log loss did respond: an intercept correction lowered it by 0.006. The drift was obvious as soon as predicted and actual strike rates were set side by side. Read across seasons, the log loss gave no warning, and the unadjusted estimator's 0.647 against Savant in 2025 could have been taken for a failure of the estimator rather than of a stale baseline.

**arviz 1.3.0 is incompatible with numpyro 0.21.0**; `az.from_numpyro` fails inside `infer_dims`. Diagnostics here use `numpyro.diagnostics` directly.

---

## 9. Reproducibility

Environment is managed with [uv](https://docs.astral.sh/uv/). v1's pipeline is unchanged; see [`v1.0`](../../tree/v1.0) for its instructions. This round adds:

```bash
uv sync

# out-of-fold baseline probabilities for 2021–2022 (~20 min, cached)
uv run python -m models.crossfit

# baseline model scored out of sample
uv run python -m models.baseline_v2

# posterior intervals and pairwise comparisons
uv run python -m models.intervals

# simulation: eight scenarios x 100 reps (~2 hours), then the confounder sweep
uv run python -m sim.run 100
uv run python -m sim.run sweep 50

# sensitivity: b free vs b = 1, then the two alternative shadow-zone thresholds;
# four fits on the train pool (~30–40 min)
uv run python -m models.sensitivity

# 2023 (fit once)
uv run python -m models.holdout

# 2024–2025: pitches, umpires and Savant's leaderboard, then one fit per season
uv run python -m data.fetch --season 2024
uv run python -m data.fetch --season 2025
uv run python -m data.umpires 2024 2025
uv run python -m data.official 2024 2025
uv run python -m models.holdout --season 2024
uv run python -m models.holdout --season 2025

# external validation: every pair of seasons 2021–2025, and Savant by season
uv run python -m models.validate

# the train-only baseline on later seasons, and the checks on the 2025 row (section 7)
uv run python -m models.holdout --transport
uv run python -m models.holdout --drift

# section 3's decomposition, and METHODS §2.3 and §4.1 (shadow-zone information, calibration, isotonic shift)
uv run python -m models.holdout --decompose
uv run python -m models.shadow_checks

uv run python -m scripts.make_figures_v2

# every table in the README, from the caches above (refits nothing)
uv run python -m scripts.make_results
```

The postscript's commands are in [ABS.md §7](ABS.md#7-reproduce).

Seeds are fixed: `20260906` for splits and cross-fitting folds, and the replication index for simulation draws. Data files, model caches and simulation output are not version-controlled and are regenerated by the commands given. The published tables are also written to version-controlled CSVs under [`results/`](results/), so they can be checked without refitting anything; [`results/README.md`](results/README.md) lists which other numbers are printed by modules instead.

**Tests** ([`tests/`](tests/), 23 in all, about ten seconds, no downloads). Eight are v1's: data cleaning and labels, the standardization and Spearman–Brown formulas, and whether mixed-model effects land on the right groups. The nine for v2 cover three things that fail silently rather than loudly. First, cross-fitting folds and the train/validation split never divide a game, which is the whole point of out-of-sample baseline probabilities. Second, the per-draw Δ computation matches a hand calculation and assigns each catcher its own effect; a misalignment there would hand every catcher someone else's number without raising an error. Third, NUTS recovers known parameters on synthetic data with a fixed seed. That is the only check that would catch a change to the v2 model itself, such as a centered parameterization, a wrong prior or a mis-wired `logit_base`. Six more were added for the ABS postscript ([`ABS.md`](ABS.md)). They check that overturned calls are flipped back to the umpire's call, and that missing or mismatched challenge records stop the pipeline instead of passing silently. Continuous integration runs them on every push.

**Project structure**

```
data/
  fetch.py             monthly Statcast fetch, cleaning, standardization
  umpires.py           home-plate umpire per game (MLB Stats API)
  challenges.py        2026 ABS challenges, to recover the umpire's original call
  heights.py           batter heights, for the height-based zone in every season
  official.py          Baseball Savant framing leaderboard (comparison)
models/
  baseline_gam.py      v1 first-stage GAM strike-probability model
  framing_runs.py      v1 unadjusted residual framing runs
  hierarchical.py      v1 two-stage crossed random-effects model (VB)
  reliability.py       split-half and year-over-year reliability
  splits.py            train/validation by game, 2023 locked
  baseline_v2.py       baseline fit on train, scored out of sample
  crossfit.py          out-of-fold baseline probabilities
  hierarchical_v2.py   crossed random effects, NUTS on the shadow zone
  compare_engines.py   VB vs NUTS, season stability
  intervals.py         posterior intervals and pairwise probabilities
  identify.py          identifiability diagnostics
  validate.py          year-over-year and Savant comparison
  sensitivity.py       b free vs fixed at 1, shadow-zone thresholds
  holdout.py           2023–2025, one fit each; baseline calibration and drift
  shadow_checks.py     shadow-zone information, calibration bins, isotonic shift
  abs_era.py           the postscript: per-season fits on the ABS zone, 2021–2026
sim/                   simulation study: generate, estimate, run
notebooks/             01_eda … 06_multiseason (v1)
tests/                 invariants: cleaning, v1 effect mapping, v2 estimator, ABS call recovery (23 tests, ~10s)
scripts/
  make_figures.py      v1 figures, both languages
  make_figures_v2.py   v2 figures, both languages
  make_results.py      every published table, as CSV, from cached fits
results/               those CSVs (version-controlled; the caches are not)
docs/images/           en/ and zh/ figures used by the two READMEs
.github/workflows/     CI: pytest on synthetic data, no downloads
```
