# Quantifying MLB Catcher Pitch Framing

**English** | [繁體中文](README.zh-TW.md)

[![tests](https://github.com/jameschen108/mlb-pitch-framing-analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/jameschen108/mlb-pitch-framing-analysis/actions/workflows/tests.yml)

Some catchers get more strike calls than others on identical pitches. This project estimates how many for each catcher, puts an interval on every estimate, and tests whether those intervals can be trusted.

**Approach.** For every called pitch in Statcast from 2021 to 2025, a baseline model gives an expected call from location and context, without catcher identity, and is always scored out of sample. On the pitches where that call is in doubt, a hierarchical model separates catcher, umpire and pitcher effects and puts a 95% posterior interval on each catcher's. A simulation with known true effects checks whether those intervals cover as often as they claim.

**Three findings.**

1. **Some catchers clearly differ from zero, but most adjacent ranks cannot be told apart.** In 2023, 27 of 102 catchers have intervals that exclude zero, and 99 of the 101 adjacent pairs in the ranking are ordered with posterior probability below 0.6. The estimates track Baseball Savant's published framing runs closely (r = 0.94).
2. **A high correlation with Savant cannot validate the intervals; simulation exposed a coverage gap.** In the simulated confounding scenarios, nominal 95% intervals from the unadjusted estimator, the common direct approach that averages each catcher's actual minus expected calls, cover 74–83% of the true effects, while the hierarchical model's cover 92.9–95.6% in every scenario, with the baseline treated as known. On real data, the interval on the difference between the two methods' Savant correlations includes zero in four of five seasons, so that check does not separate them.
3. **Baseline drift distorts the unadjusted estimator; in-model calibration absorbed this season-wide shift.** Fit on 2021–2022, the baseline overpredicts 2025's strike rate by 2.6 points, a shift that comparing log loss across seasons did not reveal. The hierarchical model takes it up in its intercept and still correlates 0.955 with Savant in 2025. The unadjusted estimator falls to 0.647, and returns to 0.964 once two calibration parameters are re-estimated on that season.

<p align="center">
  <img src="docs/images/en/caterpillar_2023.png" width="720"><br>
  <em>2023: each catcher's effect with its 95% credible interval. Blue intervals exclude zero; gray ones do not.</em>
</p>

This is the second version of the project. What the first version claimed, and which of those claims held up, is in [From v1 to v2](#from-v1-to-v2) further down. [`notebooks/00_results_tour.ipynb`](notebooks/00_results_tour.ipynb) shows the headline results straight from the published tables, and the [reading guide](#reading-this-repository) says where everything else is.

---

## Where the methods come from, and what I did

**From the literature.** The two-stage design comes from Judge, Pavlidis and Brooks (2015): a strike-probability model without catcher identity, then random effects for catcher, umpire and pitcher dividing the residual. Using a GAM for the called-strike surface follows Albert (2023), and Deshpande and Wyner (2017) give a hierarchical Bayesian framing model. The non-centered parameterization is standard practice.

**Design and implementation in this project:**

- **Data integration.** Statcast pitch data joined to MLB Stats API home-plate umpires on `game_pk`. Statcast records only the post-challenge call in 2026, so all 7,891 challenges on called pitches were checked against `reviewDetails` in the playByPlay feed, and the 4,429 overturned among them were restored to the umpire's original call.
- **Estimand.** Δ, on the probability scale and averaged over each catcher's own pitches, is the explicit target, and the simulation scores both methods against the same true Δ.
- **Out-of-sample design.** Five-fold cross-fitting split by game rather than by pitch. 2023 was locked during model selection, and 2024 and 2025 were fetched only after the design was fixed.
- **Simulation.** Eight scenarios with 100 replications each, built on real pitch locations and workloads, plus a sweep of confounder strength that measures how large confounding has to be before the intervals fail.
- **Diagnostics.** A breakdown of how the 2025 baseline drift affected the unadjusted estimator (its correlation with Savant fell to 0.647), and a check of how in-model calibration absorbed the season-wide shift.

---

## Data and design

Regular seasons, called pitches only (`called_strike` / `ball`): 1.06M pitches in 2021–2023 (1.04M after a coordinate trim) and 0.71M in 2024–2025. Pitch data comes from Statcast via `pybaseball`; home-plate umpires come from the MLB Stats API, joined on `game_pk`. Each season has one role:

| Season | Role | Baseline probabilities from |
|---|---|---|
| 2021–2022 | Training: model form, threshold and engine are chosen here | Five-fold cross-fitting, split by game |
| 2023 | Held back while the design was chosen; v1 had already analyzed it, so it is not an untouched holdout | A model fit on the 2021–2022 training split |
| 2024–2025 | Evaluation only, fetched after the design was fixed | The same model, never refit |
| 2026 | Postscript on the ABS era, with a separate pipeline ([ABS.md](ABS.md)) | Refit within each season |

2026 is outside the main analysis because the ABS challenge system went live that season, and it changes what a framing number means.

Two estimators are compared throughout. The **unadjusted estimator** averages each catcher's actual minus expected calls; summed and multiplied by 0.125 runs per strike, that gives **unadjusted runs**. The **hierarchical model** divides the same residual among catcher, umpire and pitcher effects (section 4).

Three design decisions matter most.

**Out-of-sample baseline probabilities throughout.** The framing signal is `actual − predicted`. If the prediction comes from a model fit on the same pitches, the residuals are partly flattened by the fit itself. Every baseline probability here comes from a model that never saw the pitch: within 2021–2022 by five-fold cross-fitting split on `game_pk`, and for 2023–2025 from a model fit only on 2021–2022.

**Analysis restricted to the shadow zone.** Here that means a band defined by the model: the pitches the baseline puts at 0.2 < p̂ < 0.8. It is not Statcast's Shadow Zone, which is geometric, one ball-width either side of the rule-book edge. Framing can only matter where the call is in doubt, and the numbers bear that out. Measured on 2023 with v1's baseline, the band holds 14.5% of called pitches but 60.8% of the Fisher information about a catcher's effect, because information scales with p(1−p) and a pitch down the middle carries almost none. (With the v2 baseline the band holds 15% of pitches.) By that information measure, standard errors grow by about a factor of 1.28, not the 2.6 the raw pitch counts would suggest ([`results/shadow_information.csv`](results/shadow_information.csv)). The approximation treats the baseline probabilities as known and was not checked against the posterior standard deviations of a full-data fit.

**2023 locked during v2 model selection; one primary fit.** Model form, threshold, engine and estimand were settled on 2021–2022 before the primary 2023 fit, which is cached for comparison with v1. The catcher–pitcher identification diagnostic in `models/identify.py` separately refits 2023. Later calibration checks also use that season. Since v1 had already analyzed it, 2023 is a locked evaluation set rather than an untouched holdout (METHODS §2.4). 2024 and 2025 were fetched after the design choices were fixed.

---

## Results

### 1. The strike zone has a soft edge, and it moves with the count

Called-strike rate over the plate shows a wide transition band around the nominal zone. That band is the only place framing can matter: pitches down the middle are strikes no matter who catches them, and pitches a foot outside are balls.

<p align="center">
  <img src="docs/images/en/gam_count_contours_2023.png" width="460"><br>
  <em>The 50% called-strike contour expands on 3-0 and shrinks on 0-2.</em>
</p>

The count moves that band. Comparing hitter's counts with pitcher's counts *at the same location* leaves a ring of difference around the zone edge and almost nothing in the middle:

<p align="center">
  <img src="docs/images/en/strike_rate_count_diff_2023.png" width="380">
</p>

The raw 0-2 vs 3-0 gap in called-strike rate (8% vs 63%) is mostly a location artifact, since 3-0 pitches are aimed at the middle. Only after holding location fixed does a count-related difference in called-strike rate show up. Pitch type and velocity also differ by count, so that difference is not purely the umpire's.

### 2. Baseline model, scored out of sample

The model is a tensor spline over `plate_x` × standardized `plate_z`, plus additive terms for batter side, pitcher side, balls, and strikes. It is fit on the 2021–2022 training split and scored on held-out games.

| | Log loss |
|---|--:|
| Train (in-sample) | 0.17346 |
| Validation (out-of-sample, split by game) | 0.17149 |

On this split there is no visible gap in overall log loss between the two (about 410 basis functions against 550,000 rows, with a penalty term). Whether an in-sample baseline moves the framing estimates themselves is a separate question; section 3 checks it against Savant, where re-estimating the baseline's intercept recovers most of the gap.

The model is never refit. By 2025 it overpredicts the overall strike rate by 2.6 points; section 7 covers where that drift comes from and what it does to the unadjusted estimator.

Calibration is weaker. Out of fold, the model over-predicts strikes below p̂ = 0.5 and under-predicts above it, by 0.7 to 1.6 points across the shadow zone. The pattern is consistent on 100,000 held-out pitches: all three bins below 0.5 over-predict and all three above under-predict. Recalibrating moves every catcher's runs a little, but the largest and smallest shifts differ by only 0.32 runs, against a spread of 31.5 runs across the same 84 catchers, so the baseline was left as is. That check covers the 2021–2022 unadjusted runs only; τ, the posterior intervals and the evaluation seasons were not recomputed with a recalibrated baseline.

### 3. What agreement with Savant can and cannot show

The usual external check on a framing estimate is its correlation with Savant's published framing runs. On 2023 that correlation can be taken apart ([`results/savant_decomposition.csv`](results/savant_decomposition.csv)):

| Setup | r vs Savant |
|---|--:|
| unadjusted runs, all pitches, **in-sample** baseline | **0.990** |
| unadjusted runs, all pitches, out-of-sample baseline | 0.958 |
| the same, with the baseline's intercept re-estimated on 2023 | 0.989 |
| unadjusted runs, shadow zone only, intercept re-estimated | 0.960 |
| hierarchical model, shadow zone, out-of-sample | 0.942 |

Moving to an out-of-sample baseline costs 0.032; re-estimating a single number on 2023, the baseline's intercept, brings the correlation back to 0.989. The out-of-sample baseline overpredicts 2023's overall strike rate by 0.8 points, and unadjusted runs have no intercept, so a season-wide offset becomes a term proportional to playing time (section 7). Most of what separates an in-sample baseline from an out-of-sample one is therefore the season's overall strike rate, which Savant, presumably fitting within season, also has right. 2024 and 2025 behave the same way: with the intercept re-estimated, the unadjusted estimator reaches 0.990 and 0.988 against Savant.

Restricting to the shadow zone then costs 0.029, and switching to the hierarchical model another 0.018. That last step matters because this correlation cannot assess interval calibration. In the simulated confounded settings, unadjusted nominal 95% intervals cover as little as 74%, while the hierarchical intervals cover 92.9–95.6%; hierarchical RMSE is about 40% lower (section 6). These simulations condition on a known baseline and do not validate uncertainty in the entire two-stage pipeline. For the actual 2023 comparison against intercept-recalibrated unadjusted runs, the correlation difference is −0.0182, with a paired catcher-bootstrap 95% interval of [−0.0360, +0.0015] ([`results/corrected_external_checks.csv`](results/corrected_external_checks.csv)). It includes zero; it does not establish equivalence. One intercept moves the correlation by 0.031, more than this estimator change. Agreement with Savant cannot independently confirm that either method isolated catcher skill.

<p align="center">
  <img src="docs/images/en/framing_vs_official_2023.png" width="440">
</p>

### 4. Catcher, umpire, pitcher, with intervals

The baseline logit enters as a calibration covariate, and three crossed random intercepts compete for the residual. The model is fit with NUTS (numpyro) on the shadow-zone subset, with a non-centered parameterization.

The covariate's coefficient is estimated (1.089, with a 95% interval that excludes 1), so it is not an offset; fixing it at 1 has little effect on the train-pool results tested (METHODS §4.2). NUTS supplies the joint posterior needed for within-fit comparisons below.

2023:

| | posterior mean | 95% interval |
|---|--:|---|
| τ catcher | 0.2001 | [0.163, 0.247] |
| τ umpire | 0.2264 | [0.190, 0.270] |
| τ pitcher | 0.2019 | [0.162, 0.245] |

P(τ_umpire > τ_catcher) = 0.81.

Running the same fit on each season gives the table below. 2024 and 2025 were added after v2 was locked and are fit the same way as 2023 (section 7).

| | P(τ_umpire > τ_catcher) |
|---|--:|
| 2021 | 0.88 |
| 2022 | 0.41 |
| 2023 | 0.81 |
| 2024 | 0.92 |
| 2025 | 0.74 |
| 2021–2022 pooled | 0.46 |

The ordering flips between 2021 and 2022. In the other four seasons it leans toward the umpire, but no season reaches 0.95.

The point estimates of all three components are smaller in 2025 than in 2023: catcher 0.200 to 0.172, umpire 0.226 to 0.188, pitcher 0.202 to 0.178. Overlapping intervals do not establish no change. Using the product of the independently fitted marginal posteriors, P(τ_2025 < τ_2023) is 0.843 for catcher, 0.926 for umpire and 0.795 for pitcher ([`results/variance_components.csv`](results/variance_components.csv)). Matching draw indices had overstated these probabilities; the corrected calculation is invariant to sample order. The baseline also fits 2025 worst (section 7). Calibration errors and omitted variables can distort the components, but their direction depends on their relation to catcher, umpire and pitcher; there is no general rule that they all move toward zero. These results do not separate a change in baseball from a change in model adequacy.

The shadow-zone threshold moves the ordering too. On the pooled train pool, P(τ_umpire > τ_catcher) is 0.50, 0.46 and 0.63 at the three thresholds committed to in advance ([`results/sensitivity_threshold.csv`](results/sensitivity_threshold.csv)). Season and threshold both move the ordering. The evidence is insufficient to establish a stable ordering; it does not disprove that umpire variation may be larger.

### 5. How far apart are catchers?

The figure at the top of this page shows 2023. In numbers:

| Restriction | Catchers | Intervals excluding zero | Pairs ordered with ≥95% probability |
|---|--:|--:|--:|
| All | 102 | 27 (26%) | 27% |
| ≥500 shadow pitches | 49 | 20 (41%) | 47% |
| ≥1000 shadow pitches | 15 | 8 (53%) | 55% |

Among the 63 catchers Savant lists as qualified, 23 have intervals excluding zero.

The two ends of the distribution separate clearly from zero, but the middle does not, and neighbors in the ranking are mostly indistinguishable. Of the 101 adjacent pairs, 99 have P(higher > lower) below 0.6, with a median of 0.52. Even among the 15 catchers with at least 1,000 shadow pitches, the median is 0.62. The top of 2023 is the exception, with Hedges ahead of Álvarez with posterior probability 0.98; on 2021–2022 the top two are not separable (0.77). It stays the only exception. The top two do not separate on 2024 (0.65) or 2025 (0.70), where 20% and 16% of catchers have intervals excluding zero ([`results/separability.csv`](results/separability.csv)).

Before running this, I wrote down in a working plan that a result like this would be acceptable, and that all three shadow-zone thresholds would be reported. The plan was private, so its timing cannot be verified from outside. What can be checked is whether the reported threshold turned out to be the convenient one.

All three are reported ([`results/sensitivity_threshold.csv`](results/sensitivity_threshold.csv), and METHODS §2.3). The leaderboard barely moves: per-catcher runs correlate at 0.991 and 0.987 with the reported band. The reported band is neither the one with the narrowest intervals nor the one that separates the most catchers from zero. The share of catchers separated from zero is 24.8%, 23.0% and 16.2%, from the widest band to the narrowest.

The figure carries a simulation caveat: among the third of catchers with the largest **absolute true effects**, coverage is 87–91%, while the remaining two thirds cover about 95–98%. This stratification uses the known simulation truth, not the observed leaderboard rank, so those percentages cannot be assigned to the top and bottom of the published leaderboard. Shrinkage contributes to the pattern in these settings; it is not a universal coverage guarantee or a measured real-data coverage rate.

### 6. Testing the estimator against a known truth

Every external check compares one estimate against another number whose true value is also unknown. Simulation is the only place where the truth is set rather than inferred, which is why it was the one step that could not be dropped.

Eight scenarios generate known catcher, umpire and pitcher effects on the real pitch-location and workload distributions, with 100 replications each. Both estimators are scored on bias, RMSE, coverage and rank recovery. The synthetic datasets are small on purpose, 30 catchers with about 500 pitches each, to keep the compute manageable, so the coverage figures should not be read as exact for full-season samples.

<p align="center">
  <img src="docs/images/en/sim_coverage_by_scenario.png" width="700">
</p>

The unadjusted estimator fails where the confounding is. Its nominal 95% intervals cover 83% under umpire confounding, 78% under concentrated battery pairings, and 74% with an omitted covariate correlated with the catcher. Without confounding it does better but still falls short, at 91–92%, because its binomial intervals count only pitch-to-pitch noise and not the umpire and pitcher variation that also ends up in its estimate. The hierarchical model stays between 92.9% and 95.6% throughout.

Four scenarios challenge the fitted conditional model: two location-varying catcher-effect settings, heavy-tailed effects, and an omitted covariate half the size of the catcher effect. None causes a large overall coverage failure at the tested strength. This empirical result does not make their conditional models correctly specified or guarantee coverage at other strengths.

What the intervals do not survive is a larger version of that omitted covariate. Instead of picking one size, the strength was swept on the same scenario:

<p align="center">
  <img src="docs/images/en/sim_confound_sweep.png" width="620">
</p>

Coverage holds while the confounder stays below about half the size of the catcher effect, falls to 88% when the two are equal, and reaches 66% at twice. The scenario also concentrates battery pairings, which is why the unadjusted estimator starts at 78% even with no confounder.

Catchers are not randomly assigned to pitchers. The battery scenario tests that directly, and the hierarchical model passes it at 94.0%. The sweep measures something else: a variable that follows the catcher and never appears in the data.

One result was not planned. In this figure, hollow markers select the largest third of **absolute true effects**, and cover about 4–7 percentage points less often than the full set in these simulated settings, including the baseline. The unadjusted estimator does not show the same systematic gap. These are empirical results of this design, not a general guarantee about either estimator.

### 7. What the external checks could and could not distinguish

Both estimators were run on identical pitches and compared on the two external checks available. The primary 2024 and 2025 fits use the same train-only baseline, shadow zone and sampler settings as the primary 2023 fit, and their cached summaries are reused ([`models/holdout.py`](models/holdout.py)). This gives four adjacent-season pairs and five Savant seasons. The separate 2023 identification fit is described in METHODS §2.4.

| Check | Hierarchical | Unadjusted | 95% CI on hierarchical − unadjusted |
|---|--:|--:|---|
| Year over year, 2021 → 2022 (46 catchers) | 0.684 | 0.637 | [−0.007, +0.101] |
| Year over year, 2022 → 2023 (47) | 0.545 | 0.544 | [−0.074, +0.085] |
| Year over year, 2023 → 2024 (48) | 0.596 | 0.620 | [−0.107, +0.070] |
| Year over year, 2024 → 2025 (46) | 0.569 | 0.617 | [−0.110, +0.024] |
| vs Savant, 2021 (59 catchers) | 0.892 | 0.914 | [−0.068, +0.014] |
| vs Savant, 2022 (60) | 0.952 | 0.961 | [−0.027, +0.009] |
| vs Savant, 2023 (63) | 0.942 | 0.936 | [−0.016, +0.036] |
| vs Savant, 2024 (58) | 0.942 | 0.917 | [−0.003, +0.055] |
| vs Savant, 2025 (57) | 0.955 | 0.647 | [+0.208, +0.427] |

Every interval but the last includes zero, as do the six season pairs more than a year apart. These are paired catcher-bootstrap intervals with 8,000 resamples, conditional on the fitted catcher estimates ([`models/validate.py`](models/validate.py), [`results/year_over_year.csv`](results/year_over_year.csv), [`results/vs_savant_by_season.csv`](results/vs_savant_by_season.csv)). They do not propagate baseline or hierarchical estimation uncertainty, and shared games and model parameters can leave dependence between catchers (METHODS §2.2). The 2021–2022 estimates use cross-fitted baselines; 2023–2025 use the fixed train-only model.

The simulation separates the two estimators clearly on intervals (74% coverage against 93%) and more modestly on point estimates. The external checks only see point estimates. In the first eight rows they do not favor either estimator. Most of those intervals are wide: ±0.05 to ±0.09 for year over year and ±0.04 for Savant 2021, which leaves room for sizable differences either way. The Savant intervals for 2022–2024 are narrower, at ±0.02 to ±0.03, and each rules out a difference larger than about 0.06 in either direction; smaller ones remain possible. 2022 leans slightly toward the unadjusted estimator, 2023 and 2024 slightly toward the hierarchical one.

The 2025 row separates them, but not for the reason the simulation studies. The baseline was fit on 2021–2022 and never refit, and by 2025 it overpredicts called strikes by 2.6 points across the whole season (0.358 against 0.332; in 2023 and 2024 the gap is 0.6 to 0.8). The called zone narrowed in raw `plate_x`, and Statcast's `sz_top` rose. The overall log loss gave no warning, though log loss is not comparable across seasons anyway (METHODS §4.1).

Unadjusted runs multiply the mean residual by pitch count and 0.125, so average baseline error creates a volume-related term: in 2025, unadjusted runs correlate −0.60 with shadow-zone pitch count, versus +0.18 for Savant. The hierarchical intercept and slope can absorb the global calibration component, without guaranteeing correction of catcher-specific errors. Re-estimating those two parameters on the 2025 shadow zone raises unadjusted runs’ Savant correlation to 0.964. A logit-intercept correction is not a uniform probability shift: in 2025, catcher-specific per-pitch corrections range from 10.08 to 11.97 percentage points. Its measured effect on adjacent-season correlations is small: 2024→2025 changes from 0.61657 to 0.61609 with intercept recalibration, or 0.61616 with intercept and slope ([`results/per_pitch_recalibration.csv`](results/per_pitch_recalibration.csv)). These are post-hoc diagnostics using the observed seasons, not new held-out validation ([`results/baseline_transport.csv`](results/baseline_transport.csv), [`results/baseline_drift.csv`](results/baseline_drift.csv)).

What the row shows is narrow. The unadjusted estimator has nowhere to put a calibration error in its baseline, and a baseline three seasons old costs it 0.31 in agreement with Savant. It does not show that the hierarchical model measures framing better. Refitting the baseline every season, which is the usual practice, would most likely have prevented the drop.

So the external checks still cannot do what the simulation does. As long as the baseline is calibrated, they see only the part where the two estimators differ least, and mostly without enough precision to see even that. This is also why a high correlation with Savant is not validation: a check that cannot tell a good estimator from a bad one says nothing about which one you have.

### 8. Reliability and persistence

The split-half reliability is carried over from v1 and was not re-examined in this round. Splitting each catcher's pitches at random into halves gives r = 0.82 (mean of 50 splits), or R = 0.90 after correcting back to full-season length with Spearman–Brown. It uses the unadjusted estimator, as a rate per pitch, rather than the hierarchical estimates, so that half-seasons and full seasons stay comparable without refitting the mixed model each time. It also uses v1's in-sample baseline, which section 3 shows can move a correlation.

Persistence has been redone. With 2024 and 2025 in, the single-season estimates behind sections 4 and 7 give ten pairs of seasons, all out of sample and on the shadow zone, among catchers with at least 300 shadow-zone pitches in both seasons ([`results/year_over_year.csv`](results/year_over_year.csv)). Hierarchical estimates:

| Gap | Pairs | Mean |
|---|---|--:|
| 1 year | 2021→22 0.68, 2022→23 0.54, 2023→24 0.60, 2024→25 0.57 | 0.60 |
| 2 years | 2021→23 0.22, 2022→24 0.54, 2023→25 0.59 | 0.45 |
| 3 years | 2021→24 0.09, 2022→25 0.47 | 0.28 |
| 4 years | 2021→25 0.19 | 0.19 |

The unadjusted estimator gives nearly the same numbers, as section 7 found (one-year mean 0.60).

Consecutive seasons correlate at about 0.60. Beyond one year, the picture depends on 2021. Of the two-year pairs, 2021 → 2023 is the lowest at 0.22; the other two sit at 0.54 and 0.59, about where the one-year pairs do, and 2022 → 2025 is still 0.47. Leaving 2021 aside, the correlations barely decay, which looks more like a stable trait measured with noise than like a trait that drifts each year. With 2021 in, they decay. Each pair has only 30 to 48 catchers, so a correlation of 0.5 carries an interval of roughly ±0.25, and the catchers who last four seasons are not a random sample. The data do not settle which reading is right.

---

## 2023 leaderboard

Framing runs over shadow-zone pitches, with 95% credible intervals, next to Savant's published figure for the same season, ranked by framing runs. The denominators differ: this table counts only the 15% of pitches where the call was in doubt.

| Catcher | Shadow pitches | Framing runs | 95% interval | Savant |
|---|--:|--:|---|--:|
| Austin Hedges | 791 | +11.1 | [+7.5, +14.6] | +14.5 |
| Francisco Álvarez | 1,088 | +9.2 | [+5.1, +13.1] | +13.9 |
| Jonah Heim | 1,181 | +7.1 | [+2.6, +11.6] | +11.9 |
| Patrick Bailey | 965 | +6.7 | [+3.1, +10.3] | +17.0 |
| Cal Raleigh | 1,180 | +6.3 | [+1.8, +10.6] | +6.3 |
| … | | | | |
| Keibert Ruiz | 1,410 | −6.6 | [−11.2, −1.9] | −11.9 |
| J.T. Realmuto | 1,445 | −6.7 | [−11.9, −1.7] | −14.4 |
| Martín Maldonado | 1,154 | −7.4 | [−11.7, −3.0] | −15.7 |

The ends of the list are firmer than the middle: 40 of the 63 qualified catchers have intervals that include zero.

---

## From v1 to v2

This project started with a podcast episode in which a Taiwanese data scientist working in an MLB front office mentioned that catcher framing was the first project he was handed there. So I tried it.

The first version measured framing in two steps. It fit a strike-probability model on location and context but *not* catcher identity, and treated its prediction as the expected call for a pitch of that description. Then it let catcher, umpire, and pitcher effects compete for the residual. The result was a leaderboard, three variance components, and a correlation of 0.990 with Baseball Savant's published framing runs.

This second version checks whether those numbers support what the first version said about them. The variance-component ordering does not, and the leaderboard only partly does. In both cases the reason is the same: every number in v1 was a point estimate with no interval, and its one synthetic-data test checked that effects landed on the right groups, not how the estimator behaves. The correlation still reproduces, but it shows less than it seems to.

The original analysis is unchanged and still available at tag [`v1.0`](../../tree/v1.0). This round tested four things v1 said or relied on.

| v1 | Verdict |
|---|---|
| Umpire-to-umpire variation exceeds catcher-to-catcher variation (τ 0.233 vs 0.192) | **Leans that way, but not established.** The umpire component is the larger one in four of five seasons, at posterior probability 0.81 on 2023, but no season reaches 0.95. One season pins the difference only to within about ±0.05 on the logit scale, and the differences themselves are at most 0.04, so the 2022 flip and the movement across shadow-zone thresholds are within that noise rather than evidence against |
| The leaderboard ranks catchers on point values alone | **Partly supported.** 23 of 63 qualified catchers have intervals excluding zero; most adjacent ranks are coin flips |
| r = 0.990 against Savant shows the location model is sound | **The correlation does not establish this.** Section 2 supplies out-of-sample predictive and calibration checks, with limitations. One re-estimated intercept restores the Savant correlation to 0.989; that agreement cannot assess interval calibration |
| The variational fit behind all of this had not converged | **Similar ordering in the tested subset.** On the same 2022 shadow-zone data, per-catcher VB and NUTS effects correlate at r = 0.9999. This does not establish equal magnitudes or make every unconverged v1 fit harmless |

On the umpire–catcher ordering, v1's numbers were not wrong; they reproduce closely on 2023 (section 4). The problem is that v1 stated the ordering as a finding about baseball, when in one of the three seasons it had, the ordering is close to a coin flip. The two later seasons make the lean more consistent without making it decisive.

v1 also reported its baseline's fit statistics in sample. On the split in section 2 that did not visibly inflate the fit, but that does not mean none of v1's estimates were affected. The in-sample baseline is the first row of the table in section 3.

v1 read its three seasons as AR(1). Consecutive seasons correlated at 0.603 and a two-year gap dropped to 0.320, close to 0.603² = 0.364, so framing looked like a trait that drifts a little each year. Squaring the one-year correlation ignores measurement error, which makes the predicted two-year figure too low, so that check was tilted toward seeing drift. The two-year figure also rested on a single pair, 2021 → 2023, which with more seasons turns out to be the lowest of three (section 8). v1's in-sample 2022 → 2023 figure of 0.599 comes out at 0.54 out of sample.

The names at both ends of the 2023 leaderboard match v1: Hedges, Álvarez and Bailey led v1's 2023 table too, and Maldonado and Ruiz were two of its bottom three. v1's pooled 2021–2023 fit (Jose Trevino leading at +40 runs over three years), its persistence matrix and the catcher trajectories all use v1's in-sample baseline, and they remain in [v1's README](../../blob/v1.0/README.md#6-three-seasons).

---

## What I'd do differently

I started this round for the wrong reason. What bothered me about v1 was that its hierarchical model was fit by variational Bayes and had not converged. Of everything on my list, that was the one item that mattered least: on the 2022 data where I checked it, NUTS ranked catchers almost identically to VB. The bigger gap, that nothing had an interval and nothing checked whether the estimator could be trusted, was on my own list of v1's problems. I had ranked it below the convergence issue.

Twice on the first day of this round I nearly acted on a bad number. A single 971-game validation split showed a shadow-zone bias significant at p ≈ 0.03; five-fold cross-fitting over all 4,856 games found no sign of it (z = +0.30), and I had been about to rewrite the calibration pipeline around it. The same day, my first timings said a 4-chain NUTS run finished in two seconds. JAX had returned before the computation was done, so I was timing dispatch, not the fit. Later, a simulation scenario that looked like it degraded coverage at 10 replications was back at nominal at 100. Each time, the number pointed where I already wanted to go.

Three of my simulation designs tested almost nothing. A location-varying catcher effect does little damage when every catcher faces the same distribution of locations: the model is still misspecified, but it estimates each catcher's average effect almost as well; that one stays in the eight as a control. An omitted covariate defined at the pitcher level is absorbed whole by the pitcher random effect, and one drawn fresh for every pitch is unrelated to the catcher and only adds noise. Each time I had designed something that sounded like a misspecification without checking whether the model could largely absorb it. I now work that out before writing the generator.

I kept a dated working log through this round, wrong turns included. It is not published; the technical lessons are in METHODS §8.

---

## Postscript: the ABS era

v2 ends with 2025. The ABS challenge system that went live in 2026 makes a new question answerable with real data: once any call can be challenged, do umpires' calls still vary with the catcher? The full analysis is in [ABS.md](ABS.md). It uses a different pipeline from everything above, so the same season can carry a different τ there.

Two things had to change first. Statcast records the call after any challenge and does not mark the overturned ones, so the umpire's original call was recovered from the challenge records in MLB's game feed. Every one of the 7,891 challenges on called pitches matched, with the final call agreeing each time, and the 4,429 overturned among them were flipped back. And 2026 redefined the strike zone from the batter's height, so every season was converted to that definition, with the baseline model refit inside each season.

<p align="center">
  <img src="docs/images/en/abs_tau_by_season.png" width="560">
</p>

The result does not meet the prespecified threshold for being lower than **every** baseline season. 2026 has the lowest posterior mean catcher variation of the six seasons (τ = 0.155). Under independent marginal posteriors, its probability of lying below 2022, 2023 and 2024 exceeds 0.95, but below 2021 is 0.821; the joint probability of lying below all four is 0.805. Failure of that rule does not establish no change. The decline predates ABS, and the 2025 comparison remains inconclusive. With every team switching at once and no control group, this design cannot attribute the change to ABS or separate changes in catchers from changes in umpires.

Two questions are left for next: how much framing value survives the challenges, and whether challenging is itself a catcher skill.

---

## Reading this repository

1. This README: the question, the findings and the evidence for them.
2. [`notebooks/00_results_tour.ipynb`](notebooks/00_results_tour.ipynb): the headline results read straight from the CSVs in `results/`. It downloads nothing and fits nothing, and runs in seconds.
3. [METHODS.md](METHODS.md): the estimand, identification, the simulation design, and where the numbers are weaker than they look.
4. [`results/README.md`](results/README.md): which CSV backs which number.
5. [ABS.md](ABS.md): the 2026 postscript.

Notebooks 01–06 are v1's analysis. They still run and are kept so v1 can be compared with this version, but some of their conclusions are stronger than the evidence supports. Where they disagree with this README or METHODS, the README and METHODS are current.

## Methods overview

| Stage | What | Module |
|---|---|---|
| Data pipeline | Monthly Statcast fetch, cleaning, standardization | [`data/fetch.py`](data/fetch.py) |
| Split | Train/validation by `game_pk`, 2023 locked | [`models/splits.py`](models/splits.py) |
| Baseline | Logistic GAM, fit on train, scored out of sample | [`models/baseline_v2.py`](models/baseline_v2.py) |
| Cross-fitting | Out-of-fold baseline probabilities for the train pool | [`models/crossfit.py`](models/crossfit.py) |
| Hierarchical model | Crossed random effects, NUTS on the shadow zone | [`models/hierarchical_v2.py`](models/hierarchical_v2.py) |
| Engine comparison | VB vs NUTS, season stability | [`models/compare_engines.py`](models/compare_engines.py) |
| Intervals | Δ posterior, pairwise comparison probabilities | [`models/intervals.py`](models/intervals.py) |
| Simulation | Eight scenarios, two estimators, four metrics | [`sim/`](sim/) |
| External validation | Year over year for every pair of seasons 2021–2025, Savant comparison by season | [`models/validate.py`](models/validate.py) |
| Sensitivity | `b` free vs fixed at 1, and the three shadow-zone thresholds, on the train pool | [`models/sensitivity.py`](models/sensitivity.py) |
| Evaluation seasons | primary 2023 fit run once (the identification diagnostic refits it separately); 2024 and 2025 one fit each; the baseline's calibration on each | [`models/holdout.py`](models/holdout.py) |

v1's modules (`baseline_gam.py`, `framing_runs.py`, `hierarchical.py`, `reliability.py`, notebooks 01–06) still run. Their code is unchanged; only docstrings and a header note at the top of each notebook were updated to point to the v2 corrections.

## Reproduce

Environment is managed with [uv](https://docs.astral.sh/uv/). The full command sequence and project structure are in [METHODS §9](METHODS.md#9-reproducibility); the postscript's commands are in [ABS.md §7](ABS.md#7-reproduce). Every published table is also written to a CSV under [`results/`](results/), so it can be checked without refitting anything; [`results/README.md`](results/README.md) says which file backs which number.

## Limitations

- **Associational, not causal.** The catcher term is variation associated with the catcher under this specification. Section 6 measures how far unmeasured confounding would throw off the intervals; nothing in the data can rule it out.
- Pitch type, velocity, movement, batter identity and ballpark are not modeled. Savant adjusts for park; this project does not. Pitchers enter as a random effect, but low-volume pitchers share a single pooled effect (see below).
- The count enters the baseline additively on the logit scale, with no count × location interaction. Each count moves the 50% contour to a different level of the same location surface, which can change its shape somewhat, but every count shares that one surface. The count-specific reshaping shown by fitting each count separately is not in the model that produces the framing numbers.
- Pitches with |plate_x| > 2.5 ft, or outside a standardized height of [−1, 2], are dropped before modeling to keep the spline from extrapolating: about 2% of pitches, of which exactly 1 of 7,386 in 2023 was a called strike. They carry essentially no framing signal.
- Pitchers with fewer than 100 shadow-zone pitches in the data being fitted share a single pooled effect: 749 of 1,123 pitchers (25% of pitches) in the 2021–2022 fit, and 47–49% of pitches in each of 2023, 2024 and 2025 (669 of 835 pitchers in 2023). Their individual effects would be heavily shrunk anyway, but for those pitches the model makes no pitcher-specific adjustment. (v1 pooled pitchers below 100 called pitches per season, and below 150 in its three-season fit.)
- Run value is a flat 0.125 runs per stolen strike, as in v1. The real value depends on the count, and a strike stolen with two strikes already on the batter is worth far more than one stolen on 0-0, so per-catcher totals are approximate. A count-dependent value would leave the variance components, the separability figures and P(τ_umpire > τ_catcher) unchanged, since they are on the probability scale. It could reorder the run leaderboard, though, because catchers see different mixes of counts. That has not been checked.

## References

- [Pavlidis, H. & Brooks, D. (2014). *Framing and Blocking Pitches: A Regressed, Probabilistic Model*. Baseball Prospectus.](https://www.baseballprospectus.com/news/article/22934/)
- [Judge, J., Pavlidis, H. & Brooks, D. (2015). *Moving Beyond WOWY: A Mixed Approach to Measuring Catcher Framing*. Baseball Prospectus.](https://www.baseballprospectus.com/news/article/25514/)
- [Albert, J. (2023). *Called Strikes*.](https://bayesball.github.io/BLOG/Called_Strikes.html)
- Deshpande & Wyner (2017), *A Hierarchical Bayesian Model of Pitch Framing*, JQAS.
- [Baseball Savant catcher framing leaderboard](https://baseballsavant.mlb.com/catcher_framing) (methodology notes).

## Tech stack

Python 3.12 · polars · pandas · pybaseball · pyGAM · statsmodels · numpyro/JAX · scikit-learn · matplotlib · uv

## License

[MIT](LICENSE). Statcast data is retrieved from Baseball Savant and is subject to MLB's terms. No pitch-level data is redistributed here; the CSVs in `results/` include Savant's published framing runs for comparison.
