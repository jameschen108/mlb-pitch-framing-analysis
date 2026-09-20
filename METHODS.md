# Methods

**English** | [繁體中文](METHODS.zh-TW.md)

Technical companion to [`README.md`](README.md). What is being estimated, how, what this design can and cannot identify, and where the numbers stop meaning what they appear to mean.

---

## 1. What is being estimated, and how

These are two separate things, and conflating them is what made the first version of this project hard to check.

**The estimand.** For catcher *c*, the average change in called-strike probability over the pitches he actually received, from replacing a league-average receiver with him:

```
Δ_c = mean over c's pitches of [ P(strike | c receives) − P(strike | average receiver) ]
```

Δ is defined per pitch and reported per 100 shadow-zone pitches, or summed and multiplied by 0.125 runs for a leaderboard figure.

Three properties matter. It is defined **relative to the league average**, not to zero, because a random-effects model identifies catcher effects only up to a constant — the global level is absorbed by the intercept. It is averaged over **that catcher's own pitches**, so two catchers facing different pitch mixes are each scored on what they actually saw. And it is on the probability scale, not the logit scale.

**The estimators.** Two, both targeting Δ:

- **Residual runs** (the v1 approach): `Δ̂_c = mean(actual − baseline predicted)` over that catcher's pitches. No adjustment for umpire or pitcher. Intervals from a binomial standard error.
- **Hierarchical model**: crossed random intercepts for catcher, umpire and pitcher on the residual, with the baseline logit entering as a calibration covariate whose coefficient is estimated, not fixed at 1. Δ is computed per posterior draw and summarised.

Choosing Δ rather than the logit-scale coefficient `u_catcher` is deliberate: residual runs has no `u_catcher`. Comparing the two estimators on `u` would compare different quantities. Δ is what both estimate, and it is what the leaderboard reports, so the simulation in section 6 tests exactly the number that gets published.

---

## 2. Data and design

2021–2023 regular seasons, called pitches only: 1.06M, 1.04M after dropping pitches beyond |plate_x| > 2.5 ft or standardized height outside [−1, 2] to keep the spline from extrapolating. Statcast via `pybaseball`; home-plate umpires from the MLB Stats API joined on `game_pk`. Heights standardized as `(plate_z − sz_bot) / (sz_top − sz_bot)`.

### 2.1 Out-of-sample baseline probabilities

What the whole measure rests on is the residual `actual − predicted`. A baseline model fitted on the pitches it scores flattens those residuals by construction. Every baseline probability in this project comes from a model that never saw the pitch:

- **2021–2022**: five-fold cross-fitting, folds assigned by `game_pk`. Five GAM fits, each predicting the fold it did not see.
- **2023**: predicted by a model fitted on the 2021–2022 training split only.

The two schemes differ in form; both are out of sample.

The cost of not doing this is measurable. On 2023, the same unadjusted estimator correlates with Savant at 0.990 using an in-sample baseline and 0.958 using an out-of-sample one. The in-sample version shares structure with Savant — which presumably also fits within season — and the correlation is inflated accordingly.

### 2.2 Splitting by game, not by pitch

Train/validation split for model selection is on `game_pk`, not on individual pitches. Pitches within a game share an umpire, a park, and that day's zone. Splitting by pitch puts correlated observations on both sides and makes validation loss optimistic — which defeats the purpose of splitting at all. The cost is that the split cannot land on exactly 20%; it landed on 19.93%.

The same reasoning governs resampling: where a bootstrap is used, the unit is the game, not the pitch. Clustering widens standard errors by a factor of **1.21 to 1.29** on shadow-zone residual sums — smaller than the textbook warning suggests, but not negligible.

One case takes a different unit. Where two estimators are compared against the same external target (§7 of the README), the quantities being correlated are already one number per catcher, and the game-level clustering is absorbed in the step that produced them; there the resampling unit is the catcher, and the resample is paired across the two estimators — `paired_bootstrap_diff` in [`models/validate.py`](models/validate.py). Resampling the two estimators independently would count the "which catchers are in the sample" variance twice and widen the interval, which here would have argued for the conclusion I already expected.

### 2.3 The shadow zone

**The term is borrowed; the definition is not.** Statcast's Shadow Zone is geometric — a band one ball-width either side of the rule-book strike zone. The zone used throughout this project is *model-defined*: the pitches the fitted baseline puts at 0.2 < p̂ < 0.8. The two overlap heavily but are not interchangeable, and in particular they do not share a denominator, so every figure here that sits beside a Savant figure is computed over a different set of pitches. Read "shadow zone" below as shorthand for the model-defined band.

Analysis is restricted to pitches the baseline model puts at 0.2 < p̂ < 0.8. Two justifications, in order of importance.

**Substantive**: framing can only operate where the call is in doubt. A pitch down the middle is a strike regardless of the receiver.

**Statistical**: the discarded pitches carry almost no information. Fisher information about a catcher's effect scales with p(1−p), which is near zero at the extremes. Summing over 2023:

| | Share of pitches | Share of information |
|---|--:|--:|
| Shadow zone (0.2–0.8) | 14.5% | **60.8%** |

Across the 75 catchers with at least 1,000 called pitches, the standard error of the catcher effect inflates by a factor of 1.27 on average — not the 2.6 that the raw pitch counts would suggest.

**Computational cost is a third consideration and a weak one.** Measured at matched iteration counts, NUTS costs 11.4× more per iteration on a full season than on the shadow zone, which puts a full-data fit at roughly an hour rather than out of reach. What the restriction buys is cheap *refitting* — a pooled two-season fit takes about seven minutes, and the threshold sensitivity checks, the engine comparison and the simulation all require many fits rather than one.

**Circularity**: the shadow zone is defined by the baseline model, so the baseline must be fit on *all* taken pitches first, and the subset taken afterwards.

**Sensitivity.** All three thresholds were committed to in advance, along with a commitment not to select whichever produced the narrowest intervals. Refitting the train pool at each ([`models/sensitivity.py`](models/sensitivity.py), [`results/sensitivity_threshold.csv`](results/sensitivity_threshold.csv)):

| Threshold | Pitches | Catchers | τ catcher | τ umpire | P(τ_u > τ_c) | Intervals excluding zero | Mean interval width | r vs 0.2/0.8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.15 / 0.85 | 130,159 | 149 | 0.1859 | 0.1860 | 0.50 | 37 (24.8%) | 5.61 runs | 0.991 |
| **0.20 / 0.80** | 103,615 | 148 | 0.1883 | 0.1857 | 0.46 | 34 (23.0%) | 5.08 runs | — |
| 0.25 / 0.75 | 81,971 | 148 | 0.1802 | 0.1880 | 0.63 | 24 (16.2%) | 4.41 runs | 0.987 |

**The leaderboard is stable; the variance-component ordering is not.** Per-catcher framing runs correlate at 0.991 and 0.987 against the reported band, so nothing on the leaderboard turns on where the band is drawn. But P(τ_umpire > τ_catcher) reads 0.50, 0.46 and 0.63 across the three — a second reason, independent of the season-to-season flipping, not to state that ordering as a fact about baseball. Two arbitrary analyst choices move it, and neither moves it far enough to settle it.

**The commitment is checkable, and it held.** The narrowest intervals are at 0.25/0.75 (4.41 runs), not at the reported 0.20/0.80 (5.08). Nor is the reported band the one that separates the most catchers from zero: 0.15/0.85 does, at 24.8% against 23.0%. Interval width in runs scales with the number of pitches in the band, so a narrower band buys narrower intervals and smaller run totals at once; the measure that matters is the share of catchers resolved, and on that the reported band sits in the middle of the three.

### 2.4 Holdout discipline

2023 was locked for the duration of v2's development. Model form, shadow-zone threshold, inference engine, estimand definition and the list of reported quantities were all settled on 2021–2022. 2023 was used once, at the end, so the new figures could be set against v1's published 2023 table on the same season.

**It is not a holdout in the strict sense, and the distinction is worth being exact about.** v1 analysed 2023 and published a leaderboard on it, and comparing against that published table is precisely why this round spends the season. So 2023 was never unseen data: it was seen in v1, and the research question in v2 was shaped by what v1 found. What the discipline does buy is that no v2 decision — not the threshold, not the engine, not the estimand, not the list of reported quantities — was tuned against 2023. That is a locked evaluation set for the v2 cycle. It is not an untouched holdout, and calling it one would claim more than the design supports.

This cost something and the cost should be stated: with 2023 reserved, year-over-year stability rests on a single season pair, so there is no decay curve of the kind v1 reported.

---

## 3. Identification

Random effects handle small-sample instability. They do not handle confounding.

### 3.1 Catcher and umpire separate cleanly

Assignment of umpires to games is close enough to random, and the schedule mixes enough, that the two effects are well separated. On 2023 shadow-zone pitches:

- 102 catchers × 94 umpires = 9,588 possible pairs; **3,811 observed (39.7%)**
- Among the 66 catchers with ≥300 shadow-zone pitches, the median catcher worked with **50 distinct umpires**; the least-exposed worked with 27

That is dense crossing. `u_catcher` and `v_umpire` are identified.

### 3.2 Catcher and pitcher do not

The confounding runs in one direction, and it is not the direction one first looks.

- A catcher's pitches coming from his single most frequent pitcher: median **14.9%**, 90th percentile 22.4%. Catchers see many pitchers.
- A pitcher's pitches caught by his single most frequent catcher: median **60.5%**, 90th percentile **84.8%**.

A pitcher is mostly caught by one catcher. So whatever is specific to a pitcher and not in the model gets a clean channel into that catcher's estimate.

This shows up directly in the posterior. Taking each qualified catcher and the pitcher he caught most, and correlating their effects across posterior draws (66 pairs, pooled group excluded):

| | ρ |
|---|--:|
| Median | **−0.221** |
| 10th percentile | −0.304 |
| Most negative | −0.374 |
| Share below −0.2 | 63.6% |

The negative sign is the signature of partial non-identification: within the posterior, credit for the same calls trades off between the catcher and his battery-mate. The model cannot tell whose it is; it splits the difference and widens both intervals.

**So the catcher term is variation associated with the catcher under this specification, and the pitcher channel is the main reason it cannot be read as skill.** The simulation in section 6 puts a number on what that costs.

### 3.3 What shrinkage is and is not

Partial pooling addresses the fact that a catcher with 150 shadow-zone pitches has a noisy raw estimate. It does not address the fact that his pitches came disproportionately from two pitchers. Shrinkage makes small-sample estimates less wrong about themselves; it does nothing about a systematic channel. The two failures look similar in a leaderboard and have nothing to do with each other.

---

## 4. Estimation

### 4.1 Baseline

```
logit P(strike) = te(plate_x, plate_z_std) + f(stand) + f(p_throws) + f(balls) + f(strikes)
```

Logistic GAM (pyGAM), tensor spline with 20 splines per margin. Model selection on validation log loss, not accuracy and not AUC: the framing metric is built on probability values, so calibration is what matters and AUC is insensitive to it.

In-sample and out-of-sample log loss differ by 0.002 (0.17346 vs 0.17149). With 410 basis functions against 550,000 rows and a penalty term there was no room to overfit, so v1's in-sample reporting was a methodological flaw that did not distort its fit statistics.

Calibration is the real weakness. Out-of-fold, the fitted surface over-predicts strikes below p̂ = 0.5 and under-predicts above it, monotonically across six bins on 100,000 held-out pitches — a genuine S-shape, from over-smoothing the transition band. Recalibrating isotonically moves catcher framing runs by **0.32 runs from end to end** against a leaderboard spread of 28.5. Real, and below the level at which it would change a conclusion.

### 4.2 Hierarchical model

```
logit P(strike) = a + b·logit_base + u_catcher + u_umpire + u_pitcher
u_g = τ_g · z_g,    z_g ~ N(0, 1),    τ_g ~ HalfNormal(0.5)
a ~ N(0, 2),        b ~ N(1, 1)
```

**Two-stage, but not an offset model.** The baseline logit enters the second stage as a covariate whose coefficient `b` is estimated under a N(1, 1) prior — not held at 1, which is what the word *offset* means. The distinction is not cosmetic. Under a true offset the second stage scores the first stage's predictions exactly as they come; here `b` is free to rescale them, absorbing the slope component of the §4.1 miscalibration before the random effects ever see the residual. Fit freely on the train pool, `b` has posterior mean **1.089**, 95% interval **[1.071, 1.107]**, with P(b > 1) = 1.000. The interval excludes 1 outright, so this is not a case where an offset would have been an adequate approximation loosely described — the data reject it. (v1's VB fit records 1.031 on full-season 2023 and 1.026 on the pooled three seasons, [`results/v1_fit_summary.csv`](results/v1_fit_summary.csv); a different sample and a different engine, not a contradiction.)

**Fixing `b = 1` changes the wording and nothing else.** Refitting the same pitches with the coefficient pinned at 1 — the offset model this document once described — leaves per-catcher framing runs correlated with the free fit at r = 0.9997 (Spearman 0.9991). The largest single-catcher shift is **0.40 runs against a leaderboard spread of 28.9**, the mean shift is 0.06, the top ten are the same ten, and the largest rank change among 148 catchers is 12 places. All three variance components move by under 0.005, and P(τ_umpire > τ_catcher) goes from 0.4635 to 0.4615. So the assumption is false, and the estimator does not depend on it. Both fits are in [`models/sensitivity.py`](models/sensitivity.py), summarised in [`results/sensitivity_slope.csv`](results/sensitivity_slope.csv).

**The sensitivity is run on the 2021–2022 train pool, not on 2023.** 2023 is a locked evaluation set already spent once (§2.4); refitting it for a robustness check would spend it a second time.

Pitchers with fewer than 100 pitches in a season are pooled into a single group; their individual effects would shrink to near zero regardless and the level count is otherwise unmanageable.

**Non-centred parameterisation is not optional.** When τ is small the centred form has a funnel-shaped posterior and NUTS produces divergences. Written non-centred from the start, all fits in this project returned zero or near-zero divergences (2 across 800,000 draws in the simulation study).

**The engine swap changes nothing, and that is the finding.** On identical data, variational Bayes and NUTS agree on per-catcher effects at r = 0.9999 (Spearman 0.9998), and VB's posterior standard deviation is 0.95× NUTS's. The theoretical warning that variational inference understates posterior variance is correct in direction and worth 5% here. What NUTS supplies is posterior *samples*, which pairwise comparison probabilities and coverage checks require and VB cannot provide.

---

## 5. Uncertainty

**Δ posterior.** For each posterior draw, η is reconstructed for every pitch, Δ is computed as `σ(η) − σ(η − u_catcher)`, and averaged within catcher. The result is a posterior distribution over Δ per catcher; the interval is its 2.5/97.5 percentiles.

**Runs.** `runs = Δ × shadow-zone pitches × 0.125` is computed **per draw**, then summarised. Multiplying a point estimate and attaching an interval afterwards would understate the uncertainty.

**Pairwise comparisons.** `P(Δ_A > Δ_B)` is counted directly from the joint posterior draws, which is the only way to answer "is A better than B" without pretending the two estimates are independent.

**Separability.** On 2023: 27 of 102 catchers (26%) have intervals excluding zero; 27% of pairs are resolved at 95%. Restricting to catchers with ≥1,000 shadow-zone pitches raises these to 53% and 55%. On 2021–2022, P(rank 1 > rank 2) = 0.77.

**Coverage at the extremes is lower than nominal.** Simulation shows coverage for the largest third of effects running 3 to 6 points below overall coverage, in every scenario including the unconfounded baseline — 89.0% against 94.7% there. The estimator is doing this, not the data: partial pooling buys its stability at the tails, which is where a leaderboard is read. The caterpillar plot carries the caveat as a caption.

---

## 6. Simulation design

Simulation is the only check in this project where the truth is set rather than inferred. Its conclusions are reported in the README; this section covers how it is built and what it does not establish.

### 6.1 Scale

Synthetic datasets use **30 catchers × ~500 pitches each**, about 15,000 rows, against a real analysis set of 50,000–100,000. This is a compute trade-off, not a statistical choice: a single fit takes 9 seconds at this size, making 800 fits feasible. **Coverage depends on sample size, so these figures should not be read as exact for full-season samples.**

Pitch-location probabilities and the inequality of catcher workloads are drawn from the real 2022 shadow zone rather than from a convenient parametric distribution. The real inequality — starters at an order of magnitude above backups — is one of the things being tested.

### 6.2 Scenarios

Eight, in two groups. The first four manipulate data structure; the last four attack the model's assumptions.

| Scenario | Manipulation |
|---|---|
| baseline | Umpires assigned at random |
| unequal | Workloads from 30 to 1,561 pitches |
| umpire_confound | Some catchers systematically draw permissive umpires |
| battery | Catcher–pitcher pairings concentrated |
| location_shared | Catcher effect varies with location; locations independent of catcher |
| location_mix | As above, but catchers face different location distributions |
| heavy_tail | Catcher effects from t(3), violating the normal prior |
| omitted_covariate | A variable affecting calls, correlated with catcher, absent from the model |

**`location_shared` is a deliberately preserved failure.** It looks like misspecification and is not: when locations are drawn independently of catcher, every catcher faces the same distribution, the interaction term averages to the constant `γ_c · E[centred]`, and a constant-intercept model recovers it exactly. The catcher-level spread in mean location is 0.017 under this scenario against 0.078 under `location_mix`. It is kept in the table as the control it turned out to be.

Two further scenario designs failed before `omitted_covariate` worked, and the reason is the same each time: **the fitted model absorbed what was meant to break it.** A covariate defined at pitcher level is taken up whole by the pitcher random effect. Per-pitch noise is uncorrelated with catcher and merely adds unexplained variance. Only a covariate with a catcher-level component — with the truth defined to exclude that component — bites.

That last construction is close to tautological: removing from the truth something the data cannot separate guarantees that coverage fails. It is included because the size of the failure is the useful part, and because it makes explicit that good coverage elsewhere does not license a causal reading.

### 6.3 Metrics

Bias, RMSE, 95% interval coverage, and rank recovery (Spearman against the true ordering), for both estimators, 100 replications per scenario. Monte Carlo error on coverage is ±1.1%.

**Coverage is also reported stratified by effect magnitude.** Under heavy tails only one or two of thirty catchers are outliers, so missing both moves aggregate coverage by six points and disappears into the average. The failure is at the tail; the aggregate hides it.

### 6.4 The confounding sweep

Rather than choose one confounder size and report whether it broke, the strength of the catcher-correlated confounder is swept from zero to twice the true catcher effect, at 50 replications per point. This converts "does unmeasured confounding matter?" — which has an arbitrary answer — into "how much would it take?", which does not.

### 6.5 What the simulation does not establish

The generating process is the model's own functional form in five of eight scenarios. Good coverage there is close to guaranteed and should not be read as validation. The three scenarios built to break the model did not break it, which bounds the claim: the intervals survive every misspecification that could be constructed here, not every misspecification.

---

## 7. What the numbers here bound

The full list of limitations is in [`README.md`](README.md). Three of them can be given
figures, and those figures are what this section is for. The rest are scope statements that need no elaboration beyond
the README's: unmodelled pitch characteristics, a flat run value, the coordinate trim.

**Catcher and pitcher are partially inseparable.** A pitcher's pitches are caught by his
most frequent catcher a median 60.5% of the time, 84.8% at the 90th percentile. The
posterior correlation between a catcher's effect and his most-caught pitcher's effect has
median ρ = −0.221, with 63.6% of pairs below −0.2. The catcher term is association, not
skill, and this is the channel that makes it so.

**Coverage at the extremes is about 88%, not 95%**, in every simulated scenario,
including the unconfounded baseline. Shrinkage is doing this, so it is a property of the
estimator rather than of this dataset, and it applies exactly where a leaderboard is read.

**Unmeasured catcher-correlated confounding costs coverage in a measurable way:** 94% with
none, 88% when it matches the size of the effect being measured, 66% at twice. Section 3.2
establishes that this structure is present in the real data. Nothing in the data bounds
where on that curve the real analysis sits, and no amount of better inference would change
that. It is a design property, not an estimation one.

---

## 8. Pitfalls encountered

Recorded because each cost time and each would have produced a wrong number if it had gone unnoticed.

**A single validation split is not enough to detect a bias of this size.** One 971-game split showed a shadow-zone residual bias significant at p ≈ 0.03 across all three thresholds. Five-fold cross-fitting over all 4,856 games put it at z = +0.30, covering zero. The first result was a false positive from running one 5% test against three highly correlated thresholds. This is also a caution about v1's other single-split figures.

**Timing a JAX program without blocking measures dispatch, not computation.** `mcmc.run()` returns before the samples exist; they materialise when first accessed. `jax.block_until_ready(mcmc.get_samples())` is required before stopping the clock.

**`numpyro.set_host_device_count` is ignored once XLA has initialised.** Calling it per-fit means a later 4-chain run silently falls back to sequential execution with one warning on stderr and double the wall time. Device count must be set through `XLA_FLAGS` before JAX is imported.

**Effects estimated relative to zero and effects estimated relative to the league mean differ by a constant**, which appears in a simulation as a small bias identical across every scenario. Five structurally different scenarios returning the same bias to four decimal places is arithmetic, not a statistical property.

**arviz 1.3.0 is incompatible with numpyro 0.21.0**; `az.from_numpyro` fails inside `infer_dims`. Diagnostics here use `numpyro.diagnostics` directly.

---

## 9. Reproducibility

See [`README.md`](README.md) for the command sequence. The v1 pipeline is unchanged and still runs; its instructions are at tag [`v1.0`](../../tree/v1.0).

Seeds are fixed: `20260906` for splits and cross-fitting folds, replication index for simulation draws. Data files, model caches and simulation output are not version-controlled and regenerate from the commands given. Every published figure is also written to a version-controlled CSV under [`results/`](results/), so a reader can check a number without refitting anything.

**Tests** ([`tests/`](tests/), 17 of them, about ten seconds, no downloads) cover three things that fail silently rather than loudly: that cross-fitting folds and the train/validation split never divide a game, which is the whole point of out-of-sample baseline probabilities; that the per-draw Δ computation matches a hand calculation and assigns each catcher its own effect, since a misalignment there would hand every catcher someone else's number without raising an error; and that NUTS recovers known parameters on synthetic data with a fixed seed, which is the only check that would catch the v2 model itself being changed — a centred parameterisation, a wrong prior, a mis-wired `logit_base`. Continuous integration runs them on every push.
