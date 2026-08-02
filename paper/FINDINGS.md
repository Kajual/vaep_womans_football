# Results of the MLSA 2026 revision (paper 306)

All numbers recomputed from the retrained four-variant models (A / B / ES / C,
with phase x action-type interactions), on the matched 51,172-action subset,
with paired match-level bootstrap intervals (B = 1000).

Sources: `outputs/tables/rebuttal/` and `outputs/tables/experiments/`.

---

## Part 1 — The error in the submitted paper

Three action universes were mixed. PS-VAEP exists only on the 51,172 actions
with a freeze frame; E-VAEP and P-VAEP were measured on all 60,370. Section 3.6
of the submitted paper forbids exactly this. Table 3's discrimination rows obeyed
it; Table 2, Table 3's ECE row, and all of Section 4.3 did not.

The player analysis was worst affected. Recomputed on matched actions with the
same 117 players, rho(E, PS) rises from 0.69 to 0.82, and the named rank gains
collapse:

| Player | Submitted (unmatched) | Matched |
|---|---|---|
| Wullaert | +102 | +3 |
| Blackstenius | +94 | +21 |
| González | +88 | +5 |
| Girelli | +58 | +1 |

The unmatched comparison was reproduced exactly before concluding this, so the
diagnosis is confirmed rather than inferred.

---

## Part 2 — Results after retraining

### The 2x2 ablation (pooled, matched)

| | E-VAEP | P-VAEP | ES-VAEP | PS-VAEP |
|---|---|---|---|---|
| Scoring ROC-AUC | 0.825 | 0.827 | 0.823 | 0.823 |
| Conceding ROC-AUC | 0.837 | **0.865** | 0.840 | 0.849 |
| Conceding AP | 0.171 | 0.178 | 0.167 | 0.156 |

Effect decomposition on conceding ROC-AUC:

- phase without space (P − E) = **+0.027**; phase with space (PS − ES) = +0.010
- space without phase (ES − E) = +0.003; space with phase (PS − P) = −0.015
- interaction = **−0.018**

The two layers are substitutes, not complements: they carry overlapping
information about conceding risk, and adding space on top of phase *reduces*
performance. Only the P − E contrast has a bootstrap interval excluding zero
(**+0.027, CI [+0.013, +0.046]**). The main effects of space and the interaction
are individually not identifiable at n = 31 matches, exactly as expected.

**The phase x action-type interactions helped**: the phase effect grew from
+0.023 to +0.027 after implementing them.

**PS-VAEP's conceding-AP advantage is gone.** It fell from 0.184 to 0.156, now
below both E-VAEP and P-VAEP. The "best conceding AP" claim should be dropped
outright rather than merely qualified.

### Player valuation (matched, B = 1000)

| Model pair | Spearman | 95% CI |
|---|---|---|
| E-VAEP vs P-VAEP | 0.977 | [0.933, 0.976] |
| E-VAEP vs PS-VAEP | 0.824 | [0.694, 0.845] |
| P-VAEP vs PS-VAEP | 0.808 | [0.701, 0.841] |

25 of 117 players have a rank change whose interval excludes zero. Largest
robust risers: Severini +69, Hoffmann +39, Pajor +39, Giuliani +35, Tysiak +32,
Naalsund +29. Largest robust fallers: Groenen −51, Deloose −48,
Zigiotti-Olme −32, Conca −21. Positionally mixed in both directions.

### The congestion mechanism no longer holds

On the retrained models the 3+ opponents within 5 m stratum gives
**+0.0010, CI [−0.0003, +0.0023]** — the interval now includes zero. On the
previous models this was the one stratum supporting the mechanism
(+0.0023, CI [+0.0005, +0.0039]).

No stratification — nearest-opponent distance, defensive density, pitch zone,
action type or outcome — now shows a robust positive shift. **The paper should
not claim a mechanism for the redistribution.** It can report that the
redistribution happens and that its cause is not established.

### Freeze-frame visibility (R2-C6) — clean

Correlation between visible-player count and the change in value: **0.020**.
Coverage varies by zone (14.4 visible players in midfield against 12.2 in the
penalty area) but is essentially unrelated to the revaluation. Real polygon
measures are now available: mean visible area 24% of the pitch, actor a median
6.7 m from the frame boundary.

---

## Part 3 — New experiments

### Transfer: it is mostly sample size (R1-Q2, R2-C3)

Conceding ROC-AUC, mean over 5 seeds:

| Strategy | Train matches | Conceding AUC | SD |
|---|---|---|---|
| Pooled, full | 295 | 0.854 | 0.006 |
| Pooled, domain-balanced | 190 | 0.851 | 0.008 |
| Men-only, full | 200 | 0.844 | 0.010 |
| Pooled, size-matched | 190 | 0.842 | 0.009 |
| Men-only, size-matched | 95 | 0.822 | 0.015 |
| Women-only | 95 | 0.821 | 0.009 |

At equal size (95 matches) men-only and women-only are indistinguishable
(0.822 vs 0.821). Performance tracks the number of training matches far more
than their provenance. **The honest conclusion is that more 360 data helps
regardless of whether it is men's or women's football** — which is itself a
useful finding for women's football analytics, and is what the paper should say.

Note: "size-matched" and "domain-balanced" as implemented are the same design
(95 men + 95 women) sampled with different seeds. Their 0.009 gap is therefore a
second read on seed noise, not a real contrast. Worth either merging them or
redefining domain-balanced.

### Competition type matters more than gender (R1-C6/Q5)

Proxy A-distance (0 = indistinguishable, 2 = trivially separable):

| Domain A | Domain B | A-distance |
|---|---|---|
| Men's international | Women's target | 0.713 |
| Men's club (Bundesliga) | Women's target | **0.955** |
| Men's international | Men's club | 0.214 |

Bundesliga is **further** from the women's target than men's international
football is, despite matching on gender. Size-matched transfer agrees: from 34
matches, international gives 0.755 and club 0.717. This confirms Reviewer 1's
hypothesis and vindicates the reviewer's concern about the corpus construction.

### Spatial features are recoverable from event data (R1-C4/Q4)

Predicting each spatial feature from event features alone, then rebuilding VAEP
on the predictions:

| Variant | Conceding AUC | Conceding AP |
|---|---|---|
| PS-VAEP (observed 360) | 0.853 | 0.159 |
| Pseudo-Space VAEP (predicted) | **0.858** | **0.161** |

The approximation is as good as the real thing. Zone features are perfectly
recoverable (they are deterministic functions of coordinates, which are event
features — this should be stated rather than presented as a result). Density
features reach r² 0.39–0.70. This strongly supports the paper's own conclusion
that the spatial layer adds little predictive information.

Two features returned NaN (`nearest_opponent_distance`,
`nearest_teammate_distance`) because they are NaN when the actor is not
locatable in the frame; the approximation code needs to drop those rows.

### Grouped ablation (R1-C4/Q4)

All variants trained on the freeze-frame subset, so these are not directly
comparable with Table 3.

| Variant | Conceding AUC | Conceding AP |
|---|---|---|
| Baseline only | 0.847 | 0.177 |
| All groups | 0.853 | 0.159 |
| Drop defensive density | **0.860** | **0.194** |
| Drop defensive structure | 0.858 | 0.187 |
| Drop zone context | 0.845 | 0.139 |

Dropping the defensive-density group *improves* both metrics. Grouped SHAP
ranks pressure/support highest among spatial groups (0.074) ahead of phase
(0.044), but attribution and usefulness disagree — SHAP measures how much the
model uses a feature, not whether using it helps.

### Sensitivity — the most important caveat (R1-C8/Q7)

Ten seeds, all variants trained on the freeze-frame subset:

| Variant | Mean conceding AUC | SD | Range |
|---|---|---|---|
| E-VAEP | 0.850 | 0.009 | 0.834–0.863 |
| P-VAEP | 0.848 | 0.004 | 0.842–0.856 |
| ES-VAEP | 0.849 | 0.007 | 0.838–0.857 |
| PS-VAEP | 0.852 | 0.008 | 0.836–0.864 |

**In this configuration the phase advantage disappears.** Seed-to-seed variation
(SD ≈ 0.004–0.009, range up to 0.03) is comparable to the +0.027 effect claimed
from a single seed.

This is not yet a refutation, because the sensitivity run trains every variant
on the 360 subset (459k rows), whereas production E-VAEP and P-VAEP train on the
full stream (531k rows). The two configurations are not comparable. But it does
mean **the headline result has not been shown to survive seed variation in the
configuration the paper actually reports**.

`python -m src.experiments seedcheck` was written to settle this: it repeats the
production setup across seeds and reports the paired P − E difference per seed.
**This should be run before the paper is submitted.**

Partial reassurance: the ordering holds under both alternative learners.
Logistic regression gives P 0.840 vs E 0.820; XGBoost gives P 0.865 vs E 0.860
(and XGBoost beats LightGBM throughout, 0.857–0.865).

---

## What the paper should now claim

1. Phase improves conceding-risk estimation — **pending the seedcheck result**.
2. Space does not improve aggregate prediction, and degrades it when added to
   phase. The two layers are substitutes.
3. Space reorders the player ranking substantially (rho 0.98 to 0.82), with no
   established mechanism and no claim that the reordering is more accurate.
4. Transfer benefits come from data volume, not from cross-domain information.
5. Competition type matters more than gender for source similarity.
6. The spatial features are approximable from event data, so the approach
   transfers to competitions without 360 coverage.
7. Broadcast coverage does not explain the redistribution.

Points 4, 5 and 6 are new contributions the submitted paper did not have, and
they are more interesting than the claims that were lost.
