# Verification of the submitted paper against the pipeline outputs

All numbers below were recomputed from the artefacts in `data/processed/model_outputs/`
by `src/rebuttal_analysis.py`. Results are in `outputs/tables/rebuttal/`.

The single underlying problem is that **three different action universes were mixed**
across the paper's tables:

- the full evaluation set: **60,370** actions (E-VAEP, P-VAEP)
- the 360 freeze-frame subset: **51,172** actions (PS-VAEP only)

Section 3.6 of the paper explicitly says all ablation metrics are reported on the
matched 51,172-action subset, and warns that mixing universes "would systematically
advantage PS-VAEP". Table 3's discrimination row does follow that rule. Table 2,
Table 3's ECE row, and all of Section 4.3 do not.

---

## 1. Table 2 (transfer) is computed on mixed universes

E-VAEP and P-VAEP were evaluated on 60,370 actions, PS-VAEP on 51,172.
Recomputed with every model on the matched subset:

| Strategy | E-VAEP | P-VAEP | PS-VAEP |
|---|---|---|---|
| Women-only — paper | 0.820 | 0.828 | 0.802 |
| Women-only — **matched** | **0.811** | **0.827** | 0.802 |
| Men-only — paper | 0.830 | 0.830 | 0.840 |
| Men-only — **matched** | **0.834** | **0.833** | **0.841** |
| Fine-tuned — paper | 0.824 | 0.826 | 0.837 |
| Fine-tuned — **matched** | **0.831** | **0.831** | 0.837 |
| Pooled — paper | 0.837 | 0.858 | 0.851 |
| Pooled — **matched** | 0.837 | **0.860** | 0.851 |

The paper's headline transfer conclusion — pooled beats every other strategy for all
three models — survives. But under men-only and fine-tuned transfer, P-VAEP is now
marginally *below* E-VAEP, so the claim that phase helps under every strategy does not.

## 2. Table 3's ECE row is wrong

The ECE values were taken from `outputs/tables/calibration_summary.csv`, which computes
E-VAEP and P-VAEP on the full set. The PS-VAEP values match no file in the repository
and appear to come from a superseded run.

| Metric | E-VAEP | P-VAEP | PS-VAEP |
|---|---|---|---|
| Scoring ECE — paper | 0.00323 | 0.00322 | 0.00349 |
| Scoring ECE — **actual matched** | 0.00375 | 0.00370 | **0.00356** |
| Conceding ECE — paper | 0.00144 | 0.00127 | 0.00134 |
| Conceding ECE — **actual matched** | 0.00140 | 0.00127 | **0.00146** |

Two directions flip. PS-VAEP is in fact the **best** on scoring ECE and the **worst**
on conceding ECE — the opposite of what the text asserts in both cases.

Under adaptive (equal-frequency) binning, which Reviewer 2 asks for in C4 because
equal-width bins are uninformative at a 0.34% positive rate, E-VAEP is best calibrated
on both heads (scoring 0.00366, conceding 0.00091). The claim that P-VAEP gives "the
best calibration" does not survive either binning scheme cleanly.

## 3. Table 4 and the player-valuation section are the most affected

Player VAEP totals for E-VAEP and P-VAEP were summed over 60,370 actions, but over
51,172 for PS-VAEP. Roughly 9,200 actions therefore contribute to a player's E-VAEP
total but not to her PS-VAEP total, so part of the reported "redistribution" is just
the missing 15% of each player's actions.

Recomputed on the matched action set, using the same 117 players with 270+ minutes:

| Model pair | Paper | **Matched** | 95% CI (match bootstrap) |
|---|---|---|---|
| E-VAEP vs P-VAEP | 0.973 | 0.974 | [0.940, 0.979] |
| E-VAEP vs PS-VAEP | 0.691 | **0.834** | [0.704, 0.836] |
| P-VAEP vs PS-VAEP | 0.672 | **0.828** | [0.690, 0.832] |

The redistribution is real — 0.83 against 0.97 for E-vs-P — but substantially smaller
than the reported 0.69.

**The named examples are almost entirely artefact.** The unmatched comparison exactly
reproduces the paper's figures, which confirms the diagnosis:

| Player | Paper (unmatched) | **Matched** |
|---|---|---|
| Wullaert | 116 → 14 (**+102**) | 17 → 14 (**+3**) |
| Blackstenius | 114 → 20 (**+94**) | 41 → 20 (**+21**) |
| González | 98 → 10 (**+88**) | 15 → 10 (**+5**) |
| Girelli | 79 → 21 (**+58**) | 22 → 21 (**+1**) |
| Pajor | 117 → 78 (**+39**) | 113 → 78 (**+35**) |
| Maanum | 10 → 3 | 10 → 3 (**+7**, robust) |

The PS-VAEP ranks are stable; it is the E-VAEP ranks that move, because they were
computed over the larger action set.

On matched data the largest robust risers are Severini (+50), Hoffmann (+41),
Pajor (+35), Bacha (+34), James (+31) and Naalsund (+29) — a positionally mixed group
of midfielders, forwards and a full-back, not "all centre-forwards". Only 36 of 117
players have a rank change whose 95% interval excludes zero.

Note that even in the paper's own numbers the riser list is misstated: the top six by
rank gain are Wullaert, Blackstenius, González, Girelli, Hoffmann and **Schertenleib
(+47)** — Pajor (+39) is ninth — and all six have negative E-VAEP per 90, not "four of
them".

## 4. What survives the bootstrap

Paired match-level bootstrap, B = 1000, resampling the 31 evaluation matches and
recomputing every model on the same resample.

Only two differences in the pooled ablation have a 95% interval excluding zero:

| Comparison | Metric | Difference | 95% CI | Robust |
|---|---|---|---|---|
| P-VAEP − E-VAEP | conceding ROC-AUC | +0.0231 | [+0.0081, +0.0387] | **yes** |
| PS-VAEP − P-VAEP | scoring ROC-AUC | −0.0051 | [−0.0102, −0.0002] | **yes** |
| P-VAEP − E-VAEP | conceding AP | +0.0060 | [−0.0187, +0.0298] | no |
| PS-VAEP − P-VAEP | conceding AP | +0.0062 | [−0.0272, +0.0380] | no |
| PS-VAEP − P-VAEP | conceding ROC-AUC | −0.0092 | [−0.0373, +0.0149] | no |
| P-VAEP − E-VAEP | conceding log loss | −0.0003 | [−0.0009, +0.0003] | no |

**The paper's central claim holds**: phase context robustly improves conceding
discrimination under pooled training. The claim that PS-VAEP "achieves the best
conceding AP" does not — that difference is well inside noise.

In Table 2, only the pooled P-vs-E difference is robust; every men-only, women-only and
fine-tuned difference has an interval spanning zero.

### Where the AP difference comes from

PS-VAEP's conceding AP advantage rests on a handful of actions at the very top of the
ranking. Cumulative true positives out of 175 total concessions:

| Model | top 10 | top 25 | top 50 | top 100 | top 500 |
|---|---|---|---|---|---|
| E-VAEP | 7 | 19 | 29 | 34 | 52 |
| P-VAEP | 6 | 20 | 27 | 37 | 59 |
| PS-VAEP | 8 | 17 | 30 | 35 | 53 |

PS-VAEP is *worse* than P-VAEP at every threshold from 1% to 50% of the ranking. The
interpretation offered in the draft rebuttal to R2-C4 — that spatial features sharpen
the ranking of the rare positive class — is not supported.

## 5. Two claims with no supporting code

- Section 3.3 states that the phase layer adds "the one-hot phase label **and its
  interactions with action type**". `features_phase.parquet` contains six one-hot
  columns and nothing else; no interaction terms are constructed anywhere in
  `modelling.py`.
- Section 5 states that the P-VAEP gain "remains positive under match-level bootstrap
  resampling". No bootstrap existed in the repository before this revision. The claim
  now happens to be **true** (see §4), but it was unsupported when written.

## 6. Results that came out in the paper's favour

- **Freeze-frame visibility is not driving the redistribution** (Reviewer 2, C6). The
  Spearman correlation between the number of visible players and the change in value is
  0.019 overall and 0.065 in the penalty area. Coverage does vary by zone (14.4 visible
  players in midfield against 12.2 in the penalty area), but it is essentially unrelated
  to the revaluation. This is a clean answer to the reviewer's strongest methodological
  concern.
- **The congestion mechanism has partial support.** Stratifying the change from P-VAEP
  to PS-VAEP, the only stratum with a robustly positive mean change is actions with
  3 or more opponents within 5 m (+0.0023, CI [+0.0005, +0.0039], 5.2% of actions).

  However, there is no gradient by distance to the nearest opponent (−0.0015, −0.0018,
  −0.0014, −0.0013 across <2 m, 2–5 m, 5–10 m, >10 m) and none by pitch zone. The
  largest robust negative shifts are throw-ins (−0.0105) and keeper claims (−0.0159),
  not "high-volume progression in open space"; passes shift down slightly (−0.0026) and
  dribbles are unchanged (−0.0001).

  So the density half of the mechanism claim is supported and the pressure half is not.

---

## Recommended revisions

1. Recompute Tables 2, 3 and 4 on the matched 51,172-action subset. This makes the
   paper consistent with its own Section 3.6 and makes the answer to R2-C8b true.
2. Replace the named-player examples. The +102/+94/+88/+58 figures cannot be defended.
3. Reframe the redistribution as ρ ≈ 0.83, positionally mixed, with 36 of 117 rank
   changes robust — still a genuine and reportable effect.
4. Lead with the bootstrap result. That phase robustly improves conceding
   discrimination while nothing else survives resampling *is* the prediction-versus-
   revaluation thesis, stated more sharply than the submitted version manages.
5. Drop the "interactions with action type" clause, or implement the interactions.
6. Report the visibility check — it answers R2-C6 favourably and pre-empts the concern.
7. Restate the mechanism claim as congestion-driven only, and drop the pressure and
   open-space progression framing that the strata do not support.
