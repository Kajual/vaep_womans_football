# Retraining after adding phase x action-type interactions

`src/phase_features.py` now emits 42 phase x action-family interaction indicators
alongside the six phase one-hots, so `features_phase.parquet` grows from 8 columns
to 50. Every model that consumes the `phase` feature set — B and C, under all four
training strategies — has to be retrained. Model A is unaffected.

Run these in order from the repo root with the venv active. Total is roughly
30–60 minutes depending on the machine; the two `space` models are the slow ones.

```bash
source .venv/bin/activate

# 1. Rebuild the phase features (fast, ~1 min).
python -m src.phase_features

# 2. Retrain the pooled models B and C.
python -m src.modelling --model-id model_B --features baseline --features phase \
    --train-corpus men_360_source --train-corpus women_360_finetune \
    --eval-corpus women_360_evaluation

python -m src.modelling --model-id model_C --features baseline --features phase --features space \
    --train-corpus men_360_source --train-corpus women_360_finetune \
    --eval-corpus women_360_evaluation

# 3. Retrain the three transfer variants for B and C.
python -m src.transfer --model-id model_B --features baseline --features phase
python -m src.transfer --model-id model_C --features baseline --features phase --features space

# 4. Regenerate the downstream tables.
python -m src.evaluation --model-id model_A --model-id model_B --model-id model_C
python -m src.evaluation --model-id model_A --model-id model_B --model-id model_C \
    --restrict-to-360-subset
python -m src.calibration
python -m src.vaep_values
python -m src.aggregation
python -m src.case_studies

# 5. Rerun the revision analyses (bootstrap is the slow part, ~10 min).
python -m src.rebuttal_analysis --n-boot 1000
```

Please check the `--features` flag syntax against `python -m src.modelling --help`
before running step 2 — if the option is declared with `nargs=-1` rather than
`multiple=True`, it takes `--features baseline phase` instead of repeating the flag.

## What to send back

Everything the paper draws on is regenerated into these paths:

```
outputs/tables/model_comparison_metrics.csv
outputs/tables/model_comparison_metrics_matched.csv
outputs/tables/calibration_summary.csv
outputs/tables/transfer_comparison_model_*.csv
outputs/tables/ranking_stability.csv
outputs/tables/ranking_movers.csv
outputs/tables/case_studies_summary.csv
outputs/tables/rebuttal/*.csv
```

Committing and pushing is enough — I will pull the numbers from there.

## What I expect to change

Model A is untouched, so any E-VAEP number in the paper stays valid.

The interactions give the phase layer more capacity, so P-VAEP's conceding-head
advantage may widen. It may equally shrink: 11 of the 42 interaction cells are
structurally empty (a set piece is never a carry), and the remainder are sparse
outside the six large pass/carry combinations, so the added columns could simply
dilute the split search. Either outcome is reportable — the paper's claim is about
*where* context helps, not about the size of a single gap.

Worth watching: if P-VAEP's conceding ROC-AUC gain over E-VAEP loses its
bootstrap-robust status, that would remove the paper's only resampling-robust
predictive result. Should that happen, reverting to the plain one-hot phase layer
and deleting the interaction clause from Section 3.3 is the better option, and no
reviewer would object — the submitted text describes a feature that was never
implemented, so removing the sentence is a correction either way.
