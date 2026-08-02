#!/usr/bin/env bash
#
# Regenerate every result the MLSA 2026 revision depends on.
#
# Run from the repo root with the venv active:
#     bash scripts/rerun_revision.sh
#
# Optional: also train the Event+Space model that completes Reviewer 2's 2x2
# ablation. Adds one training run plus its transfer variants.
#     WITH_ES=1 bash scripts/rerun_revision.sh
#
# Everything downstream adapts automatically to whether ES-VAEP exists.
#
# Rough runtime: 30-60 min without ES, 45-90 min with. The two space models are
# the slow ones. Safe to re-run; every step overwrites its own outputs.

set -euo pipefail

TRAIN_CORPORA="--train-corpus men_360_source --train-corpus women_360_finetune"
EVAL_CORPUS="--eval-corpus women_360_evaluation"
WITH_ES="${WITH_ES:-0}"

step() { echo; echo "=== $* ==="; }

step "0. Back up current predictions (data/ is gitignored, so git will not)"
BACKUP="data/_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"
cp data/processed/model_outputs/*.parquet "$BACKUP"/ 2>/dev/null || true
cp data/processed/features/features_phase.parquet "$BACKUP"/ 2>/dev/null || true
echo "Backed up to $BACKUP"

step "1. Rebuild phase features (now includes phase x action-type interactions)"
python -m src.phase_features

step "2. Retrain the pooled models"
# Model A is unaffected by the interaction change, but retraining it costs
# little and guarantees all four models come from one consistent run.
python -m src.modelling --model-id model_A --features baseline \
    $TRAIN_CORPORA $EVAL_CORPUS
python -m src.modelling --model-id model_B --features baseline --features phase \
    $TRAIN_CORPORA $EVAL_CORPUS
python -m src.modelling --model-id model_C --features baseline --features phase --features space \
    $TRAIN_CORPORA $EVAL_CORPUS

if [ "$WITH_ES" = "1" ]; then
  step "2b. Train ES-VAEP (event + space, no phase) for the 2x2 ablation"
  python -m src.modelling --model-id model_ES --features baseline --features space \
      $TRAIN_CORPORA $EVAL_CORPUS
fi

step "3. Retrain the transfer variants"
python -m src.transfer --model-id model_A --features baseline
python -m src.transfer --model-id model_B --features baseline --features phase
python -m src.transfer --model-id model_C --features baseline --features phase --features space
if [ "$WITH_ES" = "1" ]; then
  python -m src.transfer --model-id model_ES --features baseline --features space
fi

step "4. Evaluation and calibration"
MODEL_FLAGS="--model-id model_A --model-id model_B --model-id model_C"
if [ "$WITH_ES" = "1" ]; then
  MODEL_FLAGS="$MODEL_FLAGS --model-id model_ES"
fi
python -m src.evaluation $MODEL_FLAGS
python -m src.evaluation $MODEL_FLAGS --restrict-to-360-subset
python -m src.calibration $MODEL_FLAGS

step "5. Action values and player rankings"
python -m src.vaep_values $MODEL_FLAGS
# --matched is essential. Without it, player totals are summed over 60,370
# actions for the event/phase models and 51,172 for the space models, which is
# the error that produced the +102-rank shifts in the submitted paper.
python -m src.aggregation --matched
python -m src.case_studies

step "6. Revision analyses (bootstrap is the slow part, ~10 min)"
python -m src.rebuttal_analysis --n-boot 1000

step "Done"
cat <<'EOF'

Regenerated. The files the paper draws on:

  outputs/tables/model_comparison_metrics_matched.csv
  outputs/tables/calibration_summary.csv
  outputs/tables/transfer_comparison_model_*.csv
  outputs/tables/ranking_stability.csv
  outputs/tables/ranking_movers.csv
  outputs/tables/rebuttal/*.csv

Commit and push these; the numbers in paper/main.tex can then be refreshed
against them. Cells that need refreshing are marked in main.tex with
"% NUMBERS TO REFRESH".
EOF
