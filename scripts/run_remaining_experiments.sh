#!/usr/bin/env bash
#
# Everything the revision still needs, after scripts/rerun_revision.sh has
# produced models A / B / ES / C.
#
#     bash scripts/run_remaining_experiments.sh
#
# Each experiment writes a sentinel file and is skipped if that file already
# exists, so this is safe to interrupt and restart — you lose at most the
# experiment that was in flight. Pass --force to any single module to redo it.
#
# Rough runtime, longest first:
#   sensitivity   2-5 h   (4 variants x [10 seeds + 18 grid cells + 2 learners])
#   transfer      1-3 h   (size-matched, domain-balanced, learning curves)
#   groups        30-60 m (11 refits + grouped SHAP)
#   pseudospace   30-60 m (17 feature models + 2 VAEP refits)
#   coverage      10-20 m (parses raw 360 JSON once)
#   extras        < 5 m   (post-hoc only)
#
# Run it overnight. To do only part of it, call the modules directly:
#   python -m src.experiments groups
#   python -m src.experiments sensitivity --n-seeds 3     # quick pass

set -euo pipefail

if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "ERROR: no python or python3 on PATH. Run: source .venv/bin/activate" >&2
  exit 1
fi
echo "Using interpreter: $($PY --version 2>&1)"

LOG="outputs/revision_experiments_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs
echo "Logging to $LOG"

step() { echo | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }

step "Preflight: all four model variants must exist"
$PY - <<'EOF'
import sys
from pathlib import Path
d = Path("data/processed/model_outputs")
missing = [m for m in ("model_A", "model_B", "model_ES", "model_C")
           if not (d / f"{m}_predictions.parquet").exists()]
if missing:
    sys.exit(
        f"Missing {missing}. Run scripts/rerun_revision.sh first "
        "(with WITH_ES=1 if model_ES is the one missing)."
    )
import pandas as pd
n = pd.read_parquet("data/processed/features/features_phase.parquet").shape[1]
if n < 40:
    sys.exit(f"features_phase.parquet has {n} columns — phase interactions missing.")
print("OK: four variants present, phase interactions present.")
EOF

# Coverage first: the visibility controls in revision_extras depend on it, and
# it is cheap relative to everything else.
step "1/6 Freeze-frame coverage features (R2-C6)"
if [ -f data/processed/features/coverage_features.parquet ]; then
  echo "already present — skipping" | tee -a "$LOG"
else
  $PY -m src.coverage_features 2>&1 | tee -a "$LOG"
fi

step "2/6 Post-hoc extras: calibration, coverage controls, phase example"
$PY -m src.revision_extras all 2>&1 | tee -a "$LOG"

step "3/6 Grouped ablation and grouped SHAP (R1-C4/Q4)"
$PY -m src.experiments groups 2>&1 | tee -a "$LOG"

step "4/6 Pseudo-Space VAEP: event-data approximation (R1-C4/Q4)"
$PY -m src.experiments pseudospace 2>&1 | tee -a "$LOG"

step "5/6 Transfer controls: size-matched, domain-balanced, curves, A-distance"
$PY -m src.experiments transfer --n-seeds 5 2>&1 | tee -a "$LOG"

step "6/6 Sensitivity: seeds, hyperparameter grid, alternative learners"
$PY -m src.experiments sensitivity --n-seeds 10 2>&1 | tee -a "$LOG"

step "Done"
cat <<'EOF' | tee -a "$LOG"

New results are in:

  outputs/tables/experiments/     grouped_ablation, grouped_shap,
                                  pseudospace_*, transfer_controls,
                                  transfer_learning_curves, domain_divergence,
                                  sensitivity_seeds / _grid / _learners
  outputs/tables/rebuttal/        brier_decomposition, reliability_curves,
                                  ece_bin_sensitivity, coverage_*,
                                  phase_example_summary

Commit and push these and I will fold them into the paper.
EOF
