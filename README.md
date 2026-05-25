# vaep-womens-football

Phase- and space-aware VAEP for women's football, with men-to-women transfer learning.

Master's thesis pipeline (Kaj Skubiszak, UAM Poznań, 2026). Follows the 18-stage architecture defined in `plan_vaep_python_only.md`, extended with cross-gender transfer learning.

## Quick start

```bash
# 1. Clone StatsBomb open data (one-time, ~3GB)
cd data/raw
git clone https://github.com/statsbomb/open-data.git statsbomb_open_data

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the pipeline, stage by stage
python -m src.data_inventory       # Stage 1: inventory
python -m src.preprocessing        # Stage 3: clean events
python -m src.spadl_conversion     # Stage 4: SPADL
python -m src.labels               # Stage 5: score/concede labels
python -m src.features_baseline    # Stage 6: baseline features
python -m src.modelling --model A  # Stage 7: classical VAEP
python -m src.phase_features       # Stage 8: phase labels
python -m src.modelling --model B  # Stage 9: phase-aware VAEP
python -m src.space_features       # Stages 10-11: spatial features
python -m src.modelling --model C  # Stage 12: phase- and space-aware
python -m src.evaluation           # Stage 13: metrics
python -m src.vaep_values          # Stage 14: action values
python -m src.aggregation          # Stage 15: player rankings
python -m src.case_studies         # Stage 16: case studies
python -m src.visualisation        # Stage 17: figures
python -m src.reporting            # Stage 18: technical report
```

## Configuration

All paths and hyperparameters live in `configs/default.yaml`. Override on the CLI or set `VAEP_CONFIG=path/to/other.yaml`.

## Repo layout

```
vaep-womens-football/
  data/
    raw/statsbomb_open_data/          # cloned StatsBomb repo
    interim/                          # cleaned events, 360 joined
    processed/                        # actions, labels, features, model outputs
  notebooks/                          # exploration & validation only (no pipeline code)
  src/                                # pipeline modules
  outputs/                            # figures, tables, reports
  models/                             # trained model artifacts
  configs/                            # YAML configs
```

## Pipeline modules

| Stage | Module | Output |
|---|---|---|
| 1 | `data_inventory.py` | `data/processed/inventory/inventory_dataset.parquet` |
| 3 | `preprocessing.py` | `data/interim/events_clean/events_clean.parquet` |
| 4 | `spadl_conversion.py` | `data/processed/actions/actions_spadl.parquet` |
| 5 | `labels.py` | `data/processed/labels/vaep_labels_k{5,10,15}.parquet` |
| 6 | `features_baseline.py` | `data/processed/features/features_baseline.parquet` |
| 7 | `modelling.py --model A` | `models/model_A_classic/*.pkl` |
| 8 | `phase_features.py` | `data/processed/features/phase_labels.parquet` |
| 9 | `modelling.py --model B` | `models/model_B_phase/*.pkl` |
| 10-11 | `space_features.py` | `data/processed/features/space_features.parquet` |
| 12 | `modelling.py --model C` | `models/model_C_phase_space/*.pkl` |
| 13 | `evaluation.py`, `calibration.py` | `outputs/tables/model_comparison_metrics.csv` |
| 14 | `vaep_values.py` | `data/processed/model_outputs/all_models_vaep_values.parquet` |
| 15 | `aggregation.py` | `outputs/tables/player_rankings.csv` |
| 16 | `case_studies.py` | `outputs/figures/case_study_*.png` |
| 17 | `visualisation.py` | `outputs/figures/*.png` |
| 18 | `reporting.py` | `outputs/reports/technical_report.md` |

## Transfer learning

For Models A/B/C, the default config trains on men's 360 competitions (WC 2022, EURO 2024, etc.) and evaluates on women's UEFA EURO 2025. Three transfer variants are reported: zero-shot, fine-tuned, and women-only control. See `src/modelling.py --help` for flags.

## License

Thesis code, released under MIT. StatsBomb open data is © StatsBomb Ltd. and governed by their open-data license.
