"""Stage 13a — Predictive-performance evaluation.

For each trained model, computes the standard binary-classifier metrics on
the held-out eval corpus:

  * ROC-AUC
  * Average Precision (PR-AUC)
  * Brier score
  * Log-loss
  * Positive rate (sanity check)

Calibration (reliability diagrams, ECE) lives in ``calibration.py``.

Run:
    python -m src.evaluation --model-id model_A --model-id model_B --model-id model_C
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .config import Config, load_config

log = logging.getLogger(__name__)


def metrics_for_head(y_true: np.ndarray, y_pred: np.ndarray, head: str) -> dict[str, float]:
    """Compute the metric bundle for one head (score or concede)."""
    if len(np.unique(y_true)) < 2:
        log.warning("Head '%s' has only one class in eval set; metrics ill-defined.", head)
        return {
            f"{head}_auc": float("nan"),
            f"{head}_avg_precision": float("nan"),
            f"{head}_brier": float(np.mean((y_pred - y_true) ** 2)),
            f"{head}_log_loss": float("nan"),
            f"{head}_pos_rate_pct": float(y_true.mean() * 100),
        }
    return {
        f"{head}_auc": float(roc_auc_score(y_true, y_pred)),
        f"{head}_avg_precision": float(average_precision_score(y_true, y_pred)),
        f"{head}_brier": float(brier_score_loss(y_true, y_pred)),
        f"{head}_log_loss": float(log_loss(y_true, np.clip(y_pred, 1e-9, 1 - 1e-9))),
        f"{head}_pos_rate_pct": float(y_true.mean() * 100),
    }


def evaluate_model(predictions: pd.DataFrame, model_id: str) -> dict[str, float]:
    """Compute predictive metrics for one model's eval-set predictions."""
    out: dict[str, float] = {"model_id": model_id}
    out.update(metrics_for_head(
        y_true=predictions["score_label"].to_numpy(),
        y_pred=predictions["p_score"].to_numpy(),
        head="score",
    ))
    out.update(metrics_for_head(
        y_true=predictions["concede_label"].to_numpy(),
        y_pred=predictions["p_concede"].to_numpy(),
        head="concede",
    ))
    out["n_eval_rows"] = len(predictions)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
@click.option(
    "--model-id", "model_ids", multiple=True, required=True,
    help="Model identifier(s) to evaluate. Pass multiple to compare side by side.",
)
def main(config_path: str | None, model_ids: tuple[str, ...]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("tables_dir")

    rows: list[dict] = []
    for mid in model_ids:
        pred_path = cfg.paths.model_outputs_dir / f"{mid}_predictions.parquet"
        if not pred_path.exists():
            log.warning("Predictions missing for %s at %s — skipping", mid, pred_path)
            continue
        preds = pd.read_parquet(pred_path)
        rows.append(evaluate_model(preds, mid))

    if not rows:
        log.error("No model predictions found.")
        return

    df = pd.DataFrame(rows).set_index("model_id")
    out_path = cfg.paths.tables_dir / "model_comparison_metrics.csv"
    df.to_csv(out_path)
    log.info("Wrote model comparison metrics to %s\n%s", out_path, df.to_string())


if __name__ == "__main__":
    main()
