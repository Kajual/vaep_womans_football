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

    # Strictly matched comparison: evaluate every model only on the actions
    # that carry a 360 freeze frame (Model C's action set).
    python -m src.evaluation --model-id model_A --model-id model_B --model-id model_C \\
        --restrict-to-360-subset
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
@click.option(
    "--restrict-to-360-subset", is_flag=True, default=False,
    help="Evaluate every model only on the actions that carry a 360 freeze "
         "frame (the action set of --subset-from), for a strictly matched "
         "head-to-head comparison. Writes a separate '_matched' table.",
)
@click.option(
    "--subset-from", default="model_C",
    help="Model whose prediction set defines the 360 subset "
         "(default: model_C, the only model restricted to 360 actions).",
)
def main(
    config_path: str | None,
    model_ids: tuple[str, ...],
    restrict_to_360_subset: bool,
    subset_from: str,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("tables_dir")

    # When restricting, resolve the (match_id, action_id) keys of the 360 subset.
    subset_keys: pd.DataFrame | None = None
    if restrict_to_360_subset:
        sub_path = cfg.paths.model_outputs_dir / f"{subset_from}_predictions.parquet"
        if not sub_path.exists():
            log.error("Subset-reference predictions missing: %s", sub_path)
            return
        subset_keys = (
            pd.read_parquet(sub_path)[["match_id", "action_id"]].drop_duplicates()
        )
        log.info("Restricting evaluation to the %d-action 360 subset defined by %s",
                 len(subset_keys), subset_from)

    rows: list[dict] = []
    for mid in model_ids:
        pred_path = cfg.paths.model_outputs_dir / f"{mid}_predictions.parquet"
        if not pred_path.exists():
            log.warning("Predictions missing for %s at %s — skipping", mid, pred_path)
            continue
        preds = pd.read_parquet(pred_path)
        if subset_keys is not None:
            before = len(preds)
            preds = preds.merge(subset_keys, on=["match_id", "action_id"], how="inner")
            log.info("%s: restricted %d -> %d actions", mid, before, len(preds))
        rows.append(evaluate_model(preds, mid))

    if not rows:
        log.error("No model predictions found.")
        return

    df = pd.DataFrame(rows).set_index("model_id")
    fname = (
        "model_comparison_metrics_matched.csv"
        if restrict_to_360_subset
        else "model_comparison_metrics.csv"
    )
    out_path = cfg.paths.tables_dir / fname
    df.to_csv(out_path)
    log.info("Wrote model comparison metrics to %s\n%s", out_path, df.to_string())


if __name__ == "__main__":
    main()
