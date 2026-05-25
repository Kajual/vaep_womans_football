"""Stage 13b — Calibration analysis.

For each trained model and each head (score, concede), produces:

  * a reliability-diagram dataframe (bin centres, fraction-of-positives,
    mean-predicted-probability, bin counts) — easy to plot with mplsoccer/matplotlib
  * Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  * optional re-calibration via Platt scaling or isotonic regression

The promotor's plan emphasises calibration as a primary contribution of
phase-/space-aware models — Chapter 6 §6.2 of the thesis will plot these
reliability diagrams.

Run:
    python -m src.calibration --model-id model_A --model-id model_B --model-id model_C
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import Config, load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reliability diagram + ECE / MCE
# ---------------------------------------------------------------------------

def reliability_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Compute per-bin reliability statistics.

    Returns a DataFrame with one row per non-empty bin:
        bin_lo, bin_hi, mean_pred, frac_pos, count
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred, bin_edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append({
            "bin_lo": float(bin_edges[b]),
            "bin_hi": float(bin_edges[b + 1]),
            "mean_pred": float(y_pred[mask].mean()),
            "frac_pos": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(reliability: pd.DataFrame, total_n: int) -> float:
    """Standard ECE: weighted average of |frac_pos - mean_pred| per bin."""
    weights = reliability["count"].to_numpy() / max(total_n, 1)
    diffs = np.abs(reliability["frac_pos"].to_numpy() - reliability["mean_pred"].to_numpy())
    return float((weights * diffs).sum())


def maximum_calibration_error(reliability: pd.DataFrame) -> float:
    """MCE: largest per-bin gap between observed and predicted."""
    if reliability.empty:
        return float("nan")
    diffs = np.abs(reliability["frac_pos"].to_numpy() - reliability["mean_pred"].to_numpy())
    return float(diffs.max())


# ---------------------------------------------------------------------------
# Re-calibration helpers (used in TL recalibration variant)
# ---------------------------------------------------------------------------

def fit_platt(y_true: np.ndarray, y_pred: np.ndarray) -> LogisticRegression:
    """Platt scaling: a 1-d logistic regression on the raw predictions."""
    return LogisticRegression(C=1e9, solver="lbfgs").fit(y_pred.reshape(-1, 1), y_true)


def fit_isotonic(y_true: np.ndarray, y_pred: np.ndarray) -> IsotonicRegression:
    """Isotonic regression."""
    return IsotonicRegression(out_of_bounds="clip").fit(y_pred, y_true)


def apply_recalibrator(model: Any, y_pred: np.ndarray) -> np.ndarray:
    if isinstance(model, LogisticRegression):
        return model.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    if isinstance(model, IsotonicRegression):
        return model.transform(y_pred)
    raise TypeError(f"Unknown recalibrator: {type(model)}")


# ---------------------------------------------------------------------------
# Top-level routine
# ---------------------------------------------------------------------------

def calibration_for_model(predictions: pd.DataFrame, model_id: str, n_bins: int = 15) -> dict:
    """Compute reliability data and ECE/MCE for both heads of one model."""
    out = {"model_id": model_id}
    out["reliability_score"] = reliability_data(
        predictions["score_label"].to_numpy(),
        predictions["p_score"].to_numpy(),
        n_bins=n_bins,
    )
    out["reliability_concede"] = reliability_data(
        predictions["concede_label"].to_numpy(),
        predictions["p_concede"].to_numpy(),
        n_bins=n_bins,
    )
    total = len(predictions)
    out["ece_score"] = expected_calibration_error(out["reliability_score"], total)
    out["ece_concede"] = expected_calibration_error(out["reliability_concede"], total)
    out["mce_score"] = maximum_calibration_error(out["reliability_score"])
    out["mce_concede"] = maximum_calibration_error(out["reliability_concede"])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
@click.option(
    "--model-id", "model_ids", multiple=True, required=True,
    help="Model identifier(s) whose predictions to analyse.",
)
@click.option("--n-bins", default=15, help="Number of reliability bins (default 15).")
def main(config_path: str | None, model_ids: tuple[str, ...], n_bins: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("tables_dir")

    summary_rows: list[dict] = []
    for mid in model_ids:
        pred_path = cfg.paths.model_outputs_dir / f"{mid}_predictions.parquet"
        if not pred_path.exists():
            log.warning("Predictions missing for %s at %s — skipping", mid, pred_path)
            continue
        preds = pd.read_parquet(pred_path)
        cal = calibration_for_model(preds, mid, n_bins=n_bins)

        # Persist per-model reliability tables — used by visualisation.py to
        # plot reliability diagrams.
        for head in ["score", "concede"]:
            rel: pd.DataFrame = cal[f"reliability_{head}"]
            rel.insert(0, "model_id", mid)
            rel.insert(1, "head", head)
            rel.to_csv(cfg.paths.tables_dir / f"reliability_{mid}_{head}.csv", index=False)

        summary_rows.append({
            "model_id": mid,
            "ece_score": cal["ece_score"],
            "mce_score": cal["mce_score"],
            "ece_concede": cal["ece_concede"],
            "mce_concede": cal["mce_concede"],
        })

    if not summary_rows:
        log.error("No model predictions found.")
        return

    df = pd.DataFrame(summary_rows).set_index("model_id")
    out_path = cfg.paths.tables_dir / "calibration_summary.csv"
    df.to_csv(out_path)
    log.info("Calibration summary written to %s\n%s", out_path, df.to_string())


if __name__ == "__main__":
    main()
