"""Post-hoc analyses that need no retraining, beyond those in rebuttal_analysis.

    calibration   Reliability curves on a log probability axis, Brier
                  decomposition into reliability and resolution, and an ECE
                  bin-count sensitivity check.                    (R2-C4)

    coverage      The visibility controls of Reviewer 2's Comment 6, using the
                  real polygon measures from coverage_features.py rather than
                  visible-player counts alone.                    (R2-C6)

    phase_example A matched pair of own-third passes that differ only in phase,
                  to make the motivation in Section 1 concrete.   (R1-C1/Q1)

Run:
    python -m src.revision_extras all
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .rebuttal_analysis import (
    OUT_DIR,
    PAPER_NAME,
    PRED_DIR,
    FEAT_DIR,
    available_models,
    build_action_panel,
    add_strata,
    load_predictions,
    restrict,
    subset_keys,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration  (R2-C4)
# ---------------------------------------------------------------------------

def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

    This separates the two ways a model can lose Brier score. Reliability is how
    far predicted probabilities sit from observed frequencies (calibration);
    resolution is how much the predictions vary around the base rate
    (discrimination). A model can trade one for the other, which is exactly what
    Reviewer 2 asks us to distinguish for PS-VAEP.
    """
    base = float(y.mean())
    quantiles = np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1])
    idx = np.clip(np.digitize(p, quantiles), 0, n_bins - 1)

    reliability = resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        p_bar = float(p[mask].mean())
        o_bar = float(y[mask].mean())
        reliability += n_b * (p_bar - o_bar) ** 2
        resolution += n_b * (o_bar - base) ** 2

    n = len(y)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "reliability": reliability / n,   # lower is better
        "resolution": resolution / n,     # higher is better
        "uncertainty": base * (1 - base),
        "base_rate": base,
    }


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Equal-frequency reliability curve, suitable for a log-axis plot."""
    quantiles = np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1])
    idx = np.clip(np.digitize(p, quantiles), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()),
            "observed_frequency": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


def ece_at(y: np.ndarray, p: np.ndarray, n_bins: int, adaptive: bool) -> float:
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1])
        idx = np.clip(np.digitize(p, edges), 0, n_bins - 1)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def run_calibration(keys: pd.DataFrame) -> None:
    decomp, curves, sens = [], [], []
    for model_id in available_models():
        preds = load_predictions(model_id)
        if preds is None:
            continue
        m = restrict(preds, keys)
        for head in ("score", "concede"):
            y = m[f"{head}_label"].values
            p = m[f"p_{head}"].values
            decomp.append({"model_id": model_id, "paper_name": PAPER_NAME[model_id],
                           "head": head, **brier_decomposition(y, p)})
            c = reliability_curve(y, p)
            c["model_id"] = model_id
            c["paper_name"] = PAPER_NAME[model_id]
            c["head"] = head
            curves.append(c)
            for n_bins in (5, 10, 15, 20, 30, 50):
                for adaptive in (False, True):
                    sens.append({
                        "model_id": model_id, "paper_name": PAPER_NAME[model_id],
                        "head": head, "n_bins": n_bins,
                        "binning": "adaptive" if adaptive else "equal_width",
                        "ece": ece_at(y, p, n_bins, adaptive),
                    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decomp).to_csv(OUT_DIR / "brier_decomposition.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT_DIR / "reliability_curves.csv", index=False)
    pd.DataFrame(sens).to_csv(OUT_DIR / "ece_bin_sensitivity.csv", index=False)
    log.info("Wrote brier_decomposition.csv, reliability_curves.csv, ece_bin_sensitivity.csv")


# ---------------------------------------------------------------------------
# Coverage controls  (R2-C6)
# ---------------------------------------------------------------------------

def run_coverage(keys: pd.DataFrame) -> None:
    """Does broadcast coverage explain the redistribution?

    Three tests, in increasing strictness: how coverage varies by zone; how it
    correlates with the change in value; and whether the player-level
    redistribution survives when restricted to the best-covered actions.
    """
    cov_path = FEAT_DIR / "coverage_features.parquet"
    if not cov_path.exists():
        log.warning("coverage_features.parquet missing — run `python -m src.coverage_features`.")
        return

    panel = add_strata(build_action_panel(keys))
    cov = pd.read_parquet(cov_path)
    panel = panel.merge(cov, on=["match_id", "action_id"], how="left")

    cov_cols = ["visible_area_frac", "actor_dist_to_boundary_m", "n_visible_players"]

    by_zone = (
        panel.groupby("stratum_zone", observed=True)
        .agg(n_actions=("delta_vaep", "size"),
             mean_delta=("delta_vaep", "mean"),
             **{f"mean_{c}": (c, "mean") for c in cov_cols})
        .reset_index()
    )
    by_zone.to_csv(OUT_DIR / "coverage_by_zone.csv", index=False)

    assoc = [{
        "measure": c,
        "spearman_with_delta": panel[c].corr(panel["delta_vaep"], method="spearman"),
        "spearman_with_density_5m": panel[c].corr(
            panel["defensive_density_around_ball_5m"], method="spearman"),
        "mean": panel[c].mean(),
        "sd": panel[c].std(),
    } for c in cov_cols]
    pd.DataFrame(assoc).to_csv(OUT_DIR / "coverage_association.csv", index=False)

    # Does the redistribution survive on well-covered actions only? If the
    # effect is an artefact of framing it should weaken sharply here.
    rows = []
    for label, sub in (
        ("all_actions", panel),
        ("high_coverage_top50pct", panel[panel["visible_area_frac"]
                                        >= panel["visible_area_frac"].median()]),
        ("actor_far_from_boundary", panel[panel["actor_dist_to_boundary_m"] >= 10.0]),
    ):
        if len(sub) < 500:
            continue
        row = {"subset": label, "n_actions": len(sub),
               "mean_delta_vaep": sub["delta_vaep"].mean()}
        for model_id in ("model_B", "model_C"):
            col = f"p_concede__{model_id}"
            if col in sub:
                y = sub["concede_label"].values
                if 0 < y.sum() < len(y):
                    row[f"concede_auc_{model_id}"] = roc_auc_score(y, sub[col].values)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "coverage_subsets.csv", index=False)
    log.info("Wrote coverage_by_zone.csv, coverage_association.csv, coverage_subsets.csv")


# ---------------------------------------------------------------------------
# Worked phase example  (R1-C1 / Q1)
# ---------------------------------------------------------------------------

def run_phase_example(keys: pd.DataFrame) -> None:
    """Find own-third passes that are near-identical except for their phase.

    Reviewer 1 asks for a concrete case where missing phase information leads to
    an implausible valuation. We look for build-up and defensive-transition
    passes matched on location and outcome, and compare what each model assigns
    against what actually happened.
    """
    panel = build_action_panel(keys)
    phase = FEAT_DIR / "phase_labels.parquet"
    if not phase.exists():
        log.warning("phase_labels.parquet missing.")
        return
    panel = panel.merge(pd.read_parquet(phase), on=["match_id", "action_id"], how="left")

    own_third = panel[(panel["type_name"] == "pass") & (panel["start_x"] <= 40.0)].copy()
    if own_third.empty or "phase" not in own_third:
        log.warning("No own-third passes with phase labels found.")
        return

    own_third["x_bin"] = (own_third["start_x"] // 10).astype(int)
    own_third["y_bin"] = (own_third["start_y"] // 10).astype(int)

    rows = []
    for (xb, yb, res), grp in own_third.groupby(["x_bin", "y_bin", "result_name"], observed=True):
        for ph, sub in grp.groupby("phase", observed=True):
            if len(sub) < 30:
                continue
            rows.append({
                "x_bin": xb, "y_bin": yb, "result": res, "phase": ph,
                "n_actions": len(sub),
                "observed_concede_rate": float(sub["concede_label"].mean()),
                **{f"mean_p_concede_{m}": float(sub[f"p_concede__{m}"].mean())
                   for m in available_models() if f"p_concede__{m}" in sub},
            })

    out = pd.DataFrame(rows)
    if out.empty:
        log.warning("No sufficiently populated matched strata found.")
        return
    out.to_csv(OUT_DIR / "phase_matched_strata.csv", index=False)

    # Summarise across all matched cells: within the same location and outcome,
    # how do the models separate phases compared with the observed rates?
    summary = (
        out.groupby("phase")
        .agg(n_cells=("n_actions", "size"), n_actions=("n_actions", "sum"),
             observed=("observed_concede_rate", "mean"),
             **{m: (f"mean_p_concede_{m}", "mean") for m in available_models()
                if f"mean_p_concede_{m}" in out})
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "phase_example_summary.csv", index=False)
    log.info("Wrote phase_matched_strata.csv and phase_example_summary.csv\n%s",
             summary.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("which", type=click.Choice(["all", "calibration", "coverage", "phase_example"]))
def main(which: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    keys = subset_keys()
    if which in ("all", "calibration"):
        run_calibration(keys)
    if which in ("all", "coverage"):
        run_coverage(keys)
    if which in ("all", "phase_example"):
        run_phase_example(keys)
    log.info("Done. Results in %s", OUT_DIR)


if __name__ == "__main__":
    main()
