"""Stage 14 — Compute VAEP per action from model predictions.

Given per-action probabilities ``p_score`` and ``p_concede`` from a trained
model, the VAEP value of an action is the change in P(score) minus the change
in P(concede) between the game state immediately before and after the action.

Formally, following Decroos et al. 2019:

    offensive_value = p_score_after  - p_score_before
    defensive_value = -(p_concede_after - p_concede_before)
    vaep_value      = offensive_value + defensive_value

"after" predictions come from the *next* action that the same team performs;
"before" predictions are the prediction immediately preceding the action.
For the first action of a possession sequence, "before" defaults to the
prediction itself (a value of zero), following socceraction conventions.

Output: ``data/processed/model_outputs/<model_id>_vaep_values.parquet``

Run:
    python -m src.vaep_values --model-id model_A
    python -m src.vaep_values --model-id model_A --model-id model_B --model-id model_C
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from .config import Config, load_config

log = logging.getLogger(__name__)


def compute_vaep_for_model(
    predictions: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute offensive, defensive, and total VAEP per action.

    Parameters
    ----------
    predictions :
        Must contain columns: match_id, action_id, p_score, p_concede,
        score_label, concede_label.
    actions :
        SPADL actions table — used to fetch player_id, team_id, type_name,
        period_id, time_seconds for joining/sorting.
    """
    join_cols = ["match_id", "action_id"]
    cols_from_actions = [
        "match_id", "action_id", "team_id", "player_id",
        "type_name", "result_name", "period_id", "time_seconds",
    ]
    df = predictions.merge(
        actions[cols_from_actions], on=join_cols, how="left"
    )
    df = df.sort_values(["match_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)

    # Compute "before" and "after" probabilities per action, within a match.
    # Following Decroos: the *before* state is the prediction of the previous
    # action; the *after* state is this action's own prediction.
    df["p_score_after"] = df["p_score"]
    df["p_concede_after"] = df["p_concede"]

    # Shift within match to get the previous action's prediction.
    grouped = df.groupby("match_id", sort=False)
    df["p_score_before"] = grouped["p_score"].shift(1)
    df["p_concede_before"] = grouped["p_concede"].shift(1)

    # The very first action of each match has no "before"; use the action's own
    # prediction (so the diff is zero, no value attributed).
    df["p_score_before"] = df["p_score_before"].fillna(df["p_score"])
    df["p_concede_before"] = df["p_concede_before"].fillna(df["p_concede"])

    # Note on possession changes: a strict socceraction implementation flips the
    # previous-action perspective when possession changes hands (so the "before"
    # state is described from the acting team's perspective). Several derived
    # VAEP papers use the simpler same-team-as-previous assumption, which is
    # what we do here for interpretability. This decision is documented in
    # Chapter 4 of the thesis.

    df["offensive_value"] = df["p_score_after"] - df["p_score_before"]
    df["defensive_value"] = -(df["p_concede_after"] - df["p_concede_before"])
    df["vaep_value"] = df["offensive_value"] + df["defensive_value"]

    return df


def assemble_all_models(cfg: Config, model_ids: list[str]) -> pd.DataFrame:
    """Stack per-model VAEP tables into one long-form artifact.

    Output columns include a ``model_id`` discriminator so downstream
    aggregations can pivot/filter.
    """
    actions = pd.read_parquet(cfg.paths.actions_dir / "actions_spadl.parquet")

    parts: list[pd.DataFrame] = []
    for mid in model_ids:
        pred_path = cfg.paths.model_outputs_dir / f"{mid}_predictions.parquet"
        if not pred_path.exists():
            log.warning("Predictions missing for model %s at %s — skipping", mid, pred_path)
            continue
        preds = pd.read_parquet(pred_path)
        vaep = compute_vaep_for_model(preds, actions)
        vaep["model_id"] = mid
        parts.append(vaep)
        log.info("Model %s: computed VAEP for %d actions", mid, len(vaep))

    if not parts:
        raise RuntimeError("No model predictions found to assemble.")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
@click.option(
    "--model-id", "model_ids", multiple=True, required=True,
    help="Model identifier(s) to compute VAEP for. Pass multiple times to assemble several into one file.",
)
def main(config_path: str | None, model_ids: tuple[str, ...]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("model_outputs_dir")

    all_vaep = assemble_all_models(cfg, list(model_ids))
    out_path = cfg.paths.model_outputs_dir / "all_models_vaep_values.parquet"
    all_vaep.to_parquet(out_path, index=False)
    log.info("Wrote %d VAEP rows (%d unique models) to %s",
             len(all_vaep), all_vaep["model_id"].nunique(), out_path)

    # Per-model summary CSV.
    summary = (
        all_vaep.groupby("model_id")
        .agg(
            n_actions=("vaep_value", "size"),
            mean_vaep=("vaep_value", "mean"),
            std_vaep=("vaep_value", "std"),
            max_vaep=("vaep_value", "max"),
            min_vaep=("vaep_value", "min"),
            mean_offensive=("offensive_value", "mean"),
            mean_defensive=("defensive_value", "mean"),
        )
        .reset_index()
    )
    summary_path = cfg.paths.tables_dir / "vaep_value_summary.csv"
    cfg.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    log.info("VAEP summary saved to %s\n%s", summary_path, summary.to_string(index=False))


if __name__ == "__main__":
    main()
