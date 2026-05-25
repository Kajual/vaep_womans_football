"""Stage 8 — Game-phase annotation.

Assigns each SPADL action a phase label from the six-class taxonomy:

  * ``set_piece``             — corner, free kick, throw-in, goal kick, kick-off, keeper-distribution
  * ``transition_attack``     — first seconds of a possession that started after winning the ball
  * ``defensive_transition``  — defensive action immediately after losing the ball
  * ``final_third_creation``  — action that starts or ends in the attacking third
  * ``build_up``              — open-play action in the own third
  * ``progression``           — open-play action in the middle third (residual)

Rules in §13 of the promotor's plan; thresholds in ``configs/default.yaml``.

The labeler is rule-based on purpose: the rules are interpretable and the
distribution can be inspected by eye. The thesis discusses this trade-off
in Chapter 4.

Outputs
-------
* ``data/processed/features/phase_labels.parquet``  — match_id, action_id, phase
* ``data/processed/features/features_phase.parquet`` — one-hot phase columns
* ``outputs/tables/phase_distribution.csv``         — QC: phase frequency

Run:
    python -m src.phase_features
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from .config import Config, load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase rules — constants
# ---------------------------------------------------------------------------

PHASES = (
    "set_piece",
    "transition_attack",
    "defensive_transition",
    "final_third_creation",
    "build_up",
    "progression",
)

SET_PIECE_PLAY_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Throw In",
    "From Goal Kick",
    "From Kick Off",
    "From Keeper",
}

SET_PIECE_ACTION_TYPES = {
    "throw_in",
    "corner_crossed",
    "corner_short",
    "freekick_short",
    "freekick_crossed",
    "goalkick",
}

# Actions a team takes when it has *just won* the ball.
RECOVERY_ACTION_TYPES = {
    "tackle",
    "interception",
    "ball_recovery",
    "keeper_save",
    "keeper_pick_up",
    "keeper_punch",
    "keeper_claim",
}

# Defensive on-ball actions that signal "defensive transition" when they occur
# right after losing possession.
DEFENSIVE_ACTION_TYPES = {
    "tackle",
    "interception",
    "clearance",
    "block",
    "foul",
}


# ---------------------------------------------------------------------------
# Merging play_pattern and possession from events_clean
# ---------------------------------------------------------------------------

def merge_possession_metadata(actions: pd.DataFrame, events_clean: pd.DataFrame) -> pd.DataFrame:
    """Bring ``play_pattern`` and ``possession`` onto each action.

    socceraction's SPADL converter strips these fields. We recover them by
    joining on the ``original_event_id ↔ event_id`` relationship.
    """
    needed_cols = ["event_id", "play_pattern", "possession", "possession_team_id"]
    missing = [c for c in needed_cols if c not in events_clean.columns]
    if missing:
        raise KeyError(f"events_clean is missing columns: {missing}")
    e = events_clean[needed_cols].drop_duplicates(subset=["event_id"])
    merged = actions.merge(
        e,
        left_on="original_event_id",
        right_on="event_id",
        how="left",
        suffixes=("", "_evt"),
    )
    # Where actions table already had these columns as NaN placeholders,
    # the merge brings the real values.
    for col in ["play_pattern", "possession", "possession_team_id"]:
        if f"{col}_evt" in merged.columns:
            # Coalesce: prefer _evt value when our placeholder was NaN.
            merged[col] = merged[col].where(merged[col].notna(), merged[f"{col}_evt"])
            merged = merged.drop(columns=[f"{col}_evt"])
    merged = merged.drop(columns=["event_id"], errors="ignore")
    return merged


# ---------------------------------------------------------------------------
# Phase assignment
# ---------------------------------------------------------------------------

def assign_phase(
    actions: pd.DataFrame,
    final_third_x: float = 70.0,
    own_third_x: float = 35.0,
    transition_window_s: float = 5.0,
) -> pd.DataFrame:
    """Add a ``phase`` column to ``actions``.

    Expects: match_id, action_id, period_id, time_seconds, team_id,
    type_name, start_x, end_x, play_pattern, possession.

    Phase priority (later overrides earlier):
      1. zone-based (build_up / progression / final_third_creation)
      2. defensive_transition
      3. transition_attack
      4. set_piece
    """
    df = actions.sort_values(["match_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)

    # --- Per-action context ---
    poss_start_time = df.groupby(["match_id", "possession"])["time_seconds"].transform("min")
    df["_time_in_poss"] = df["time_seconds"] - poss_start_time
    df["_poss_first"] = df.groupby(["match_id", "possession"]).cumcount() == 0

    df["_prev_team"] = df.groupby("match_id")["team_id"].shift(1)
    df["_prev_type"] = df.groupby("match_id")["type_name"].shift(1)
    df["_team_changed"] = (df["_prev_team"] != df["team_id"]) & df["_prev_team"].notna()

    # First action's type per (match_id, possession). This is what tells us
    # whether the possession was won by a recovery / ball recovery / tackle
    # interception — i.e. whether this is a transition possession.
    poss_first_type = (
        df[df["_poss_first"]]
        .set_index(["match_id", "possession"])["type_name"]
    )
    df["_poss_first_type"] = pd.Series(
        df.set_index(["match_id", "possession"]).index.map(poss_first_type).to_numpy(),
        index=df.index,
    )

    # --- Default + zone-based ---
    phase = pd.Series("progression", index=df.index, dtype="object")
    in_final_third = (df["start_x"] >= final_third_x) | (df["end_x"] >= final_third_x)
    in_own_third = df["start_x"] < own_third_x
    phase.loc[in_final_third] = "final_third_creation"
    phase.loc[~in_final_third & in_own_third] = "build_up"

    # --- Defensive transition ---
    # Defensive action by this team in the own third right after the ball
    # changed hands. Imperfect (we'd want true tracking), but a reasonable
    # rule-based approximation.
    defensive_now = df["type_name"].isin(DEFENSIVE_ACTION_TYPES)
    def_trans = defensive_now & in_own_third & df["_team_changed"]
    phase.loc[def_trans] = "defensive_transition"

    # --- Transition attack ---
    # Possession started with a recovery-type action OR play_pattern marks
    # a counter-attack origin. Actions in the first `transition_window_s`
    # seconds of such possessions are transition_attack.
    poss_began_with_recovery = df["_poss_first_type"].isin(RECOVERY_ACTION_TYPES)
    is_counter = df["play_pattern"].fillna("") == "From Counter"
    trans_atk_window = df["_time_in_poss"] <= transition_window_s
    trans_atk_mask = trans_atk_window & (poss_began_with_recovery | is_counter)
    phase.loc[trans_atk_mask] = "transition_attack"

    # --- Set pieces (highest priority) ---
    # Only the actual restart action type is a set piece, OR the first action
    # of a possession that began from a set-piece play_pattern. Without the
    # "_poss_first" gate, set-piece bleeds across the entire follow-up
    # possession because StatsBomb's play_pattern tags the whole sequence.
    sp_by_type = df["type_name"].isin(SET_PIECE_ACTION_TYPES)
    sp_by_pattern_first = (
        df["_poss_first"]
        & df["play_pattern"].fillna("").isin(SET_PIECE_PLAY_PATTERNS)
    )
    sp_mask = sp_by_type | sp_by_pattern_first
    phase.loc[sp_mask] = "set_piece"

    df["phase"] = phase

    df = df.drop(columns=[
        "_time_in_poss", "_poss_first", "_prev_team", "_prev_type",
        "_team_changed", "_poss_first_type",
    ])
    return df


# ---------------------------------------------------------------------------
# Feature output (one-hot)
# ---------------------------------------------------------------------------

def build_phase_features(phase_labels: pd.DataFrame) -> pd.DataFrame:
    """Return one-hot encoded phase features keyed by (match_id, action_id)."""
    out = phase_labels[["match_id", "action_id", "phase"]].copy()
    for p in PHASES:
        out[f"phase_{p}"] = (out["phase"] == p).astype("int8")
    return out.drop(columns=["phase"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
def main(config_path: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("features_dir", "tables_dir")

    actions_path = cfg.paths.actions_dir / "actions_spadl.parquet"
    events_path = cfg.paths.events_clean_dir / "events_clean.parquet"
    if not actions_path.exists():
        raise FileNotFoundError(actions_path)
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    actions = pd.read_parquet(actions_path)
    events = pd.read_parquet(events_path)
    log.info("Loaded %d actions and %d events", len(actions), len(events))

    actions = merge_possession_metadata(actions, events)
    log.info("Merged play_pattern and possession metadata onto actions")

    final_third_x = float(cfg.phases.pitch_thirds_x["final_third_min"])
    own_third_x = float(cfg.phases.pitch_thirds_x["own_third_max"])
    transition_window_s = float(cfg.phases.transition_window_seconds)

    labeled = assign_phase(
        actions,
        final_third_x=final_third_x,
        own_third_x=own_third_x,
        transition_window_s=transition_window_s,
    )

    labels_out = labeled[["match_id", "action_id", "phase"]]
    labels_path = cfg.paths.features_dir / "phase_labels.parquet"
    labels_out.to_parquet(labels_path, index=False)
    log.info("Wrote %d phase labels to %s", len(labels_out), labels_path)

    feats = build_phase_features(labels_out)
    feats_path = cfg.paths.features_dir / "features_phase.parquet"
    feats.to_parquet(feats_path, index=False)
    log.info("Wrote phase features (%d cols) to %s", feats.shape[1], feats_path)

    # QC: phase distribution
    dist = labels_out["phase"].value_counts(normalize=False).reset_index()
    dist.columns = ["phase", "n_actions"]
    dist["pct"] = (dist["n_actions"] / len(labels_out) * 100).round(2)
    dist_path = cfg.paths.tables_dir / "phase_distribution.csv"
    dist.to_csv(dist_path, index=False)
    log.info("Phase distribution:\n%s", dist.to_string(index=False))


if __name__ == "__main__":
    main()
