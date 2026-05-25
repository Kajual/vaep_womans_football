"""Stage 16 — Case-study extraction for thesis §6.4.

Selects a small number of individual actions that best illustrate *why* the
context-aware models value play differently from the event-only baseline, and
assembles, for each, everything needed to discuss and visualise it:

  * the action itself (player, team, type, result, location, minute);
  * its game phase (Stage 8) and spatial features (Stages 10-11);
  * the scoring / conceding probabilities and the offensive / defensive / total
    VAEP under Models A, B and C;
  * the StatsBomb 360 freeze frame (every visible player's position) so the
    moment can be drawn on a pitch by ``visualisation.py``.

Three cases are selected automatically by default:

  1. ``space_rewards``   — the attacking action whose VAEP rises most from
     Model A to Model C: spatial context credits play the event-only model
     missed.
  2. ``space_discounts`` — the action whose VAEP falls most from A to C:
     spatial context discounts play the event-only model over-credited.
  3. ``phase_matters``   — the defensive action whose VAEP changes most from
     Model A to Model B: phase context re-values a defensive moment.

Specific actions can be forced with ``--action MATCH_ID:ACTION_ID`` (repeatable).

Outputs
-------
* ``outputs/tables/case_studies_summary.csv``    flat, one row per case
* ``outputs/tables/case_study_candidates.csv``   ranked shortlist per category
* ``outputs/reports/case_studies_detail.json``   full per-case data + freeze frame

Run
---
    python -m src.case_studies
    python -m src.case_studies --n-per-category 15
    python -m src.case_studies --action 3943043:1187 --action 3943077:455
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from . import data_loading as dl
from .config import Config, load_config

log = logging.getLogger(__name__)

# socceraction is used only to resolve player and team names. Without it the
# module still runs; names degrade to "player_<id>" / "team_<id>".
try:
    from socceraction.data.statsbomb import StatsBombLoader
    _HAVE_SOCCERACTION = True
except Exception:  # pragma: no cover
    _HAVE_SOCCERACTION = False


# Model identifiers as written by modelling.py / vaep_values.py.
MODEL_BASE = "model_A"    # event-only
MODEL_PHASE = "model_B"   # event + phase
MODEL_FULL = "model_C"    # event + phase + space

# SPADL pitch is 105 x 68; StatsBomb event / 360 space is 120 x 80.
SPADL_LENGTH, SPADL_WIDTH = 105.0, 68.0
SB_LENGTH, SB_WIDTH = 120.0, 80.0

# SPADL x of the final-third boundary (configs/default.yaml: final_third_min).
FINAL_THIRD_X_SPADL = 70.0

# Action types that signal a defensive moment (mirrors phase_features.py).
DEFENSIVE_ACTION_TYPES = {
    "tackle", "interception", "clearance", "block", "foul",
    "keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up",
}

# Half offsets (minutes) for converting period + seconds to a match minute.
PERIOD_OFFSET_MIN = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}

VAEP_METRICS = ["vaep_value", "offensive_value", "defensive_value", "p_score", "p_concede"]


# ---------------------------------------------------------------------------
# Loading and joining
# ---------------------------------------------------------------------------

def widen_vaep(vaep: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot the long-form VAEP table to one row per action, columns per model.

    Returns the wide table and the list of model ids found.
    """
    models = sorted(vaep["model_id"].dropna().unique())
    wide: pd.DataFrame | None = None
    for mid in models:
        sub = vaep.loc[vaep["model_id"] == mid, ["match_id", "action_id", *VAEP_METRICS]]
        sub = sub.rename(columns={m: f"{m}__{mid}" for m in VAEP_METRICS})
        wide = sub if wide is None else wide.merge(sub, on=["match_id", "action_id"], how="outer")
    if wide is None:
        raise RuntimeError("VAEP table is empty.")
    return wide, models


def assemble_candidates(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    """Build one wide table joining VAEP, action details, phase and space features."""
    vaep_path = cfg.paths.model_outputs_dir / "all_models_vaep_values.parquet"
    actions_path = cfg.paths.actions_dir / "actions_spadl.parquet"
    phase_path = cfg.paths.features_dir / "phase_labels.parquet"
    space_path = cfg.paths.features_dir / "space_features.parquet"
    for p in (vaep_path, actions_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    vaep = pd.read_parquet(vaep_path)
    wide, models = widen_vaep(vaep)
    log.info("VAEP: %d actions across models %s", len(wide), models)

    action_cols = [
        "match_id", "action_id", "original_event_id", "period_id", "time_seconds",
        "team_id", "player_id", "type_name", "result_name", "bodypart_name",
        "start_x", "start_y", "end_x", "end_y", "possession", "play_pattern",
    ]
    actions = pd.read_parquet(actions_path)
    actions = actions[[c for c in action_cols if c in actions.columns]]
    df = wide.merge(actions, on=["match_id", "action_id"], how="left")

    if phase_path.exists():
        phases = pd.read_parquet(phase_path)[["match_id", "action_id", "phase"]]
        df = df.merge(phases, on=["match_id", "action_id"], how="left")
    else:
        log.warning("Phase labels not found at %s — phase will be blank.", phase_path)
        df["phase"] = None

    if space_path.exists():
        space = pd.read_parquet(space_path)
        df = df.merge(space, on=["match_id", "action_id"], how="left", suffixes=("", "_sp"))
    else:
        log.warning("Space features not found at %s — case selection will be limited.", space_path)

    return df, models


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------

def _attacking_area(df: pd.DataFrame) -> pd.Series:
    """Boolean: action is in/around the final third or penalty box."""
    in_final_third = df.get("start_x", pd.Series(np.nan, index=df.index)) >= FINAL_THIRD_X_SPADL
    box = df.get("is_box_action", pd.Series(0, index=df.index)).fillna(0) > 0
    entry = df.get("is_final_third_entry", pd.Series(0, index=df.index)).fillna(0) > 0
    return in_final_third.fillna(False) | box | entry


def select_auto(
    df: pd.DataFrame, models: list[str], n_per_category: int
) -> tuple[list[dict], pd.DataFrame]:
    """Pick the headline cases and return them plus a ranked shortlist table."""
    needed = {MODEL_BASE, MODEL_PHASE, MODEL_FULL}
    if not needed.issubset(set(models)):
        raise RuntimeError(
            f"Auto-selection needs models {sorted(needed)}; found {models}. "
            "Use --action to pick cases manually."
        )

    cand = df.copy()
    # Keep only actions valued by all three models and carrying a freeze frame.
    cand = cand[cand[[f"vaep_value__{m}" for m in needed]].notna().all(axis=1)]
    if "nearest_opponent_distance" in cand.columns:
        cand = cand[cand["nearest_opponent_distance"].notna()]
    if "original_event_id" in cand.columns:
        cand = cand[cand["original_event_id"].notna()]
    log.info("Case-selection pool after filtering: %d actions", len(cand))
    if cand.empty:
        raise RuntimeError("No actions survive case-selection filtering.")

    cand["delta_vaep_CA"] = cand[f"vaep_value__{MODEL_FULL}"] - cand[f"vaep_value__{MODEL_BASE}"]
    cand["delta_vaep_BA"] = cand[f"vaep_value__{MODEL_PHASE}"] - cand[f"vaep_value__{MODEL_BASE}"]
    cand["abs_delta_BA"] = cand["delta_vaep_BA"].abs()
    atk = _attacking_area(cand)

    shortlists: dict[str, pd.DataFrame] = {
        # Spatial context credits an attacking action the baseline missed.
        "space_rewards": cand[atk].sort_values("delta_vaep_CA", ascending=False),
        # Spatial context discounts an action the baseline over-valued.
        "space_discounts": cand[~atk].sort_values("delta_vaep_CA", ascending=True),
        # Phase context re-values a defensive moment.
        "phase_matters": cand[cand["type_name"].isin(DEFENSIVE_ACTION_TYPES)]
        .sort_values("abs_delta_BA", ascending=False),
    }

    picks: list[dict] = []
    used: set[tuple] = set()
    for label, sl in shortlists.items():
        for row in sl.itertuples(index=False):
            key = (int(row.match_id), int(row.action_id))
            if key in used:
                continue
            picks.append({"case_label": label, "match_id": key[0], "action_id": key[1]})
            used.add(key)
            break
        else:
            log.warning("No candidate found for category '%s'.", label)

    # Ranked shortlist table for manual curation.
    keep = [
        "match_id", "action_id", "type_name", "result_name", "phase",
        "start_x", "start_y", "end_x", "end_y",
        "nearest_opponent_distance", "is_box_action",
        f"vaep_value__{MODEL_BASE}", f"vaep_value__{MODEL_PHASE}", f"vaep_value__{MODEL_FULL}",
        "delta_vaep_CA", "delta_vaep_BA",
    ]
    parts = []
    for label, sl in shortlists.items():
        head = sl.head(n_per_category).copy()
        head.insert(0, "category", label)
        parts.append(head[["category", *[c for c in keep if c in head.columns]]])
    candidates_table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return picks, candidates_table


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def build_lookups(
    cfg: Config, match_ids: list[int]
) -> tuple[dict[int, str], dict[int, str]]:
    """Return (player_id -> name, team_id -> name) for the given matches."""
    player_map: dict[int, str] = {}
    team_map: dict[int, str] = {}
    if not _HAVE_SOCCERACTION:
        log.warning("socceraction unavailable — player/team names will be placeholders.")
        return player_map, team_map
    loader = StatsBombLoader(root=str(dl.data_root(cfg.paths.statsbomb_root)), getter="local")
    for mid in match_ids:
        try:
            players = loader.players(game_id=int(mid))
            player_map.update(dict(zip(players["player_id"], players["player_name"])))
        except Exception as exc:  # pragma: no cover
            log.warning("loader.players failed for match %s: %s", mid, exc)
        try:
            teams = loader.teams(game_id=int(mid))
            team_map.update(dict(zip(teams["team_id"], teams["team_name"])))
        except Exception as exc:  # pragma: no cover
            log.warning("loader.teams failed for match %s: %s", mid, exc)
    return player_map, team_map


# ---------------------------------------------------------------------------
# Freeze-frame extraction
# ---------------------------------------------------------------------------

def extract_freeze_frame(cfg: Config, match_id: int, event_uuid: str) -> dict[str, Any] | None:
    """Return the 360 freeze frame for one event as plain, plottable data."""
    frames = dl.load_360(cfg.paths.statsbomb_root, match_id)
    if not frames:
        return None
    frame = next((f for f in frames if f.get("event_uuid") == event_uuid), None)
    if frame is None:
        return None
    players: list[dict[str, Any]] = []
    for p in frame.get("freeze_frame", []) or []:
        loc = p.get("location") or [None, None]
        if loc[0] is None:
            continue
        if p.get("actor", False):
            role = "actor"
        elif p.get("teammate", False):
            role = "teammate"
        else:
            role = "opponent"
        players.append({
            "x": float(loc[0]), "y": float(loc[1]),
            "role": role, "keeper": bool(p.get("keeper", False)),
        })
    return {"players": players, "visible_area": frame.get("visible_area")}


# ---------------------------------------------------------------------------
# Per-case assembly
# ---------------------------------------------------------------------------

def _minute(period_id: Any, time_seconds: Any) -> float | None:
    if pd.isna(period_id) or pd.isna(time_seconds):
        return None
    return round(PERIOD_OFFSET_MIN.get(int(period_id), 0) + float(time_seconds) / 60.0, 2)


def _spadl_to_sb(x: float, y: float) -> tuple[float, float]:
    return x / SPADL_LENGTH * SB_LENGTH, y / SPADL_WIDTH * SB_WIDTH


def build_case(
    cfg: Config,
    case_id: int,
    case_label: str,
    row: pd.Series,
    models: list[str],
    player_map: dict[int, str],
    team_map: dict[int, str],
    match_team_ids: dict[int, list[int]],
) -> dict[str, Any]:
    """Assemble the full detail dict for one case."""
    mid = int(row["match_id"])
    aid = int(row["action_id"])
    team_id = int(row["team_id"]) if pd.notna(row.get("team_id")) else None
    player_id = int(row["player_id"]) if pd.notna(row.get("player_id")) else None

    team_ids = match_team_ids.get(mid, [])
    teams = [team_map.get(t, f"team_{t}") for t in team_ids]
    opp_name = next((team_map.get(t, f"team_{t}") for t in team_ids if t != team_id), None)

    space_cols = [
        "nearest_opponent_distance", "nearest_teammate_distance",
        "n_opponents_visible", "n_teammates_visible",
        "n_opponents_ahead_of_ball", "n_opponents_between_ball_and_goal",
        "defensive_density_around_ball_5m", "defensive_density_around_ball_10m",
        "defensive_density_around_ball_15m", "defensive_density_in_front_of_ball",
        "is_central_zone", "is_half_space_left", "is_half_space_right",
        "is_wide_zone", "is_box_action", "is_box_entry", "is_final_third_entry",
    ]
    space_features = {
        c: (None if pd.isna(row.get(c)) else round(float(row[c]), 3))
        for c in space_cols if c in row.index
    }

    per_model: dict[str, dict[str, float | None]] = {}
    for m in models:
        per_model[m] = {
            metric: (None if pd.isna(row.get(f"{metric}__{m}"))
                     else round(float(row[f"{metric}__{m}"]), 6))
            for metric in VAEP_METRICS
        }

    start_x = float(row["start_x"]) if pd.notna(row.get("start_x")) else None
    start_y = float(row["start_y"]) if pd.notna(row.get("start_y")) else None
    end_x = float(row["end_x"]) if pd.notna(row.get("end_x")) else None
    end_y = float(row["end_y"]) if pd.notna(row.get("end_y")) else None

    action = {
        "match_id": mid,
        "action_id": aid,
        "original_event_id": row.get("original_event_id"),
        "minute": _minute(row.get("period_id"), row.get("time_seconds")),
        "period_id": None if pd.isna(row.get("period_id")) else int(row["period_id"]),
        "type_name": row.get("type_name"),
        "result_name": row.get("result_name"),
        "bodypart_name": row.get("bodypart_name"),
        "phase": row.get("phase"),
        "play_pattern": row.get("play_pattern"),
        "player_id": player_id,
        "player_name": player_map.get(player_id, f"player_{player_id}") if player_id else None,
        "team_id": team_id,
        "team_name": team_map.get(team_id, f"team_{team_id}") if team_id else None,
        "opponent_name": opp_name,
        "match_label": " vs ".join(teams) if teams else f"match_{mid}",
        "start_spadl": [start_x, start_y],
        "end_spadl": [end_x, end_y],
    }
    # StatsBomb-space coordinates so the arrow overlays the freeze frame.
    if start_x is not None and start_y is not None:
        action["start_sb"] = list(_spadl_to_sb(start_x, start_y))
    if end_x is not None and end_y is not None:
        action["end_sb"] = list(_spadl_to_sb(end_x, end_y))

    frame = None
    if pd.notna(row.get("original_event_id")):
        frame = extract_freeze_frame(cfg, mid, str(row["original_event_id"]))
    if frame is None:
        log.warning("Case %d (%s): no freeze frame for %s/%s", case_id, case_label, mid, aid)

    def _delta(model_hi: str, model_lo: str) -> float | None:
        hi = per_model.get(model_hi, {}).get("vaep_value")
        lo = per_model.get(model_lo, {}).get("vaep_value")
        if hi is None or lo is None:
            return None
        return round(hi - lo, 6)

    return {
        "case_id": case_id,
        "case_label": case_label,
        "action": action,
        "space_features": space_features,
        "vaep_by_model": per_model,
        "delta_vaep_CA": _delta(MODEL_FULL, MODEL_BASE),
        "delta_vaep_BA": _delta(MODEL_PHASE, MODEL_BASE),
        "freeze_frame": frame,
    }


def case_to_summary_row(case: dict[str, Any]) -> dict[str, Any]:
    """Flatten one case into a single CSV row for the thesis table."""
    a = case["action"]
    sf = case["space_features"]
    vm = case["vaep_by_model"]

    def mv(model: str, metric: str) -> float | None:
        return vm.get(model, {}).get(metric)

    return {
        "case_id": case["case_id"],
        "case_label": case["case_label"],
        "match": a.get("match_label"),
        "minute": a.get("minute"),
        "player": a.get("player_name"),
        "team": a.get("team_name"),
        "action_type": a.get("type_name"),
        "result": a.get("result_name"),
        "phase": a.get("phase"),
        "nearest_opponent_distance_m": sf.get("nearest_opponent_distance"),
        "nearest_teammate_distance_m": sf.get("nearest_teammate_distance"),
        "defensive_density_5m": sf.get("defensive_density_around_ball_5m"),
        "defensive_density_10m": sf.get("defensive_density_around_ball_10m"),
        "n_opp_between_ball_and_goal": sf.get("n_opponents_between_ball_and_goal"),
        "is_box_action": sf.get("is_box_action"),
        "p_score_A": mv(MODEL_BASE, "p_score"),
        "p_score_B": mv(MODEL_PHASE, "p_score"),
        "p_score_C": mv(MODEL_FULL, "p_score"),
        "p_concede_A": mv(MODEL_BASE, "p_concede"),
        "p_concede_B": mv(MODEL_PHASE, "p_concede"),
        "p_concede_C": mv(MODEL_FULL, "p_concede"),
        "offensive_A": mv(MODEL_BASE, "offensive_value"),
        "offensive_B": mv(MODEL_PHASE, "offensive_value"),
        "offensive_C": mv(MODEL_FULL, "offensive_value"),
        "defensive_A": mv(MODEL_BASE, "defensive_value"),
        "defensive_B": mv(MODEL_PHASE, "defensive_value"),
        "defensive_C": mv(MODEL_FULL, "defensive_value"),
        "vaep_A": mv(MODEL_BASE, "vaep_value"),
        "vaep_B": mv(MODEL_PHASE, "vaep_value"),
        "vaep_C": mv(MODEL_FULL, "vaep_value"),
        "delta_vaep_CA": case.get("delta_vaep_CA"),
        "delta_vaep_BA": case.get("delta_vaep_BA"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_action(spec: str) -> tuple[int, int]:
    try:
        m, a = spec.split(":")
        return int(m), int(a)
    except Exception as exc:
        raise click.BadParameter(f"--action must be MATCH_ID:ACTION_ID, got '{spec}'") from exc


@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
@click.option("--n-per-category", default=15, type=int,
              help="Rows per category written to the candidates shortlist CSV.")
@click.option("--action", "actions", multiple=True,
              help="Force a specific case as MATCH_ID:ACTION_ID. Repeatable; "
                   "overrides automatic selection.")
def main(config_path: str | None, n_per_category: int, actions: tuple[str, ...]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("tables_dir", "reports_dir")

    df, models = assemble_candidates(cfg)
    df = df.set_index(["match_id", "action_id"], drop=False)

    # --- choose cases -----------------------------------------------------
    if actions:
        picks = []
        for i, spec in enumerate(actions, start=1):
            mid, aid = _parse_action(spec)
            picks.append({"case_label": f"manual_{i}", "match_id": mid, "action_id": aid})
        candidates_table = pd.DataFrame()
    else:
        picks, candidates_table = select_auto(df, models, n_per_category)

    if not candidates_table.empty:
        cand_path = cfg.paths.tables_dir / "case_study_candidates.csv"
        candidates_table.round(5).to_csv(cand_path, index=False)
        log.info("Wrote candidate shortlist (%d rows) to %s", len(candidates_table), cand_path)

    # --- name lookups for the involved matches ----------------------------
    match_ids = sorted({p["match_id"] for p in picks})
    player_map, team_map = build_lookups(cfg, match_ids)
    match_team_ids: dict[int, list[int]] = {}
    for mid in match_ids:
        sub = df[df["match_id"] == mid]
        match_team_ids[mid] = sorted(int(t) for t in sub["team_id"].dropna().unique())

    # --- assemble each case ----------------------------------------------
    cases: list[dict[str, Any]] = []
    for case_id, pick in enumerate(picks, start=1):
        key = (pick["match_id"], pick["action_id"])
        if key not in df.index:
            log.error("Action %s not found in the VAEP table — skipping.", key)
            continue
        row = df.loc[key]
        if isinstance(row, pd.DataFrame):  # duplicate index guard
            row = row.iloc[0]
        cases.append(build_case(
            cfg, case_id, pick["case_label"], row, models,
            player_map, team_map, match_team_ids,
        ))

    if not cases:
        log.error("No cases could be assembled.")
        return

    # --- write outputs ----------------------------------------------------
    detail_path = cfg.paths.reports_dir / "case_studies_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False, default=str)
    log.info("Wrote %d-case detail to %s", len(cases), detail_path)

    summary = pd.DataFrame([case_to_summary_row(c) for c in cases])
    summary_path = cfg.paths.tables_dir / "case_studies_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Wrote case-study summary to %s\n%s", summary_path, summary.to_string(index=False))


if __name__ == "__main__":
    main()
