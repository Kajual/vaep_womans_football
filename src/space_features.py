"""Stages 10–11 — Spatial features from StatsBomb 360 freeze frames.

For every action that has a 360 frame, we extract a small set of interpretable
quantities describing the spatial context at the moment of the action:

  * nearest_opponent_distance      (pressure proxy)
  * nearest_teammate_distance      (support proxy)
  * n_opponents_visible            (sanity check on broadcast coverage)
  * n_teammates_visible            (same)
  * n_opponents_ahead_of_ball      (defensive line shape, in attacking direction)
  * n_opponents_between_ball_and_goal
  * defensive_density_around_ball_5m
  * defensive_density_around_ball_10m
  * defensive_density_around_ball_15m
  * defensive_density_in_front_of_ball   (forward cone, 60° half-angle)
  * is_central_zone                (binary: y in [27, 53])
  * is_half_space_left             (binary)
  * is_half_space_right            (binary)
  * is_wide_zone                   (binary)
  * is_box_entry                   (binary: action ends in penalty box)
  * is_final_third_entry           (binary: action crosses x = 80)

Frame coordinates are in StatsBomb event-space (0–120 × 0–80), from the
acting team's perspective: the attacking goal is at x = 120. Distances are
reported in metres assuming the canonical 105 × 68 pitch dimensions.

Outputs
-------
* ``data/processed/features/space_features.parquet`` — features keyed by (match_id, action_id)
* ``outputs/tables/space_features_qc.csv`` — coverage and value-range QA

Run:
    python -m src.space_features
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import data_loading as dl
from .config import Config, load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry constants (StatsBomb event coordinates)
# ---------------------------------------------------------------------------
# StatsBomb pitch: 120 (length) × 80 (width).
# Real pitch:      105m × 68m.
SB_LENGTH = 120.0
SB_WIDTH = 80.0
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
X_SCALE = PITCH_LENGTH_M / SB_LENGTH      # metres per StatsBomb-x unit
Y_SCALE = PITCH_WIDTH_M / SB_WIDTH        # metres per StatsBomb-y unit

# Penalty box (attacking, opponent goal at x = 120):
BOX_X_MIN = 102.0
BOX_Y_MIN = 18.0
BOX_Y_MAX = 62.0
# Final third boundary:
FINAL_THIRD_X = 80.0
# Half spaces (rough channels between centre and wide zones):
HS_LEFT_Y = (18.0, 30.0)
HS_RIGHT_Y = (50.0, 62.0)
CENTRAL_Y = (30.0, 50.0)


# ---------------------------------------------------------------------------
# Per-action feature extraction
# ---------------------------------------------------------------------------

def _dist_m(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points, returned in metres."""
    dx = (x2 - x1) * X_SCALE
    dy = (y2 - y1) * Y_SCALE
    return math.hypot(dx, dy)


def _features_from_frame(
    frame: dict[str, Any],
    end_x: float | None,
    end_y: float | None,
) -> dict[str, float]:
    """Compute the feature dict for one freeze frame.

    The frame dict is StatsBomb's per-event 360 record: it has a ``freeze_frame``
    list of {teammate, actor, keeper, location: [x, y]} entries.
    ``end_x`` / ``end_y`` are the action's end coordinates (StatsBomb event
    space) — used for box-entry and final-third-entry flags.
    """
    players = frame.get("freeze_frame", []) or []
    actor = None
    teammates: list[tuple[float, float, bool]] = []
    opponents: list[tuple[float, float, bool]] = []

    for p in players:
        loc = p.get("location") or [None, None]
        if loc[0] is None:
            continue
        x = float(loc[0])
        y = float(loc[1])
        is_keeper = bool(p.get("keeper", False))
        if p.get("actor", False):
            actor = (x, y)
        elif p.get("teammate", False):
            teammates.append((x, y, is_keeper))
        else:
            opponents.append((x, y, is_keeper))

    feats: dict[str, float] = {}

    # If we couldn't locate the actor, we can't compute pressure/density.
    # Return NaNs so the merge downstream still works.
    if actor is None:
        nan = float("nan")
        return {
            "nearest_opponent_distance": nan,
            "nearest_teammate_distance": nan,
            "n_opponents_visible": float(len(opponents)),
            "n_teammates_visible": float(len(teammates)),
            "n_opponents_ahead_of_ball": nan,
            "n_opponents_between_ball_and_goal": nan,
            "defensive_density_around_ball_5m": nan,
            "defensive_density_around_ball_10m": nan,
            "defensive_density_around_ball_15m": nan,
            "defensive_density_in_front_of_ball": nan,
            "is_central_zone": nan,
            "is_half_space_left": nan,
            "is_half_space_right": nan,
            "is_wide_zone": nan,
            "is_box_entry": nan,
            "is_final_third_entry": nan,
            "is_box_action": nan,
        }

    bx, by = actor

    # --- Distances to nearest others ---
    opp_dists = [_dist_m(bx, by, ox, oy) for ox, oy, _ in opponents] or [float("nan")]
    tm_dists = [_dist_m(bx, by, tx, ty) for tx, ty, _ in teammates] or [float("nan")]
    feats["nearest_opponent_distance"] = float(np.nanmin(opp_dists))
    feats["nearest_teammate_distance"] = float(np.nanmin(tm_dists))
    feats["n_opponents_visible"] = float(len(opponents))
    feats["n_teammates_visible"] = float(len(teammates))

    # --- Direction-aware opponent counts ---
    # The acting team's attacking goal is at x = 120 (StatsBomb convention).
    n_ahead = sum(1 for ox, _, _ in opponents if ox > bx)
    n_between = sum(
        1 for ox, oy, _ in opponents
        if ox > bx and BOX_Y_MIN <= oy <= BOX_Y_MAX
    )
    feats["n_opponents_ahead_of_ball"] = float(n_ahead)
    feats["n_opponents_between_ball_and_goal"] = float(n_between)

    # --- Defensive density (within radius) ---
    for r in (5.0, 10.0, 15.0):
        feats[f"defensive_density_around_ball_{int(r)}m"] = float(
            sum(1 for d in opp_dists if not math.isnan(d) and d <= r)
        )

    # --- Defensive density in a forward cone (60° half-angle) ---
    # Counts opponents within 15m and within ±30° of the line from ball to goal.
    goal_dir = np.array([SB_LENGTH - bx, 40.0 - by], dtype=float)  # ball -> goal centre
    norm = np.linalg.norm(goal_dir)
    if norm > 1e-6:
        goal_unit = goal_dir / norm
        fwd_count = 0
        for ox, oy, _ in opponents:
            d = _dist_m(bx, by, ox, oy)
            if d > 15.0 or d < 1e-6:
                continue
            v = np.array([ox - bx, oy - by], dtype=float)
            v_unit = v / np.linalg.norm(v)
            cos = float(np.clip(np.dot(v_unit, goal_unit), -1.0, 1.0))
            if cos >= math.cos(math.radians(30.0)):
                fwd_count += 1
        feats["defensive_density_in_front_of_ball"] = float(fwd_count)
    else:
        feats["defensive_density_in_front_of_ball"] = float("nan")

    # --- Zone indicators (based on ball position) ---
    feats["is_central_zone"] = float(CENTRAL_Y[0] <= by < CENTRAL_Y[1])
    feats["is_half_space_left"] = float(HS_LEFT_Y[0] <= by < HS_LEFT_Y[1])
    feats["is_half_space_right"] = float(HS_RIGHT_Y[0] < by <= HS_RIGHT_Y[1])
    feats["is_wide_zone"] = float(by < HS_LEFT_Y[0] or by > HS_RIGHT_Y[1])
    feats["is_box_action"] = float(bx >= BOX_X_MIN and BOX_Y_MIN <= by <= BOX_Y_MAX)

    # --- End-location flags (action's destination, not the ball position) ---
    feats["is_box_entry"] = float(
        end_x is not None and end_y is not None
        and end_x >= BOX_X_MIN
        and BOX_Y_MIN <= end_y <= BOX_Y_MAX
    )
    feats["is_final_third_entry"] = float(
        end_x is not None and end_x >= FINAL_THIRD_X
    )

    return feats


# ---------------------------------------------------------------------------
# Per-match processing
# ---------------------------------------------------------------------------

def process_match(
    statsbomb_root: Path,
    match_id: int,
    actions_for_match: pd.DataFrame,
    events_for_match: pd.DataFrame,
) -> pd.DataFrame:
    """Extract spatial features for one match's actions, where 360 is available."""
    frames = dl.load_360(statsbomb_root, match_id)
    if not frames:
        return pd.DataFrame()

    # Build event_id -> freeze frame lookup.
    frame_lookup = {f["event_uuid"]: f for f in frames if "event_uuid" in f}

    # event_id -> end_location from events_clean — used for box-entry flag.
    end_lookup: dict[str, tuple[float | None, float | None]] = {
        row["event_id"]: (row.get("end_location_x"), row.get("end_location_y"))
        for _, row in events_for_match.iterrows()
        if pd.notna(row.get("event_id"))
    }

    rows: list[dict[str, Any]] = []
    for action in actions_for_match.itertuples(index=False):
        eid = getattr(action, "original_event_id")
        if pd.isna(eid):
            continue
        if eid not in frame_lookup:
            continue
        end_x, end_y = end_lookup.get(eid, (None, None))
        feats = _features_from_frame(frame_lookup[eid], end_x, end_y)
        feats["match_id"] = int(action.match_id)
        feats["action_id"] = int(action.action_id)
        rows.append(feats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level routine
# ---------------------------------------------------------------------------

def build_all(cfg: Config) -> pd.DataFrame:
    """Compute spatial features for every action across every match with 360."""
    actions_path = cfg.paths.actions_dir / "actions_spadl.parquet"
    events_path = cfg.paths.events_clean_dir / "events_clean.parquet"
    inv_path = cfg.paths.inventory_dir / "inventory_dataset.parquet"
    for p in (actions_path, events_path, inv_path):
        if not p.exists():
            raise FileNotFoundError(p)

    inv = pd.read_parquet(inv_path)
    inv_360 = inv[inv["has_360"].fillna(False) & inv["has_events"].fillna(False)]
    matches_with_360 = set(int(m) for m in inv_360["match_id"].tolist())
    log.info("Matches with 360 coverage: %d", len(matches_with_360))

    actions = pd.read_parquet(actions_path)
    events = pd.read_parquet(events_path)

    # Filter to only the matches with 360.
    actions = actions[actions["match_id"].astype(int).isin(matches_with_360)]
    events = events[events["match_id"].astype(int).isin(matches_with_360)]
    log.info("Filtered actions: %d  |  filtered events: %d", len(actions), len(events))

    parts: list[pd.DataFrame] = []
    grouped_actions = dict(tuple(actions.groupby("match_id", sort=False)))
    grouped_events = dict(tuple(events.groupby("match_id", sort=False)))

    for mid in tqdm(sorted(matches_with_360), desc="Spatial features"):
        if mid not in grouped_actions:
            continue
        df = process_match(
            cfg.paths.statsbomb_root,
            mid,
            grouped_actions[mid],
            grouped_events.get(mid, pd.DataFrame()),
        )
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def quality_checks(features: pd.DataFrame, actions: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {"empty": True}
    coverage_pct = float(len(features) / len(actions) * 100)
    return {
        "n_actions_with_features": len(features),
        "n_actions_total": len(actions),
        "coverage_pct_of_all_actions": coverage_pct,
        "nearest_opp_dist_mean": float(features["nearest_opponent_distance"].mean()),
        "nearest_opp_dist_min": float(features["nearest_opponent_distance"].min()),
        "n_opps_visible_mean": float(features["n_opponents_visible"].mean()),
        "n_teammates_visible_mean": float(features["n_teammates_visible"].mean()),
        "is_box_action_pct": float(features["is_box_action"].mean() * 100),
        "is_final_third_entry_pct": float(features["is_final_third_entry"].mean() * 100),
        "missing_nearest_opp_pct": float(features["nearest_opponent_distance"].isna().mean() * 100),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
def main(config_path: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("features_dir", "tables_dir")

    features = build_all(cfg)
    if features.empty:
        log.error("No spatial features extracted. Check 360 coverage.")
        return

    out_path = cfg.paths.features_dir / "space_features.parquet"
    features.to_parquet(out_path, index=False)
    log.info("Wrote %d rows × %d cols to %s", len(features), features.shape[1], out_path)

    actions = pd.read_parquet(cfg.paths.actions_dir / "actions_spadl.parquet")
    qc = quality_checks(features, actions)
    qc_path = cfg.paths.tables_dir / "space_features_qc.csv"
    pd.DataFrame([{k: str(v) for k, v in qc.items()}]).to_csv(qc_path, index=False)
    log.info("Spatial-features QA:\n%s", pd.Series(qc).to_string())


if __name__ == "__main__":
    main()
