"""Broadcast-coverage features from the StatsBomb 360 visible-area polygon.

Reviewer 2 (Comment 6) points out that freeze-frame counts depend on which
players the broadcast camera happened to capture, and that coverage varies
systematically with pitch position. If coverage were driving the valuation
changes we report, the redistribution would reflect camera framing rather than
defensive structure.

Testing that needs coverage itself to be measured. This module extracts three
measures per event, as promised in Section 3.1 of the manuscript:

    visible_area_m2          area of the visible-area polygon, in square metres
    visible_area_frac        that area as a fraction of the full pitch
    actor_dist_to_boundary_m distance from the actor to the polygon edge
    n_visible_players        players recorded in the freeze frame

The last is already implied by the spatial features, but is repeated here so
the coverage block is self-contained.

An actor near the boundary of the visible area is one whose surroundings are
partly outside the camera view, so the opponent counts around her are
under-estimates. That distance is therefore the most direct per-action measure
of how trustworthy the spatial features are.

Run:
    python -m src.coverage_features
Output:
    data/processed/features/coverage_features.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import data_loading as dl
from .config import Config, load_config
from .preprocessing import select_matches
from .space_features import PITCH_LENGTH_M, PITCH_WIDTH_M, X_SCALE, Y_SCALE

log = logging.getLogger(__name__)

PITCH_AREA_M2 = PITCH_LENGTH_M * PITCH_WIDTH_M


def _polygon_area_m2(points: list[tuple[float, float]]) -> float:
    """Shoelace area of the visible-area polygon, converted to square metres."""
    if len(points) < 3:
        return float("nan")
    xs = np.array([p[0] for p in points]) * X_SCALE
    ys = np.array([p[1] for p in points]) * Y_SCALE
    return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def _point_to_segment_m(px: float, py: float,
                        ax: float, ay: float,
                        bx: float, by: float) -> float:
    """Distance in metres from a point to a segment, in scaled coordinates."""
    px, ax, bx = px * X_SCALE, ax * X_SCALE, bx * X_SCALE
    py, ay, by = py * Y_SCALE, ay * Y_SCALE, by * Y_SCALE
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))


def _dist_to_boundary_m(px: float, py: float,
                        points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    return min(
        _point_to_segment_m(px, py, points[i][0], points[i][1],
                            points[(i + 1) % len(points)][0], points[(i + 1) % len(points)][1])
        for i in range(len(points))
    )


def _parse_visible_area(raw: Any) -> list[tuple[float, float]]:
    """StatsBomb stores the polygon as a flat [x1, y1, x2, y2, ...] list."""
    if not raw:
        return []
    flat = list(raw)
    if len(flat) >= 2 and isinstance(flat[0], (list, tuple)):
        return [(float(p[0]), float(p[1])) for p in flat if p and p[0] is not None]
    if len(flat) % 2 != 0:
        flat = flat[:-1]
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]


def _coverage_from_frame(frame: dict[str, Any]) -> dict[str, float]:
    players = frame.get("freeze_frame", []) or []
    poly = _parse_visible_area(frame.get("visible_area"))

    actor = None
    for p in players:
        loc = p.get("location") or [None, None]
        if loc[0] is None:
            continue
        if p.get("actor", False):
            actor = (float(loc[0]), float(loc[1]))
            break

    area = _polygon_area_m2(poly)
    return {
        "visible_area_m2": area,
        "visible_area_frac": area / PITCH_AREA_M2 if np.isfinite(area) else float("nan"),
        "actor_dist_to_boundary_m": (
            _dist_to_boundary_m(actor[0], actor[1], poly) if actor and poly else float("nan")
        ),
        "n_visible_players": float(len([
            p for p in players if (p.get("location") or [None])[0] is not None
        ])),
    }


def process_match(statsbomb_root: Path, match_id: int,
                  actions_for_match: pd.DataFrame) -> pd.DataFrame:
    frames = dl.load_360(statsbomb_root, match_id)
    if not frames:
        return pd.DataFrame()
    lookup = {f["event_uuid"]: f for f in frames if "event_uuid" in f}

    rows: list[dict[str, Any]] = []
    for action in actions_for_match.itertuples(index=False):
        eid = getattr(action, "original_event_id")
        if pd.isna(eid) or eid not in lookup:
            continue
        feats = _coverage_from_frame(lookup[eid])
        feats["match_id"] = int(action.match_id)
        feats["action_id"] = int(action.action_id)
        rows.append(feats)
    return pd.DataFrame(rows)


@click.command()
@click.option("--config", "config_path", default=None)
def main(config_path: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("features_dir")

    actions = pd.read_parquet(cfg.paths.actions_dir / "actions_spadl.parquet")
    inv = select_matches(cfg)
    with_360 = inv[inv["has_360"]] if "has_360" in inv.columns else inv
    match_ids = sorted(set(with_360["match_id"].astype(int)) & set(actions["match_id"].unique()))
    log.info("Extracting coverage for %d matches with 360 data", len(match_ids))

    root = Path(cfg.paths.statsbomb_root)
    parts = []
    for mid in tqdm(match_ids, desc="coverage"):
        sub = actions[actions["match_id"] == mid]
        if sub.empty:
            continue
        try:
            part = process_match(root, mid, sub)
        except Exception as exc:  # noqa: BLE001 — one bad match must not abort
            log.warning("Match %s failed: %s", mid, exc)
            continue
        if not part.empty:
            parts.append(part)

    if not parts:
        log.error("No coverage features extracted.")
        return

    out = pd.concat(parts, ignore_index=True)
    path = cfg.paths.features_dir / "coverage_features.parquet"
    out.to_parquet(path, index=False)
    log.info("Wrote %d rows to %s", len(out), path)
    log.info("Summary:\n%s", out[[
        "visible_area_frac", "actor_dist_to_boundary_m", "n_visible_players"
    ]].describe().to_string())


if __name__ == "__main__":
    main()
