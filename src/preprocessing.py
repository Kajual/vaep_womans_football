"""Stage 3 — Clean and standardize StatsBomb event data.

Flattens nested event JSON into a single tidy DataFrame with the canonical
columns listed in the promotor's plan §8. The result is saved as
``data/interim/events_clean/events_clean.parquet`` and is the input to:

  * SPADL conversion (Stage 4)
  * phase labeling (Stage 8, needs ``possession`` and ``play_pattern``)
  * 360 join (Stage 10, needs ``event_id``)

Run:
    python -m src.preprocessing
    python -m src.preprocessing --competitions men_360_source
    python -m src.preprocessing --gender female
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import click
import pandas as pd
from tqdm import tqdm

from . import data_loading as dl
from .config import Config, load_config

log = logging.getLogger(__name__)


CANONICAL_COLUMNS = [
    "match_id",
    "competition_id",
    "season_id",
    "competition_gender",
    "period",
    "timestamp",
    "minute",
    "second",
    "elapsed_seconds",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "position_id",
    "position_name",
    "event_id",
    "event_type",
    "event_result",
    "possession",
    "possession_team_id",
    "play_pattern",
    "location_x",
    "location_y",
    "end_location_x",
    "end_location_y",
    "under_pressure",
    "shot_statsbomb_xg",
    "pass_length",
    "pass_angle",
    "pass_height",
    "pass_outcome",
    "carry_end_location_x",
    "carry_end_location_y",
]


# ---------------------------------------------------------------------------
# Event flattening
# ---------------------------------------------------------------------------

def _flatten_match_events(
    events: list[dict[str, Any]],
    match_id: int,
    competition_id: int,
    season_id: int,
    competition_gender: str,
) -> pd.DataFrame:
    """Convert one match's raw event list into a tidy DataFrame."""
    rows: list[dict[str, Any]] = []
    for ev in events:
        row = _flatten_one_event(ev)
        row["match_id"] = match_id
        row["competition_id"] = competition_id
        row["season_id"] = season_id
        row["competition_gender"] = competition_gender
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Compute elapsed_seconds = (period-1)*45*60 + minute*60 + second (best effort)
    # for cross-match-period ordering.
    df["elapsed_seconds"] = (
        (df["period"].fillna(1).astype(int) - 1) * 45 * 60
        + df["minute"].fillna(0).astype(int) * 60
        + df["second"].fillna(0).astype(int)
    )

    # Restrict to the canonical column set, preserving order.
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[CANONICAL_COLUMNS]


def _flatten_one_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Pull the canonical fields out of one nested StatsBomb event dict."""
    loc = ev.get("location") or [None, None]
    end_loc_x = end_loc_y = None
    pass_outcome = None
    pass_length = ev.get("pass", {}).get("length") if isinstance(ev.get("pass"), dict) else None
    pass_angle = ev.get("pass", {}).get("angle") if isinstance(ev.get("pass"), dict) else None
    pass_height = (
        ev.get("pass", {}).get("height", {}).get("name")
        if isinstance(ev.get("pass"), dict) and isinstance(ev.get("pass", {}).get("height"), dict)
        else None
    )
    shot_xg = ev.get("shot", {}).get("statsbomb_xg") if isinstance(ev.get("shot"), dict) else None
    carry_end_x = carry_end_y = None

    # Pass-specific.
    p = ev.get("pass")
    if isinstance(p, dict):
        end = p.get("end_location") or [None, None]
        end_loc_x, end_loc_y = (end + [None, None])[:2]
        outcome = p.get("outcome")
        pass_outcome = outcome.get("name") if isinstance(outcome, dict) else None

    # Shot-specific.
    s = ev.get("shot")
    if isinstance(s, dict):
        end = s.get("end_location") or [None, None, None]
        end_loc_x, end_loc_y = (end + [None, None])[:2]

    # Carry-specific.
    c = ev.get("carry")
    if isinstance(c, dict):
        end = c.get("end_location") or [None, None]
        carry_end_x, carry_end_y = (end + [None, None])[:2]
        # Carries also define an end_location; mirror it onto the generic
        # end_location columns for downstream consumers.
        if end_loc_x is None:
            end_loc_x, end_loc_y = carry_end_x, carry_end_y

    event_type = (ev.get("type") or {}).get("name")
    event_result = None
    # StatsBomb does not have a single "result" field; surface the most common
    # success/failure indicator per event type.
    if event_type == "Pass":
        event_result = "complete" if pass_outcome is None else pass_outcome.lower()
    elif event_type == "Shot" and isinstance(s, dict):
        outcome = s.get("outcome")
        event_result = outcome.get("name") if isinstance(outcome, dict) else None
    elif event_type in {"Duel", "50/50"}:
        outcome = (ev.get("duel") or {}).get("outcome")
        event_result = outcome.get("name") if isinstance(outcome, dict) else None
    elif event_type == "Dribble":
        outcome = (ev.get("dribble") or {}).get("outcome")
        event_result = outcome.get("name") if isinstance(outcome, dict) else None

    return {
        "event_id": ev.get("id"),
        "period": ev.get("period"),
        "timestamp": ev.get("timestamp"),
        "minute": ev.get("minute"),
        "second": ev.get("second"),
        "team_id": (ev.get("team") or {}).get("id"),
        "team_name": (ev.get("team") or {}).get("name"),
        "player_id": (ev.get("player") or {}).get("id"),
        "player_name": (ev.get("player") or {}).get("name"),
        "position_id": (ev.get("position") or {}).get("id"),
        "position_name": (ev.get("position") or {}).get("name"),
        "event_type": event_type,
        "event_result": event_result,
        "possession": ev.get("possession"),
        "possession_team_id": (ev.get("possession_team") or {}).get("id"),
        "play_pattern": (ev.get("play_pattern") or {}).get("name"),
        "location_x": loc[0] if loc else None,
        "location_y": loc[1] if loc else None,
        "end_location_x": end_loc_x,
        "end_location_y": end_loc_y,
        "under_pressure": ev.get("under_pressure", False),
        "shot_statsbomb_xg": shot_xg,
        "pass_length": pass_length,
        "pass_angle": pass_angle,
        "pass_height": pass_height,
        "pass_outcome": pass_outcome,
        "carry_end_location_x": carry_end_x,
        "carry_end_location_y": carry_end_y,
    }


# ---------------------------------------------------------------------------
# Match selection
# ---------------------------------------------------------------------------

def select_matches(
    cfg: Config,
    competitions_group: str | list[str] | None = None,
    gender: str | None = None,
) -> pd.DataFrame:
    """Return the slice of the inventory we should process this run.

    Parameters
    ----------
    competitions_group :
        Either ``None`` (use ``cfg.competitions.default_corpus`` — the union of
        every configured group), a single group name (string), or a list of
        group names. Group names must exist under ``cfg.competitions``.
    gender :
        Optional ``competition_gender`` filter ("male" or "female") applied
        on top of any competitions_group filter.
    """
    inv_path = cfg.paths.inventory_dir / "inventory_dataset.parquet"
    if not inv_path.exists():
        raise FileNotFoundError(
            f"Inventory not found at {inv_path}. Run `python -m src.data_inventory` first."
        )
    inv = pd.read_parquet(inv_path)
    inv = inv[inv["has_events"].fillna(False)]

    if gender is not None:
        inv = inv[inv["competition_gender"] == gender]

    # Resolve groups: None -> default_corpus, str -> [str], list -> list.
    if competitions_group is None:
        groups = list(cfg.competitions.default_corpus)
    elif isinstance(competitions_group, str):
        groups = [competitions_group]
    else:
        groups = list(competitions_group)

    wanted: set[tuple[str, str]] = set()
    for g_name in groups:
        if g_name not in cfg.competitions:
            raise KeyError(
                f"competitions group '{g_name}' not found in config. "
                f"Available: {list(cfg.competitions.keys())}"
            )
        for entry in cfg.competitions[g_name]:
            wanted.add((entry["name"], str(entry["season"])))

    inv = inv[
        inv.apply(
            lambda r: (r["competition_name"], str(r["season_name"])) in wanted, axis=1
        )
    ]
    return inv.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level processing
# ---------------------------------------------------------------------------

def process_matches(cfg: Config, matches: pd.DataFrame) -> pd.DataFrame:
    """Flatten the events of every match in ``matches`` and concatenate."""
    parts: list[pd.DataFrame] = []
    for m in tqdm(matches.itertuples(index=False), total=len(matches), desc="Cleaning events"):
        try:
            events = dl.load_events(cfg.paths.statsbomb_root, int(m.match_id))
        except FileNotFoundError as e:
            log.warning("%s", e)
            continue
        df = _flatten_match_events(
            events,
            match_id=int(m.match_id),
            competition_id=int(m.competition_id),
            season_id=int(m.season_id),
            competition_gender=str(m.competition_gender),
        )
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def quality_checks(df: pd.DataFrame) -> dict[str, Any]:
    """Cheap QA per the promotor's plan §8.

    Reports event counts per match, missing locations, missing player/team,
    distributions of event_type and play_pattern, x/y coordinate ranges.
    """
    if df.empty:
        return {"empty": True}
    return {
        "n_events": len(df),
        "n_matches": df["match_id"].nunique(),
        "events_per_match_mean": float(df.groupby("match_id").size().mean()),
        "missing_location_pct": float(df["location_x"].isna().mean() * 100),
        "missing_player_pct": float(df["player_id"].isna().mean() * 100),
        "missing_team_pct": float(df["team_id"].isna().mean() * 100),
        "x_min": float(df["location_x"].min()),
        "x_max": float(df["location_x"].max()),
        "y_min": float(df["location_y"].min()),
        "y_max": float(df["location_y"].max()),
        "top_event_types": df["event_type"].value_counts().head(10).to_dict(),
        "top_play_patterns": df["play_pattern"].value_counts().head(10).to_dict(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
@click.option(
    "--competitions",
    "competitions_groups",
    multiple=True,
    help="Name of a competition group from the config. Pass multiple times "
         "to union groups. Default: cfg.competitions.default_corpus.",
)
@click.option(
    "--gender",
    default=None,
    type=click.Choice(["male", "female"]),
    help="Filter by competition gender.",
)
def main(config_path: str | None, competitions_groups: tuple[str, ...], gender: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("events_clean_dir", "tables_dir")

    groups: list[str] | None = list(competitions_groups) if competitions_groups else None
    matches = select_matches(cfg, competitions_group=groups, gender=gender)
    log.info("Selected %d matches for cleaning", len(matches))
    if matches.empty:
        log.warning("Nothing to do.")
        return

    df = process_matches(cfg, matches)
    log.info("Flattened %d events from %d matches", len(df), df["match_id"].nunique())

    out_path = cfg.paths.events_clean_dir / "events_clean.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Wrote events to %s", out_path)

    qc = quality_checks(df)
    qc_path = cfg.paths.tables_dir / "events_clean_qc.csv"
    pd.DataFrame([{k: str(v) for k, v in qc.items()}]).to_csv(qc_path, index=False)
    log.info("Quality report saved to %s", qc_path)


if __name__ == "__main__":
    main()
