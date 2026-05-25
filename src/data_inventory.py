"""Stage 1 — Data inventory.

Walks the local StatsBomb open-data clone and produces a single table listing
every match referenced in ``competitions.json``, annotated with:

  * competition / season metadata
  * gender (men / women)
  * whether event, lineup, and 360 files exist
  * event and 360-frame counts

The output drives match selection in Stage 2 and feeds Table 3.1 in Chapter 3
of the thesis.

Run:
    python -m src.data_inventory
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from . import data_loading as dl
from .config import load_config

log = logging.getLogger(__name__)


INVENTORY_COLUMNS = [
    "competition_id",
    "season_id",
    "competition_name",
    "season_name",
    "competition_gender",
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "has_events",
    "has_lineups",
    "has_360",
    "n_events",
    "n_360_frames",
]


def build_inventory(statsbomb_root: Path, *, count_events: bool = True) -> pd.DataFrame:
    """Return the inventory DataFrame for every match in the StatsBomb repo.

    Parameters
    ----------
    statsbomb_root :
        Path to the cloned StatsBomb open-data repository.
    count_events :
        If True, opens every event JSON to count events. Adds runtime but
        gives accurate counts. If False, ``n_events`` and ``n_360_frames``
        are populated with NaN.
    """
    matches = dl.iter_all_matches(statsbomb_root)
    if matches.empty:
        log.warning("No matches found under %s", statsbomb_root)
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    # Normalise the few nested-team columns that json_normalize creates.
    home_col = "home_team_home_team_name" if "home_team_home_team_name" in matches.columns else "home_team_name"
    away_col = "away_team_away_team_name" if "away_team_away_team_name" in matches.columns else "away_team_name"

    records: list[dict] = []
    iterator = tqdm(matches.itertuples(index=False), total=len(matches), desc="Inventorying")
    for m in iterator:
        match_id = int(getattr(m, "match_id"))
        has_evt = dl.has_events(statsbomb_root, match_id)
        has_lup = dl.has_lineups(statsbomb_root, match_id)
        has_360 = dl.has_360(statsbomb_root, match_id)

        n_evt = dl.count_events(statsbomb_root, match_id) if (count_events and has_evt) else pd.NA
        n_360 = dl.count_360_frames(statsbomb_root, match_id) if (count_events and has_360) else pd.NA

        records.append(
            {
                "competition_id": int(getattr(m, "competition_id")),
                "season_id": int(getattr(m, "season_id")),
                "competition_name": getattr(m, "competition_name"),
                "season_name": getattr(m, "season_name"),
                "competition_gender": getattr(m, "competition_gender", "unknown"),
                "match_id": match_id,
                "match_date": getattr(m, "match_date", None),
                "home_team": getattr(m, home_col, None),
                "away_team": getattr(m, away_col, None),
                "has_events": has_evt,
                "has_lineups": has_lup,
                "has_360": has_360,
                "n_events": n_evt,
                "n_360_frames": n_360,
            }
        )

    return pd.DataFrame.from_records(records, columns=INVENTORY_COLUMNS)


def summarise_inventory(inv: pd.DataFrame) -> pd.DataFrame:
    """Return a per-competition+season summary suitable for the thesis table.

    Columns: competition_name, season_name, gender, n_matches,
             n_matches_with_360, total_events, total_360_frames.
    """
    if inv.empty:
        return pd.DataFrame()
    grouped = inv.groupby(
        ["competition_name", "season_name", "competition_gender"], dropna=False
    )
    out = grouped.agg(
        n_matches=("match_id", "nunique"),
        n_matches_with_360=("has_360", "sum"),
        total_events=("n_events", "sum"),
        total_360_frames=("n_360_frames", "sum"),
    ).reset_index()
    return out.sort_values(
        ["competition_gender", "competition_name", "season_name"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
@click.option("--no-counts", is_flag=True, help="Skip per-match event/360 counts (faster).")
def main(config_path: str | None, no_counts: bool) -> None:
    """Build the data inventory and write it to disk."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("inventory_dir", "tables_dir")

    log.info("Scanning %s ...", cfg.paths.statsbomb_root)
    inv = build_inventory(cfg.paths.statsbomb_root, count_events=not no_counts)
    summary = summarise_inventory(inv)

    parquet_path = cfg.paths.inventory_dir / "inventory_dataset.parquet"
    csv_path = cfg.paths.tables_dir / "inventory_dataset.csv"
    summary_csv = cfg.paths.tables_dir / "inventory_summary.csv"

    inv.to_parquet(parquet_path, index=False)
    inv.to_csv(csv_path, index=False)
    summary.to_csv(summary_csv, index=False)

    log.info("Wrote %d matches to %s", len(inv), parquet_path)
    log.info("Wrote summary table to %s", summary_csv)

    if not inv.empty:
        # Quick sanity readout.
        women = inv[inv["competition_gender"] == "female"]
        men = inv[inv["competition_gender"] == "male"]
        log.info("Men's matches:   %d total, %d with 360", len(men), men["has_360"].sum())
        log.info("Women's matches: %d total, %d with 360", len(women), women["has_360"].sum())


if __name__ == "__main__":
    main()
