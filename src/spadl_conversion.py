"""Stage 4 — Convert StatsBomb events to SPADL actions.

Uses ``socceraction``'s reference converter so we inherit the canonical
SPADL semantics described in Decroos et al. 2019. Output columns follow
the promotor's plan §9.

The crucial side-output is the mapping ``action_id ↔ original_event_id``,
which is what later stages use to join 360 freeze-frame data onto actions.

Run:
    python -m src.spadl_conversion
    python -m src.spadl_conversion --competitions men_360_source
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import pandas as pd
from tqdm import tqdm

from . import data_loading as dl
from .config import Config, load_config
from .preprocessing import select_matches

log = logging.getLogger(__name__)


_SOCCERACTION_IMPORT_ERROR: Exception | None = None
try:
    from socceraction.data.statsbomb import StatsBombLoader
    from socceraction.spadl import statsbomb as ss
    from socceraction.spadl import config as spadlconfig
    _HAVE_SOCCERACTION = True
except Exception as _exc:  # pragma: no cover
    _HAVE_SOCCERACTION = False
    _SOCCERACTION_IMPORT_ERROR = _exc


def _add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Populate type_name / result_name / bodypart_name from their numeric IDs.

    socceraction's SPADL converter returns only the IDs; downstream code
    (notably ``socceraction.vaep.labels.scores`` and ``concedes``) compares
    against the string names. We add them here so labels and feature
    one-hots work as expected.
    """
    type_names = list(spadlconfig.actiontypes)
    result_names = list(spadlconfig.results)
    bodypart_names = list(spadlconfig.bodyparts)

    def _safe_lookup(idx, names: list[str]) -> str | None:
        if pd.isna(idx):
            return None
        try:
            return names[int(idx)]
        except (IndexError, ValueError, TypeError):
            return None

    actions["type_name"] = actions["type_id"].map(lambda x: _safe_lookup(x, type_names))
    actions["result_name"] = actions["result_id"].map(lambda x: _safe_lookup(x, result_names))
    actions["bodypart_name"] = actions["bodypart_id"].map(
        lambda x: _safe_lookup(x, bodypart_names)
    )
    return actions


SPADL_COLUMNS = [
    "match_id",
    "game_id",
    "original_event_id",
    "action_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "type_id",
    "type_name",
    "result_id",
    "result_name",
    "bodypart_id",
    "bodypart_name",
    "possession",
    "play_pattern",
]


def _ensure_socceraction() -> None:
    if not _HAVE_SOCCERACTION:
        raise ImportError(
            "Could not import socceraction. The underlying error was:\n"
            f"  {type(_SOCCERACTION_IMPORT_ERROR).__name__}: {_SOCCERACTION_IMPORT_ERROR}\n\n"
            "Check `pip show socceraction` -- if the package is installed, the "
            "submodule path may have changed in your version. See "
            "https://socceraction.readthedocs.io for the current API."
        ) from _SOCCERACTION_IMPORT_ERROR


def convert_match(loader: "StatsBombLoader", match_id: int, home_team_id: int) -> pd.DataFrame:
    """Run socceraction's SPADL converter for one match."""
    events = loader.events(game_id=match_id, load_360=False)
    actions = ss.convert_to_actions(events=events, home_team_id=home_team_id)

    # action_id within a match (preserves chronological order).
    actions = actions.reset_index(drop=True)
    actions["action_id"] = actions.index.astype("int64")

    actions["match_id"] = match_id
    # Preserve original event id for the 360 join.
    if "original_event_id" not in actions.columns and "event_id" in actions.columns:
        actions["original_event_id"] = actions["event_id"]

    # Populate string-name columns from their numeric IDs.
    actions = _add_names(actions)

    # Ensure all canonical columns exist (fill missing with NA).
    for c in SPADL_COLUMNS:
        if c not in actions.columns:
            actions[c] = pd.NA

    return actions[SPADL_COLUMNS]


def convert_matches(cfg: Config, matches: pd.DataFrame) -> pd.DataFrame:
    """Convert every match in ``matches`` and return one concatenated DataFrame."""
    _ensure_socceraction()
    # StatsBombLoader expects the directory holding competitions.json (the
    # `data/` subfolder of the cloned repo). `dl.data_root` resolves either layout.
    loader = StatsBombLoader(root=str(dl.data_root(cfg.paths.statsbomb_root)), getter="local")

    parts: list[pd.DataFrame] = []
    for m in tqdm(matches.itertuples(index=False), total=len(matches), desc="SPADL"):
        match_id = int(m.match_id)
        try:
            # We need the home_team_id; pull it from the match record itself.
            games = loader.games(competition_id=int(m.competition_id), season_id=int(m.season_id))
            row = games[games["game_id"] == match_id]
            if row.empty:
                log.warning("Match %s missing from loader.games(), skipping", match_id)
                continue
            home_team_id = int(row.iloc[0]["home_team_id"])
            actions = convert_match(loader, match_id, home_team_id)
        except Exception as exc:  # pragma: no cover - per-match resilience
            log.exception("SPADL conversion failed for match %s: %s", match_id, exc)
            continue
        parts.append(actions)

    if not parts:
        return pd.DataFrame(columns=SPADL_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def quality_checks(df: pd.DataFrame) -> dict[str, Any]:
    """Cheap QA per the promotor's plan §9."""
    if df.empty:
        return {"empty": True}
    return {
        "n_actions": len(df),
        "n_matches": df["match_id"].nunique(),
        "actions_per_match_mean": float(df.groupby("match_id").size().mean()),
        "x_min": float(df["start_x"].min()),
        "x_max": float(df["start_x"].max()),
        "y_min": float(df["start_y"].min()),
        "y_max": float(df["start_y"].max()),
        "top_type_names": df["type_name"].value_counts().head(15).to_dict(),
        "top_result_names": df["result_name"].value_counts().head(10).to_dict(),
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
    help="Competition group(s) to convert. Pass multiple times to union. "
         "Default: cfg.competitions.default_corpus.",
)
@click.option(
    "--gender", default=None, type=click.Choice(["male", "female"]), help="Filter by gender."
)
def main(
    config_path: str | None, competitions_groups: tuple[str, ...], gender: str | None
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("actions_dir", "tables_dir")

    groups: list[str] | None = list(competitions_groups) if competitions_groups else None
    matches = select_matches(cfg, competitions_group=groups, gender=gender)
    log.info("Converting %d matches to SPADL", len(matches))
    if matches.empty:
        log.warning("Nothing to do.")
        return

    actions = convert_matches(cfg, matches)
    out_path = cfg.paths.actions_dir / "actions_spadl.parquet"
    actions.to_parquet(out_path, index=False)
    log.info("Wrote %d actions to %s", len(actions), out_path)

    qc = quality_checks(actions)
    qc_path = cfg.paths.tables_dir / "actions_spadl_qc.csv"
    pd.DataFrame([{k: str(v) for k, v in qc.items()}]).to_csv(qc_path, index=False)
    log.info("Quality report saved to %s", qc_path)


if __name__ == "__main__":
    main()
