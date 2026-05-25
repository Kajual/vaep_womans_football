"""Stage 6 — Baseline action features.

Builds the canonical VAEP feature set following Decroos et al. 2019, using
socceraction's reference feature functions. The feature set is:

  * type / result / body-part one-hot
  * start and end coordinates, normalised to attacking direction
  * distance and angle to goal at start and end
  * movement vector (dx, dy)
  * elapsed time and time-delta to previous action
  * goalscore (current score state)
  * the same fields for the previous ``n_previous_actions`` actions

Run:
    python -m src.features_baseline
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from .config import Config, load_config

log = logging.getLogger(__name__)

_SOCCERACTION_IMPORT_ERROR: Exception | None = None
try:
    from socceraction.vaep import features as fs
    _HAVE_SOCCERACTION = True
except Exception as _exc:  # pragma: no cover
    _HAVE_SOCCERACTION = False
    _SOCCERACTION_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# pandas 2.x compatibility patch for socceraction.vaep.features.gamestates
# ---------------------------------------------------------------------------
# socceraction 1.5.x was written for pandas 1.x. Its `gamestates()` calls
# `df.fillna(series)`, which pandas 2.x rejects (`Must specify a fill 'value'
# or 'method'.`). We swap in a drop-in replacement that uses `fillna(dict)`,
# which is unambiguous across pandas versions. Behaviour is identical:
# for each (game_id, period_id) group, shift actions by `i`, and fill the
# first `i` (now-NaN) rows with the first action of the group.

def _gamestates_pd2_compat(actions: pd.DataFrame, nb_prev_actions: int = 3) -> list[pd.DataFrame]:
    """pandas-2-compatible reimplementation of socceraction's ``gamestates``.

    pandas 2.x rejects NaN values inside the fillna dict (validator error
    "Must specify a fill 'value' or 'method'."), so we strip NaN entries from
    the first-row dict before applying fillna. Columns whose first-row value
    was already NaN simply remain NaN, which matches the original semantics
    (you can't fill NaN with NaN meaningfully).
    """
    states: list[pd.DataFrame] = [actions]
    for i in range(1, nb_prev_actions + 1):
        parts: list[pd.DataFrame] = []
        for _, group in actions.groupby(["game_id", "period_id"], sort=False, group_keys=False):
            shifted = group.shift(i)
            if not group.empty:
                first_row = {k: v for k, v in group.iloc[0].to_dict().items() if pd.notna(v)}
                if first_row:
                    shifted = shifted.fillna(value=first_row)
            parts.append(shifted)
        prev_actions = pd.concat(parts, ignore_index=False)
        states.append(prev_actions)
    return states


if _HAVE_SOCCERACTION:
    fs.gamestates = _gamestates_pd2_compat  # type: ignore[attr-defined]


# socceraction feature transformer functions to apply.
# (Imported lazily inside _feature_functions to avoid import errors when
# socceraction is not installed.)
def _feature_functions() -> list:
    return [
        fs.actiontype_onehot,
        fs.result_onehot,
        fs.bodypart_onehot,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.startpolar,
        fs.endpolar,
        fs.team,
        fs.time,
        fs.time_delta,
        fs.goalscore,
    ]


def _ensure_socceraction() -> None:
    if not _HAVE_SOCCERACTION:
        raise ImportError(
            "socceraction is not installed. Run `pip install socceraction`."
        )


def build_match_features(
    match_actions: pd.DataFrame,
    home_team_id: int,
    n_previous_actions: int = 3,
) -> pd.DataFrame:
    """Compute the baseline feature DataFrame for one match's actions."""
    _ensure_socceraction()
    actions = match_actions.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(
        drop=True
    )
    gamestates = fs.gamestates(actions, nb_prev_actions=n_previous_actions)
    gamestates = fs.play_left_to_right(gamestates, home_team_id)

    parts = [fn(gamestates) for fn in _feature_functions()]
    feats = pd.concat(parts, axis=1)

    # Carry through identifiers; socceraction's feature functions return
    # one row per action but typically drop our identifier columns.
    feats.insert(0, "match_id", actions["match_id"].values)
    feats.insert(1, "action_id", actions["action_id"].values)
    return feats


def build_features(actions: pd.DataFrame, home_team_lookup: dict[int, int],
                   n_previous_actions: int) -> pd.DataFrame:
    """Build features for every match in ``actions``.

    Parameters
    ----------
    actions :
        Full SPADL actions table.
    home_team_lookup :
        Mapping match_id -> home_team_id. SPADL conversion has already
        normalised attacking direction, but socceraction's
        ``play_left_to_right`` still needs the home team to mirror correctly.
    n_previous_actions :
        Window of preceding actions to include as lag features.
    """
    parts: list[pd.DataFrame] = []
    for match_id, group in tqdm(
        actions.groupby("match_id", sort=False),
        total=actions["match_id"].nunique(),
        desc="Baseline features",
    ):
        if match_id not in home_team_lookup:
            log.warning("Skipping match %s: home_team_id unknown", match_id)
            continue
        feats = build_match_features(
            group,
            home_team_id=home_team_lookup[match_id],
            n_previous_actions=n_previous_actions,
        )
        parts.append(feats)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_home_team_lookup(cfg: Config) -> dict[int, int]:
    """Build {match_id: home_team_id} from the StatsBomb match files."""
    from . import data_loading as dl

    inv = pd.read_parquet(cfg.paths.inventory_dir / "inventory_dataset.parquet")
    lookup: dict[int, int] = {}
    # Use the StatsBomb match JSON for the home_team_id (it is not in our
    # cleaned events table).
    cache: dict[tuple[int, int], pd.DataFrame] = {}
    for _, row in inv.iterrows():
        key = (int(row["competition_id"]), int(row["season_id"]))
        if key not in cache:
            cache[key] = dl.load_matches(cfg.paths.statsbomb_root, *key)
        df = cache[key]
        if df.empty:
            continue
        match_row = df[df["match_id"] == int(row["match_id"])]
        if match_row.empty:
            continue
        # The home_team id column name depends on json_normalize output.
        home_id_col = (
            "home_team_home_team_id"
            if "home_team_home_team_id" in match_row.columns
            else "home_team_id"
        )
        if home_id_col not in match_row.columns:
            continue
        lookup[int(row["match_id"])] = int(match_row.iloc[0][home_id_col])
    return lookup


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
def main(config_path: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("features_dir")

    actions_path = cfg.paths.actions_dir / "actions_spadl.parquet"
    if not actions_path.exists():
        raise FileNotFoundError(
            f"SPADL actions not found at {actions_path}. Run `python -m src.spadl_conversion`."
        )
    actions = pd.read_parquet(actions_path)
    log.info("Loaded %d actions across %d matches", len(actions), actions["match_id"].nunique())

    log.info("Building home_team_id lookup from match files ...")
    home_team_lookup = load_home_team_lookup(cfg)
    log.info("Resolved home_team_id for %d matches", len(home_team_lookup))

    feats = build_features(
        actions,
        home_team_lookup=home_team_lookup,
        n_previous_actions=cfg.features_baseline.n_previous_actions,
    )
    out_path = cfg.paths.features_dir / "features_baseline.parquet"
    feats.to_parquet(out_path, index=False)
    log.info("Wrote %d feature rows × %d cols to %s", len(feats), len(feats.columns), out_path)


if __name__ == "__main__":
    main()
