"""Low-level readers for the StatsBomb open-data directory layout.

The StatsBomb open-data repo stores its JSON under a ``data/`` subfolder:

    <repo_root>/
        data/
            competitions.json
            matches/<competition_id>/<season_id>.json
            events/<match_id>.json
            lineups/<match_id>.json
            three-sixty/<match_id>.json

``data_root()`` resolves either the cloned-repo root (which contains the
``data/`` subfolder) or the ``data/`` subfolder itself, so the config can
point at either form.

This module does not do any cleaning or feature engineering — that's downstream.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolve the directory that holds competitions.json
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _data_root_cached(statsbomb_root_str: str) -> Path:
    """Cached helper for ``data_root``. Strings hash; Paths don't reliably."""
    root = Path(statsbomb_root_str)
    for candidate in (root, root / "data"):
        if (candidate / "competitions.json").exists():
            return candidate
    raise FileNotFoundError(
        f"competitions.json not found under {root} or {root}/data. "
        "Did you clone https://github.com/statsbomb/open-data into "
        "data/raw/statsbomb_open_data?"
    )


def data_root(statsbomb_root: Path | str) -> Path:
    """Return the directory containing ``competitions.json`` and friends.

    Accepts either the cloned-repo root (which has a ``data/`` subfolder)
    or that subfolder itself. The result is cached.
    """
    return _data_root_cached(str(statsbomb_root))


# ---------------------------------------------------------------------------
# Competitions and matches
# ---------------------------------------------------------------------------

def load_competitions(statsbomb_root: Path) -> pd.DataFrame:
    """Return the competitions index as a DataFrame."""
    path = data_root(statsbomb_root) / "competitions.json"
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def load_matches(statsbomb_root: Path, competition_id: int, season_id: int) -> pd.DataFrame:
    """Return the matches list for one competition+season."""
    path = data_root(statsbomb_root) / "matches" / str(competition_id) / f"{season_id}.json"
    if not path.exists():
        log.warning("Matches file missing: %s", path)
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.json_normalize(rows, sep="_")
    df["competition_id"] = competition_id
    df["season_id"] = season_id
    return df


# ---------------------------------------------------------------------------
# Per-match files
# ---------------------------------------------------------------------------

def _safe_json_load(path: Path, match_id: int, kind: str, strict: bool):
    """Read a JSON file, returning ``None`` on malformed JSON unless ``strict``.

    The StatsBomb open-data dataset occasionally ships malformed files
    (most often in three-sixty); we log the offender and continue.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        if strict:
            raise
        log.warning("Malformed %s JSON for match %s at %s — skipping (%s)",
                    kind, match_id, path.name, exc)
        return None


def load_events(
    statsbomb_root: Path, match_id: int, *, strict: bool = False
) -> list[dict[str, Any]]:
    """Return the raw events list for one match (or ``[]`` on malformed JSON)."""
    path = data_root(statsbomb_root) / "events" / f"{match_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Events file missing for match {match_id}: {path}")
    data = _safe_json_load(path, match_id, kind="events", strict=strict)
    return data if data is not None else []


def load_lineups(
    statsbomb_root: Path, match_id: int, *, strict: bool = False
) -> list[dict[str, Any]]:
    """Return the raw lineups for one match (or ``[]`` if missing/malformed)."""
    path = data_root(statsbomb_root) / "lineups" / f"{match_id}.json"
    if not path.exists():
        log.warning("Lineups file missing for match %s: %s", match_id, path)
        return []
    data = _safe_json_load(path, match_id, kind="lineups", strict=strict)
    return data if data is not None else []


def load_360(
    statsbomb_root: Path, match_id: int, *, strict: bool = False
) -> list[dict[str, Any]] | None:
    """Return the raw 360 freeze-frames for one match.

    Returns ``None`` if the file is absent **or malformed** (the latter is
    logged as a warning). Pass ``strict=True`` to raise instead.
    """
    path = data_root(statsbomb_root) / "three-sixty" / f"{match_id}.json"
    if not path.exists():
        return None
    return _safe_json_load(path, match_id, kind="three-sixty", strict=strict)


# ---------------------------------------------------------------------------
# Existence checks (cheap)
# ---------------------------------------------------------------------------

def has_events(statsbomb_root: Path, match_id: int) -> bool:
    return (data_root(statsbomb_root) / "events" / f"{match_id}.json").exists()


def has_lineups(statsbomb_root: Path, match_id: int) -> bool:
    return (data_root(statsbomb_root) / "lineups" / f"{match_id}.json").exists()


def has_360(statsbomb_root: Path, match_id: int) -> bool:
    return (data_root(statsbomb_root) / "three-sixty" / f"{match_id}.json").exists()


def count_events(statsbomb_root: Path, match_id: int) -> int:
    """Cheap count: load the file and len() it."""
    try:
        return len(load_events(statsbomb_root, match_id))
    except FileNotFoundError:
        return 0


def count_360_frames(statsbomb_root: Path, match_id: int) -> int:
    frames = load_360(statsbomb_root, match_id)
    return len(frames) if frames else 0


# ---------------------------------------------------------------------------
# Convenience: iterate all matches across competitions+seasons
# ---------------------------------------------------------------------------

def iter_all_matches(statsbomb_root: Path) -> pd.DataFrame:
    """Return a single DataFrame of every match referenced in competitions.json.

    The dataframe has one row per (competition_id, season_id, match_id).
    """
    comps = load_competitions(statsbomb_root)
    parts: list[pd.DataFrame] = []
    for _, comp in comps.iterrows():
        m = load_matches(statsbomb_root, comp["competition_id"], comp["season_id"])
        if m.empty:
            continue
        # Carry through competition metadata for convenience.
        m["competition_name"] = comp["competition_name"]
        m["season_name"] = comp["season_name"]
        m["competition_gender"] = comp.get("competition_gender", "unknown")
        parts.append(m)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
