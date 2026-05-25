"""Stage 15 — Player-level VAEP aggregation and rankings.

Takes the per-action VAEP artifact produced by ``vaep_values.py`` and rolls it
up to the player level: total offensive / defensive / overall VAEP, a per-90
normalisation, and a ranking of players under each model (A / B / C).

This module produces the numbers behind thesis Chapter 6 §6.3 — *does adding
phase and spatial context change who the model says the best players are?* It
answers that by reporting the rank correlation between models and by listing
the players whose rank moves most between Model A (event-only) and Model C
(event + phase + space).

Inputs
------
``data/processed/model_outputs/all_models_vaep_values.parquet``
    Long-form per-action VAEP, one block of rows per ``model_id``.

Outputs (written to ``outputs/tables/``)
-------
``player_rankings.parquet``      full per-(model, player) table with ranks
``player_rankings_summary.csv``  top-N players per model, human-readable
``ranking_stability.csv``        Spearman / Kendall rank correlation per pair
``ranking_movers.csv``           players with the largest A->C rank change

Run
---
    python -m src.aggregation
    python -m src.aggregation --top-n 30 --min-minutes 270
    python -m src.aggregation --matched    # restrict to actions all models share
"""
from __future__ import annotations

import logging

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import data_loading as dl
from .config import Config, load_config

log = logging.getLogger(__name__)


# socceraction is only needed to look up player names and minutes played.
# Everything else works without it (rankings then fall back to raw totals).
try:
    from socceraction.data.statsbomb import StatsBombLoader
    _HAVE_SOCCERACTION = True
except Exception:  # pragma: no cover
    _HAVE_SOCCERACTION = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mode(series: pd.Series):
    """Return the most common non-null value, or ``None`` if the series is empty."""
    clean = series.dropna()
    if clean.empty:
        return None
    m = clean.mode()
    return m.iat[0] if not m.empty else clean.iat[0]


# ---------------------------------------------------------------------------
# Player directory: names, teams, minutes played
# ---------------------------------------------------------------------------

def build_player_directory(cfg: Config, match_ids: list[int]) -> pd.DataFrame:
    """Return a per-player lookup table for the given matches.

    Columns: ``player_id, player_name, team_name, minutes_played, n_matches``.

    Uses socceraction's ``StatsBombLoader``, which derives ``minutes_played``
    from the lineup file plus substitution / red-card events. If socceraction
    is unavailable or a match fails to load, that match is skipped with a
    warning. A fully empty result is handled by the caller, which then ranks
    on raw VAEP totals instead of a per-90 rate.
    """
    empty = pd.DataFrame(
        columns=["player_id", "player_name", "team_name", "minutes_played", "n_matches"]
    )
    if not _HAVE_SOCCERACTION:
        log.warning(
            "socceraction unavailable — player names and minutes will be blank; "
            "rankings will use raw VAEP totals."
        )
        return empty

    loader = StatsBombLoader(
        root=str(dl.data_root(cfg.paths.statsbomb_root)), getter="local"
    )

    player_parts: list[pd.DataFrame] = []
    team_map: dict[int, str] = {}
    for mid in tqdm(sorted(int(m) for m in match_ids), desc="players"):
        try:
            players = loader.players(game_id=mid)
        except Exception as exc:  # pragma: no cover - per-match resilience
            log.warning("loader.players failed for match %s: %s", mid, exc)
            continue
        player_parts.append(players)
        try:
            teams = loader.teams(game_id=mid)
            team_map.update(dict(zip(teams["team_id"], teams["team_name"])))
        except Exception as exc:  # pragma: no cover
            log.warning("loader.teams failed for match %s: %s", mid, exc)

    if not player_parts:
        log.warning("No player data could be loaded — falling back to raw totals.")
        return empty

    allp = pd.concat(player_parts, ignore_index=True)
    if "team_name" not in allp.columns:
        allp["team_name"] = allp["team_id"].map(team_map)
    else:
        allp["team_name"] = allp["team_name"].fillna(allp["team_id"].map(team_map))

    directory = (
        allp.groupby("player_id")
        .agg(
            player_name=("player_name", _safe_mode),
            team_name=("team_name", _safe_mode),
            minutes_played=("minutes_played", "sum"),
            n_matches=("game_id", "nunique"),
        )
        .reset_index()
    )
    directory["player_id"] = directory["player_id"].astype("int64")
    return directory


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_player_vaep(
    vaep: pd.DataFrame,
    directory: pd.DataFrame,
    min_minutes: float,
) -> tuple[pd.DataFrame, tuple[str, str, str]]:
    """Roll per-action VAEP up to per-(model, player) totals, rates and ranks.

    Returns the rankings table and the triple of metric columns the ranks
    were computed on (per-90 rates when minutes are available, otherwise
    raw sums).
    """
    df = vaep.dropna(subset=["player_id"]).copy()
    df["player_id"] = df["player_id"].astype("int64")

    grp = (
        df.groupby(["model_id", "player_id"], as_index=False)
        .agg(
            n_actions=("vaep_value", "size"),
            sum_offensive=("offensive_value", "sum"),
            sum_defensive=("defensive_value", "sum"),
            sum_vaep=("vaep_value", "sum"),
        )
    )

    if not directory.empty:
        grp = grp.merge(directory, on="player_id", how="left")
    else:
        grp["player_name"] = None
        grp["team_name"] = None
        grp["minutes_played"] = np.nan
        grp["n_matches"] = np.nan

    # Always give every player a printable name.
    grp["player_name"] = grp["player_name"].fillna(
        "player_" + grp["player_id"].astype(str)
    )

    # Per-90 normalisation (only where minutes are known and positive).
    have_minutes = grp["minutes_played"].notna() & (grp["minutes_played"] > 0)
    safe_minutes = grp["minutes_played"].where(have_minutes)
    grp["vaep_per90"] = grp["sum_vaep"] / safe_minutes * 90.0
    grp["off_per90"] = grp["sum_offensive"] / safe_minutes * 90.0
    grp["def_per90"] = grp["sum_defensive"] / safe_minutes * 90.0

    # Choose the metric the ranking is built on.
    if have_minutes.any():
        grp["qualified"] = grp["minutes_played"].fillna(0.0) >= min_minutes
        metrics = ("vaep_per90", "off_per90", "def_per90")
        log.info(
            "Ranking on per-90 rates; %d / %d players clear the %.0f-minute bar.",
            int(grp["qualified"].sum()), len(grp), min_minutes,
        )
    else:
        log.warning("No minutes data available — ranking on raw VAEP totals.")
        grp["qualified"] = grp["n_actions"] >= 1
        metrics = ("sum_vaep", "sum_offensive", "sum_defensive")

    # Rank within each model, over qualified players only.
    m_total, m_off, m_def = metrics
    q = grp[grp["qualified"]].copy()
    q["rank_total"] = q.groupby("model_id")[m_total].rank(ascending=False, method="min")
    q["rank_off"] = q.groupby("model_id")[m_off].rank(ascending=False, method="min")
    q["rank_def"] = q.groupby("model_id")[m_def].rank(ascending=False, method="min")
    grp = grp.merge(
        q[["model_id", "player_id", "rank_total", "rank_off", "rank_def"]],
        on=["model_id", "player_id"], how="left",
    )

    grp = grp.sort_values(["model_id", "rank_total"], na_position="last").reset_index(drop=True)
    return grp, metrics


# ---------------------------------------------------------------------------
# Ranking stability and movers
# ---------------------------------------------------------------------------

def compute_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    """Spearman and Kendall rank correlation between every pair of models."""
    models = sorted(rankings["model_id"].unique())
    rows: list[dict] = []
    for metric, label in [
        ("rank_total", "total"),
        ("rank_off", "offensive"),
        ("rank_def", "defensive"),
    ]:
        wide = rankings.pivot_table(index="player_id", columns="model_id", values=metric)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                if a not in wide.columns or b not in wide.columns:
                    continue
                pair = wide[[a, b]].dropna()
                if len(pair) < 3:
                    continue
                rows.append({
                    "metric": label,
                    "model_pair": f"{a}_vs_{b}",
                    "n_players": len(pair),
                    "spearman": round(float(pair[a].corr(pair[b], method="spearman")), 4),
                    "kendall": round(float(pair[a].corr(pair[b], method="kendall")), 4),
                })
    return pd.DataFrame(rows)


def compute_movers(
    rankings: pd.DataFrame, base_model: str, adv_model: str
) -> pd.DataFrame:
    """Players whose rank changes most going from ``base_model`` to ``adv_model``.

    ``delta_*`` is ``rank_base - rank_adv``; a positive value means the player
    rose (a smaller rank number) under the more context-rich model.
    """
    base_cols = ["player_id", "player_name", "team_name",
                 "rank_total", "rank_off", "rank_def", "vaep_per90"]
    adv_cols = ["player_id", "rank_total", "rank_off", "rank_def", "vaep_per90"]
    base = rankings.loc[rankings["model_id"] == base_model, base_cols]
    adv = rankings.loc[rankings["model_id"] == adv_model, adv_cols]
    m = base.merge(adv, on="player_id", suffixes=(f"_{base_model}", f"_{adv_model}"))

    for kind in ("total", "off", "def"):
        m[f"delta_{kind}"] = (
            m[f"rank_{kind}_{base_model}"] - m[f"rank_{kind}_{adv_model}"]
        )

    m = m.dropna(subset=["delta_total"])
    m = m.reindex(m["delta_total"].abs().sort_values(ascending=False).index)
    return m.reset_index(drop=True)


def build_summary(rankings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Top-N players per model, with the most useful columns, rounded."""
    cols = [
        "model_id", "rank_total", "player_name", "team_name", "minutes_played",
        "n_actions", "sum_offensive", "sum_defensive", "sum_vaep",
        "off_per90", "def_per90", "vaep_per90", "rank_off", "rank_def",
    ]
    out = (
        rankings[rankings["rank_total"].notna()]
        .sort_values(["model_id", "rank_total"])
        .groupby("model_id", group_keys=False)
        .head(top_n)
    )
    cols = [c for c in cols if c in out.columns]
    out = out[cols].copy()
    if "minutes_played" in out.columns:
        out["minutes_played"] = out["minutes_played"].round(0)
    return out.round(4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
@click.option("--min-minutes", default=270.0, type=float,
              help="Minimum minutes played for a player to be ranked (default 270 = 3 full matches).")
@click.option("--top-n", default=25, type=int, help="Rows per model in the summary / movers CSVs.")
@click.option("--matched/--no-matched", default=False,
              help="Restrict to (match_id, action_id) pairs present for ALL models "
                   "before aggregating — a fair head-to-head ranking comparison.")
def main(config_path: str | None, min_minutes: float, top_n: int, matched: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("tables_dir")

    vaep_path = cfg.paths.model_outputs_dir / "all_models_vaep_values.parquet"
    if not vaep_path.exists():
        raise FileNotFoundError(
            f"VAEP values not found at {vaep_path}. Run `python -m src.vaep_values` first."
        )
    vaep = pd.read_parquet(vaep_path)
    log.info("Loaded %d VAEP rows across %d models: %s",
             len(vaep), vaep["model_id"].nunique(), sorted(vaep["model_id"].unique()))

    if matched:
        key = ["match_id", "action_id"]
        n_models = vaep["model_id"].nunique()
        key_counts = vaep.groupby(key)["model_id"].nunique().reset_index(name="_nm")
        common = key_counts.loc[key_counts["_nm"] == n_models, key]
        before = len(vaep)
        vaep = vaep.merge(common, on=key, how="inner")
        log.info("Matched mode: %d -> %d rows (%d actions shared by all models).",
                 before, len(vaep), len(common))

    match_ids = vaep["match_id"].dropna().unique().tolist()
    directory = build_player_directory(cfg, match_ids)
    if not directory.empty:
        log.info("Player directory: %d players across %d matches.",
                 len(directory), len(match_ids))

    rankings, metrics = aggregate_player_vaep(vaep, directory, min_minutes)

    # --- write full table -------------------------------------------------
    rankings_path = cfg.paths.tables_dir / "player_rankings.parquet"
    rankings.to_parquet(rankings_path, index=False)
    log.info("Wrote %d ranking rows to %s", len(rankings), rankings_path)

    # --- summary ----------------------------------------------------------
    summary = build_summary(rankings, top_n)
    summary_path = cfg.paths.tables_dir / "player_rankings_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Top-%d-per-model summary saved to %s", top_n, summary_path)

    # --- stability --------------------------------------------------------
    stability = compute_stability(rankings)
    stability_path = cfg.paths.tables_dir / "ranking_stability.csv"
    stability.to_csv(stability_path, index=False)
    log.info("Ranking stability saved to %s\n%s",
             stability_path, stability.to_string(index=False))

    # --- movers (first model vs last model, i.e. A vs C) ------------------
    models = sorted(rankings["model_id"].unique())
    if len(models) >= 2:
        movers = compute_movers(rankings, models[0], models[-1])
        movers_path = cfg.paths.tables_dir / "ranking_movers.csv"
        movers.head(top_n).to_csv(movers_path, index=False)
        log.info("Biggest %s->%s movers saved to %s", models[0], models[-1], movers_path)

    # --- console preview --------------------------------------------------
    for mid in models:
        top5 = summary[summary["model_id"] == mid].head(5)
        log.info("Top 5 — %s:\n%s", mid, top5.to_string(index=False))


if __name__ == "__main__":
    main()
