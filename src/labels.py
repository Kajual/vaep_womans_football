"""Stage 5 — Score and concede labels.

For each SPADL action, computes two binary labels:

  * ``score_label``   — the acting team scores within the next k actions
  * ``concede_label`` — the acting team concedes within the next k actions

We use socceraction's reference implementation (:mod:`socceraction.vaep.labels`)
so we inherit the canonical edge-case handling (period boundaries, own goals).

Three horizons are produced per the promotor's plan:
  * k = 10 (main configuration)
  * k = 5, 15 (control configurations)

Run:
    python -m src.labels
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from .config import Config, load_config

log = logging.getLogger(__name__)

try:
    from socceraction.vaep import labels as sl
    _HAVE_SOCCERACTION = True
except ImportError:  # pragma: no cover
    _HAVE_SOCCERACTION = False


def _ensure_socceraction() -> None:
    if not _HAVE_SOCCERACTION:
        raise ImportError(
            "socceraction is not installed. Run `pip install socceraction`."
        )


def compute_labels_for_match(match_actions: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return a DataFrame with score_label and concede_label for one match."""
    _ensure_socceraction()
    s = sl.scores(match_actions, nr_actions=k)
    c = sl.concedes(match_actions, nr_actions=k)
    # socceraction returns single-column DataFrames named "scores" / "concedes".
    out = pd.DataFrame(
        {
            "match_id": match_actions["match_id"].values,
            "action_id": match_actions["action_id"].values,
            "team_id": match_actions["team_id"].values,
            "score_label": s.iloc[:, 0].astype(int).values,
            "concede_label": c.iloc[:, 0].astype(int).values,
            "k": k,
        }
    )
    return out


def compute_labels(actions: pd.DataFrame, k: int) -> pd.DataFrame:
    """Compute labels for all matches in ``actions``, processed match-by-match."""
    parts: list[pd.DataFrame] = []
    for match_id, group in tqdm(
        actions.groupby("match_id", sort=False),
        total=actions["match_id"].nunique(),
        desc=f"Labels k={k}",
    ):
        match = group.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(drop=True)
        parts.append(compute_labels_for_match(match, k=k))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
def main(config_path: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("labels_dir", "tables_dir")

    actions_path = cfg.paths.actions_dir / "actions_spadl.parquet"
    if not actions_path.exists():
        raise FileNotFoundError(
            f"SPADL actions not found at {actions_path}. Run `python -m src.spadl_conversion`."
        )
    actions = pd.read_parquet(actions_path)
    log.info("Loaded %d actions across %d matches", len(actions), actions["match_id"].nunique())

    k_values = [cfg.labels.k_main, *cfg.labels.k_control]
    summary_rows: list[dict] = []

    for k in k_values:
        labels = compute_labels(actions, k=k)
        out_path = cfg.paths.labels_dir / f"vaep_labels_k{k}.parquet"
        labels.to_parquet(out_path, index=False)
        log.info("Wrote %d labels (k=%d) to %s", len(labels), k, out_path)

        summary_rows.append(
            {
                "k": k,
                "n_actions": len(labels),
                "score_rate_pct": float(labels["score_label"].mean() * 100),
                "concede_rate_pct": float(labels["concede_label"].mean() * 100),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = cfg.paths.tables_dir / "labels_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Label horizon summary written to %s\n%s", summary_path, summary.to_string(index=False))


if __name__ == "__main__":
    main()
