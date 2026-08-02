"""Post-hoc analyses for the MLSA 2026 revision (paper 306).

Everything here is computed from artefacts the pipeline has already written:
the per-model prediction parquets, the VAEP value table, the SPADL action
stream and the spatial features. No model is retrained, so this module can be
run on its own after the main pipeline has completed.

It produces four families of results, each answering a specific reviewer point:

  1. Matched metrics (R1-C5, R2-C8b)
     Every model and every transfer variant evaluated on the *identical*
     360 subset defined by Model C, including correctly-scoped ECE. The
     submitted paper mixed the 60,370-action and 51,172-action universes.

  2. Paired match-level bootstrap (R1-C5, R2-C2)
     Matches, not actions, are the resampling unit. All models are recomputed
     on the same resample so the intervals describe *differences*.

  3. Stratified revaluation analysis (R1-C3/Q3, R2-C5)
     Decomposition of the per-action change from P-VAEP to PS-VAEP across
     nearest-opponent distance, defensive density, action type, pitch zone
     and outcome, with within-stratum discrimination.

  4. Ranking bootstrap (R2-C2, R2-C8c)
     Player aggregates and Spearman correlations recomputed under match
     resampling, so the positional conclusions carry uncertainty.

Run:
    python -m src.rebuttal_analysis                      # everything
    python -m src.rebuttal_analysis --only matched
    python -m src.rebuttal_analysis --n-boot 200         # quick pass
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

log = logging.getLogger(__name__)

# Model identifiers as used by the pipeline, and their paper-facing names.
BASE_MODELS = ["model_A", "model_B", "model_C"]
PAPER_NAME = {"model_A": "E-VAEP", "model_B": "P-VAEP", "model_C": "PS-VAEP"}
TRANSFER_VARIANTS = ["men_only", "women_only", "fine_tuned"]
HEADS = ["score", "concede"]

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "processed" / "model_outputs"
FEAT_DIR = ROOT / "data" / "processed" / "features"
ACT_DIR = ROOT / "data" / "processed" / "actions"
OUT_DIR = ROOT / "outputs" / "tables" / "rebuttal"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE over equal-width bins, matching the submitted paper's definition."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def adaptive_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE over equal-frequency bins.

    Reviewer 2 (C4) notes that equal-width bins are close to uninformative at a
    positive rate of 0.34%, because almost every action lands in the lowest bin.
    Equal-frequency binning spreads the mass across bins instead.
    """
    if len(p) < n_bins:
        return float("nan")
    quantiles = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    idx = np.clip(np.digitize(p, quantiles), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def head_metrics(y: np.ndarray, p: np.ndarray, prefix: str) -> dict[str, float]:
    """Discrimination, probability quality and calibration for one head."""
    if y.sum() == 0 or y.sum() == len(y):
        # Degenerate stratum or bootstrap replicate: discrimination undefined.
        return {
            f"{prefix}_auc": float("nan"),
            f"{prefix}_ap": float("nan"),
            f"{prefix}_brier": float(np.mean((p - y) ** 2)),
            f"{prefix}_logloss": float("nan"),
            f"{prefix}_ece": float("nan"),
            f"{prefix}_ace": float("nan"),
            f"{prefix}_pos_rate": float(y.mean()),
        }
    return {
        f"{prefix}_auc": float(roc_auc_score(y, p)),
        f"{prefix}_ap": float(average_precision_score(y, p)),
        f"{prefix}_brier": float(np.mean((p - y) ** 2)),
        f"{prefix}_logloss": float(log_loss(y, p, labels=[0, 1])),
        f"{prefix}_ece": expected_calibration_error(y, p),
        f"{prefix}_ace": adaptive_calibration_error(y, p),
        f"{prefix}_pos_rate": float(y.mean()),
    }


def all_metrics(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {"n": len(df)}
    for head in HEADS:
        out.update(head_metrics(df[f"{head}_label"].values, df[f"p_{head}"].values, head))
    return out


def top_k_precision_recall(y: np.ndarray, p: np.ndarray, k_frac: float) -> tuple[float, float]:
    """Precision and recall in the top ``k_frac`` of the predicted ranking.

    Reviewer 2 (C4) asks where PS-VAEP's average-precision gain comes from,
    given that its ROC-AUC and log loss are worse. If the gain is confined to
    the sharp end of the ranking, it will show up here and not in ROC-AUC.
    """
    n_top = max(1, int(round(k_frac * len(p))))
    order = np.argsort(-p)[:n_top]
    hits = y[order].sum()
    precision = hits / n_top
    recall = hits / y.sum() if y.sum() else float("nan")
    return float(precision), float(recall)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(model_id: str, variant: str | None = None) -> pd.DataFrame | None:
    name = model_id if variant is None else f"{model_id}_{variant}"
    path = PRED_DIR / f"{name}_predictions.parquet"
    if not path.exists():
        log.warning("Missing predictions: %s", path)
        return None
    return pd.read_parquet(path)


def subset_keys() -> pd.DataFrame:
    """The (match_id, action_id) keys of the 360 subset, defined by Model C."""
    c = load_predictions("model_C")
    if c is None:
        raise FileNotFoundError("model_C predictions are required to define the 360 subset.")
    return c[["match_id", "action_id"]].drop_duplicates()


def restrict(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    return df.merge(keys, on=["match_id", "action_id"], how="inner")


# ---------------------------------------------------------------------------
# 1. Matched metrics  (R1-C5, R2-C8b)
# ---------------------------------------------------------------------------

def run_matched_metrics(keys: pd.DataFrame) -> pd.DataFrame:
    """Every model and variant on the identical 360 subset.

    The submitted Table 2 reported pooled E-VAEP and P-VAEP on the full
    60,370-action set while PS-VAEP was necessarily restricted to 51,172, and
    the submitted Table 3 took its ECE row from a calibration table computed on
    the full set. Both are recomputed here on one action universe.
    """
    rows: list[dict] = []
    for model_id in BASE_MODELS:
        for variant in [None, *TRANSFER_VARIANTS]:
            preds = load_predictions(model_id, variant)
            if preds is None:
                continue
            matched = restrict(preds, keys)
            rows.append({
                "model_id": model_id,
                "paper_name": PAPER_NAME[model_id],
                "strategy": "pooled" if variant is None else variant,
                **all_metrics(matched),
            })
    df = pd.DataFrame(rows)
    log.info("Matched metrics: %d model x strategy cells on %d actions", len(df), len(keys))
    return df


def run_top_k_analysis(keys: pd.DataFrame) -> pd.DataFrame:
    """Precision/recall at the sharp end of the conceding ranking (R2-C4)."""
    rows: list[dict] = []
    for model_id in BASE_MODELS:
        preds = load_predictions(model_id)
        if preds is None:
            continue
        matched = restrict(preds, keys)
        y = matched["concede_label"].values
        p = matched["p_concede"].values
        row = {"model_id": model_id, "paper_name": PAPER_NAME[model_id]}
        for k_frac in (0.001, 0.01, 0.05, 0.10):
            prec, rec = top_k_precision_recall(y, p, k_frac)
            row[f"precision_top{k_frac:.3f}"] = prec
            row[f"recall_top{k_frac:.3f}"] = rec
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Paired match-level bootstrap  (R1-C5, R2-C2)
# ---------------------------------------------------------------------------

BOOT_METRICS = [
    f"{h}_{m}" for h in HEADS
    for m in ("auc", "ap", "brier", "logloss", "ece", "ace")
]
BOOT_PAIRS = [("model_B", "model_A"), ("model_C", "model_B"), ("model_C", "model_A")]
DRAWS_DIR = OUT_DIR / "boot_draws"


def draw_bootstrap_chunk(
    keys: pd.DataFrame,
    n_boot: int,
    seed: int,
    variant: str | None = None,
    chunk_id: int = 0,
) -> Path:
    """Draw one chunk of paired bootstrap replicates and persist the raw draws.

    Matches are resampled with replacement and every model is re-evaluated on
    the same resample, so the resulting intervals describe the *difference*
    between two models rather than each model's own sampling variability.
    Paired intervals are the relevant ones here because the paper's claims are
    all comparative and the models share an evaluation action set.

    Chunks are written separately and combined by :func:`combine_bootstrap`, so
    a long run can be split across several invocations without losing work.
    Each chunk uses ``seed + chunk_id`` so the replicates remain independent.
    """
    strategy = "pooled" if variant is None else variant
    frames: dict[str, pd.DataFrame] = {}
    for model_id in BASE_MODELS:
        preds = load_predictions(model_id, variant)
        if preds is None:
            continue
        frames[model_id] = restrict(preds, keys).set_index("match_id", drop=False)

    if len(frames) < 2:
        raise RuntimeError(f"Need at least two models for strategy {strategy}")

    matches = np.sort(frames[BASE_MODELS[0]]["match_id"].unique())
    rng = np.random.default_rng(seed + chunk_id)

    records: list[dict] = []
    for b in range(n_boot):
        sampled = rng.choice(matches, size=len(matches), replace=True)
        rec: dict[str, float] = {"replicate": b}
        for model_id, frame in frames.items():
            metrics = all_metrics(frame.loc[sampled])
            for name in BOOT_METRICS:
                rec[f"{model_id}__{name}"] = metrics[name]
        records.append(rec)

    DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    path = DRAWS_DIR / f"draws_{strategy}_chunk{chunk_id:03d}.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    log.info("Wrote %d replicates for '%s' to %s", n_boot, strategy, path.name)
    return path


def combine_bootstrap(keys: pd.DataFrame) -> pd.DataFrame:
    """Turn all persisted draw chunks into percentile intervals for differences."""
    if not DRAWS_DIR.exists():
        log.warning("No bootstrap draws found in %s", DRAWS_DIR)
        return pd.DataFrame()

    rows: list[dict] = []
    for strategy in ["pooled", *TRANSFER_VARIANTS]:
        chunks = sorted(DRAWS_DIR.glob(f"draws_{strategy}_chunk*.parquet"))
        if not chunks:
            continue
        draws = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)

        # Point estimates on the observed (unresampled) sample.
        variant = None if strategy == "pooled" else strategy
        point: dict[str, dict[str, float]] = {}
        for model_id in BASE_MODELS:
            preds = load_predictions(model_id, variant)
            if preds is not None:
                point[model_id] = all_metrics(restrict(preds, keys))

        for a, b_ in BOOT_PAIRS:
            for metric in BOOT_METRICS:
                col_a, col_b = f"{a}__{metric}", f"{b_}__{metric}"
                if col_a not in draws or col_b not in draws:
                    continue
                diffs = (draws[col_a] - draws[col_b]).replace([np.inf, -np.inf], np.nan).dropna()
                if diffs.empty:
                    continue
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                rows.append({
                    "strategy": strategy,
                    "comparison": f"{PAPER_NAME[a]} - {PAPER_NAME[b_]}",
                    "metric": metric,
                    "point_diff": point[a][metric] - point[b_][metric],
                    "ci_lo": lo,
                    "ci_hi": hi,
                    # The interval excluding zero means the sign of the
                    # difference is stable under match resampling.
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n_boot": len(diffs),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Stratified revaluation analysis  (R1-C3/Q3, R2-C5)
# ---------------------------------------------------------------------------

def build_action_panel(keys: pd.DataFrame) -> pd.DataFrame:
    """One row per evaluation action, carrying VAEP values and spatial context."""
    values = pd.read_parquet(PRED_DIR / "all_models_vaep_values.parquet")
    values = values[values["model_id"].isin(BASE_MODELS)]

    wide = values.pivot_table(
        index=["match_id", "action_id"],
        columns="model_id",
        values=["vaep_value", "offensive_value", "defensive_value"],
    )
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    meta = (
        values[values["model_id"] == "model_A"]
        [["match_id", "action_id", "type_name", "result_name", "player_id", "team_id",
          "concede_label", "score_label"]]
    )
    panel = wide.merge(meta, on=["match_id", "action_id"], how="left")

    space = pd.read_parquet(FEAT_DIR / "space_features.parquet")
    panel = panel.merge(space, on=["match_id", "action_id"], how="inner")

    actions = pd.read_parquet(ACT_DIR / "actions_spadl.parquet")
    panel = panel.merge(
        actions[["match_id", "action_id", "start_x", "start_y"]],
        on=["match_id", "action_id"], how="left",
    )

    panel = restrict(panel, keys)

    # Prediction columns for within-stratum discrimination.
    for model_id in BASE_MODELS:
        preds = load_predictions(model_id)
        if preds is None:
            continue
        panel = panel.merge(
            preds[["match_id", "action_id", "p_score", "p_concede"]]
            .rename(columns={"p_score": f"p_score__{model_id}",
                             "p_concede": f"p_concede__{model_id}"}),
            on=["match_id", "action_id"], how="left",
        )

    # The quantity the revaluation claim is about: what adding space does on
    # top of phase.
    panel["delta_vaep"] = panel["vaep_value__model_C"] - panel["vaep_value__model_B"]
    panel["delta_off"] = panel["offensive_value__model_C"] - panel["offensive_value__model_B"]
    panel["delta_def"] = panel["defensive_value__model_C"] - panel["defensive_value__model_B"]

    log.info("Action panel: %d rows", len(panel))
    return panel


def _pitch_zone(row) -> str:
    x, y = row["start_x"], row["start_y"]
    if x >= 88.5 and 13.84 <= y <= 54.16:
        return "penalty_area"
    if x >= 70.0:
        return "final_third"
    if x <= 40.0:
        return "own_third"
    return "middle_third"


def _lane(row) -> str:
    y = row["start_y"]
    if y < 13.84 or y > 54.16:
        return "wide"
    if 24.84 <= y <= 43.16:
        return "central"
    return "half_space"


def add_strata(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["stratum_opp_dist"] = pd.cut(
        panel["nearest_opponent_distance"],
        bins=[-np.inf, 2.0, 5.0, 10.0, np.inf],
        labels=["<2m", "2-5m", "5-10m", ">10m"],
    )
    panel["stratum_density_5m"] = pd.cut(
        panel["defensive_density_around_ball_5m"],
        bins=[-np.inf, 0.5, 1.5, 2.5, np.inf],
        labels=["0", "1", "2", "3+"],
    )
    panel["stratum_type"] = panel["type_name"]
    panel["stratum_zone"] = panel.apply(_pitch_zone, axis=1)
    panel["stratum_lane"] = panel.apply(_lane, axis=1)
    panel["stratum_outcome"] = panel["result_name"]
    # Broadcast coverage, for the visibility check requested by R2-C6.
    panel["stratum_visibility"] = pd.qcut(
        panel["n_opponents_visible"] + panel["n_teammates_visible"],
        q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop",
    )
    return panel


def run_stratified(panel: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Mean change in value per stratum, with within-stratum discrimination.

    This replaces the single Maanum case study as evidence for the claim that
    the spatial layer acts on pressure and congestion: if the mechanism is real,
    the mean change should increase monotonically with defensive density and
    decrease with distance to the nearest opponent.
    """
    rng = np.random.default_rng(seed)
    strata_cols = [c for c in panel.columns if c.startswith("stratum_")]
    matches = np.sort(panel["match_id"].unique())

    rows: list[dict] = []
    for col in strata_cols:
        for level, group in panel.groupby(col, observed=True):
            if len(group) < 50:
                continue
            row = {
                "stratification": col.replace("stratum_", ""),
                "level": str(level),
                "n_actions": len(group),
                "share_pct": 100.0 * len(group) / len(panel),
                "mean_delta_vaep": group["delta_vaep"].mean(),
                "mean_delta_off": group["delta_off"].mean(),
                "mean_delta_def": group["delta_def"].mean(),
            }

            # Match-level bootstrap of the mean change within this stratum.
            by_match = group.groupby("match_id")["delta_vaep"].agg(["sum", "count"])
            present = by_match.index.values
            if len(present) > 1:
                draws = []
                for _ in range(n_boot):
                    sampled = rng.choice(present, size=len(matches), replace=True)
                    sel = by_match.reindex(sampled)
                    total, count = sel["sum"].sum(), sel["count"].sum()
                    if count > 0:
                        draws.append(total / count)
                if draws:
                    row["ci_lo"], row["ci_hi"] = np.percentile(draws, [2.5, 97.5])

            # Discrimination inside the stratum, where the claim is made.
            for model_id in ("model_B", "model_C"):
                y = group["concede_label"].values
                p = group[f"p_concede__{model_id}"].values
                if 0 < y.sum() < len(y):
                    row[f"concede_auc_{model_id}"] = roc_auc_score(y, p)
                    row[f"concede_ap_{model_id}"] = average_precision_score(y, p)
            rows.append(row)

    return pd.DataFrame(rows)


def run_visibility_check(panel: pd.DataFrame) -> pd.DataFrame:
    """Association between broadcast coverage and the revaluation (R2-C6).

    Coverage is denser in the penalty area than in midfield, which is the same
    contrast as the reported positional redistribution. If the correlation
    between visible-player count and the change in value is strong, part of the
    redistribution may reflect camera framing rather than defensive structure.
    """
    panel = panel.copy()
    panel["n_visible"] = panel["n_opponents_visible"] + panel["n_teammates_visible"]

    rows: list[dict] = []
    for zone, group in panel.groupby("stratum_zone", observed=True):
        rows.append({
            "zone": zone,
            "n_actions": len(group),
            "mean_n_visible": group["n_visible"].mean(),
            "mean_n_opponents_visible": group["n_opponents_visible"].mean(),
            "mean_delta_vaep": group["delta_vaep"].mean(),
            "corr_visible_delta": group["n_visible"].corr(group["delta_vaep"], method="spearman"),
        })
    overall = {
        "zone": "ALL",
        "n_actions": len(panel),
        "mean_n_visible": panel["n_visible"].mean(),
        "mean_n_opponents_visible": panel["n_opponents_visible"].mean(),
        "mean_delta_vaep": panel["delta_vaep"].mean(),
        "corr_visible_delta": panel["n_visible"].corr(panel["delta_vaep"], method="spearman"),
    }
    return pd.DataFrame([overall, *rows])


# ---------------------------------------------------------------------------
# 4. Ranking bootstrap  (R2-C2, R2-C8c)
# ---------------------------------------------------------------------------

def load_player_minutes(match_ids: list[int]) -> pd.DataFrame:
    """Per-match minutes played, via socceraction's StatsBomb loader.

    Mirrors what ``aggregation.py`` does, but keeps the per-match granularity
    that match-level resampling requires.
    """
    try:
        from socceraction.data.statsbomb import StatsBombLoader
    except ImportError:
        log.warning("socceraction unavailable — skipping the ranking bootstrap.")
        return pd.DataFrame()

    root = ROOT / "data" / "raw" / "statsbomb_open_data" / "data"
    loader = StatsBombLoader(root=str(root), getter="local")
    frames = []
    for match_id in match_ids:
        try:
            pg = loader.players(game_id=match_id)
        except Exception as exc:  # noqa: BLE001 - one bad match shouldn't abort
            log.warning("Could not load players for match %s: %s", match_id, exc)
            continue
        frames.append(pg[["game_id", "player_id", "player_name", "team_id", "minutes_played"]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).rename(columns={"game_id": "match_id"})
    log.info("Loaded minutes for %d player-match rows", len(out))
    return out


def run_ranking_bootstrap(
    panel: pd.DataFrame,
    minutes: pd.DataFrame,
    n_boot: int,
    seed: int,
    min_minutes: float = 270.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlations and rank changes under match resampling.

    The submitted paper treated the 117 qualifying players as independent
    observations. Resampling matches instead propagates the fact that a
    player's value comes from a handful of matches, which is where most of the
    uncertainty in the positional conclusions actually lives.
    """
    if minutes.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Per-player, per-match VAEP totals for each model.
    value_cols = {m: f"vaep_value__{m}" for m in BASE_MODELS}
    per_match = (
        panel.groupby(["match_id", "player_id"])[list(value_cols.values())]
        .sum()
        .reset_index()
    )
    per_match = per_match.merge(
        minutes[["match_id", "player_id", "player_name", "minutes_played"]],
        on=["match_id", "player_id"], how="inner",
    )

    matches = np.sort(per_match["match_id"].unique())
    rng = np.random.default_rng(seed)

    def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        grp = frame.groupby("player_id").agg(
            minutes_played=("minutes_played", "sum"),
            **{m: (value_cols[m], "sum") for m in BASE_MODELS},
        )
        grp = grp[grp["minutes_played"] >= min_minutes]
        for m in BASE_MODELS:
            grp[f"per90_{m}"] = grp[m] / grp["minutes_played"] * 90.0
        return grp

    observed = aggregate(per_match)
    pairs = [("model_A", "model_B"), ("model_A", "model_C"), ("model_B", "model_C")]

    corr_draws: dict[str, list[float]] = {f"{a}_vs_{b}": [] for a, b in pairs}
    rank_draws: dict[int, list[float]] = {}

    # Index once by match so each replicate is a lookup rather than 31 scans.
    indexed = per_match.set_index("match_id", drop=False).sort_index()

    for _ in range(n_boot):
        sampled = rng.choice(matches, size=len(matches), replace=True)
        # .loc with repeats concatenates the sampled matches, repeats included.
        agg = aggregate(indexed.loc[sampled])
        if len(agg) < 10:
            continue
        for a, b in pairs:
            rho = spearmanr(agg[f"per90_{a}"], agg[f"per90_{b}"]).statistic
            corr_draws[f"{a}_vs_{b}"].append(rho)
        ra = agg[f"per90_model_A"].rank(ascending=False)
        rc = agg[f"per90_model_C"].rank(ascending=False)
        delta = ra - rc  # positive: player rises under PS-VAEP
        for pid, d in delta.items():
            rank_draws.setdefault(pid, []).append(float(d))

    corr_rows = []
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        draws = np.array(corr_draws[key], dtype=float)
        if len(draws) == 0:
            continue
        lo, hi = np.percentile(draws, [2.5, 97.5])
        corr_rows.append({
            "model_pair": f"{PAPER_NAME[a]} vs {PAPER_NAME[b]}",
            "spearman_observed": spearmanr(observed[f"per90_{a}"], observed[f"per90_{b}"]).statistic,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_boot": len(draws),
        })

    obs_rank_a = observed["per90_model_A"].rank(ascending=False)
    obs_rank_c = observed["per90_model_C"].rank(ascending=False)
    name_lookup = per_match.drop_duplicates("player_id").set_index("player_id")["player_name"]

    rank_rows = []
    for pid, draws_list in rank_draws.items():
        draws = np.array(draws_list, dtype=float)
        if len(draws) < max(10, n_boot // 10) or pid not in obs_rank_a.index:
            continue
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rank_rows.append({
            "player_id": pid,
            "player_name": name_lookup.get(pid, ""),
            "rank_E_VAEP": obs_rank_a.get(pid),
            "rank_PS_VAEP": obs_rank_c.get(pid),
            "delta_rank_observed": obs_rank_a.get(pid) - obs_rank_c.get(pid),
            "ci_lo": lo,
            "ci_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": len(draws),
        })

    rank_df = pd.DataFrame(rank_rows)
    if not rank_df.empty:
        rank_df = rank_df.sort_values("delta_rank_observed", ascending=False)
    return pd.DataFrame(corr_rows), rank_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _write(df: pd.DataFrame, name: str) -> None:
    if df is None or df.empty:
        log.warning("Nothing to write for %s", name)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap replicates (default 1000).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default="all",
                        choices=["all", "matched", "bootstrap", "combine",
                                 "stratified", "ranking"])
    parser.add_argument("--strategy", default=None,
                        help="Restrict --only bootstrap to one strategy "
                             "(pooled, men_only, women_only, fine_tuned).")
    parser.add_argument("--chunk-id", type=int, default=0,
                        help="Chunk index for resumable bootstrap runs; the "
                             "effective seed is --seed + --chunk-id.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    keys = subset_keys()
    log.info("360 subset: %d actions", len(keys))
    want = args.only

    if want in ("all", "matched"):
        _write(run_matched_metrics(keys), "matched_metrics_all_strategies.csv")
        _write(run_top_k_analysis(keys), "concede_top_k.csv")

    if want in ("all", "bootstrap"):
        strategies = [args.strategy] if args.strategy else ["pooled", *TRANSFER_VARIANTS]
        for strategy in strategies:
            variant = None if strategy == "pooled" else strategy
            draw_bootstrap_chunk(keys, args.n_boot, args.seed, variant, args.chunk_id)

    if want in ("all", "bootstrap", "combine"):
        _write(combine_bootstrap(keys), "bootstrap_model_differences.csv")

    if want in ("all", "stratified", "ranking"):
        panel = add_strata(build_action_panel(keys))
        if want in ("all", "stratified"):
            _write(run_stratified(panel, args.n_boot, args.seed), "stratified_delta.csv")
            _write(run_visibility_check(panel), "visibility_check.csv")
        if want in ("all", "ranking"):
            minutes = load_player_minutes(sorted(panel["match_id"].unique().tolist()))
            corr, ranks = run_ranking_bootstrap(panel, minutes, args.n_boot, args.seed)
            _write(corr, "ranking_correlation_ci.csv")
            _write(ranks, "ranking_change_ci.csv")

    log.info("Done. Results in %s", OUT_DIR)


if __name__ == "__main__":
    main()
