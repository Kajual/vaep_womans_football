"""Stage 7 / 9 / 12 — Train the VAEP score and concede classifiers.

Two LightGBM binary classifiers are trained per model:

  * scoring head:  predicts P(team scores within next k actions)
  * concede head:  predicts P(team concedes within next k actions)

Models A / B / C differ only in their feature set, controlled by ``--features``.

The training protocol enforces match-level (group) splits so no actions of
the same match end up in both train and eval — this matters because actions
from the same match are highly correlated and would inflate metrics if
split randomly.

Run:
    # Model A — event-only features.
    python -m src.modelling --model-id model_A --features baseline \\
        --train-corpus men_360_source women_360_finetune \\
        --eval-corpus  women_360_evaluation

    # Model B — baseline + phase (requires phase_features to have run).
    python -m src.modelling --model-id model_B --features baseline phase \\
        --train-corpus men_360_source women_360_finetune \\
        --eval-corpus  women_360_evaluation

    # Model C — baseline + phase + space (requires space_features).
    python -m src.modelling --model-id model_C --features baseline phase space \\
        --train-corpus men_360_source women_360_finetune \\
        --eval-corpus  women_360_evaluation
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import Config, load_config
from .preprocessing import select_matches

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(cfg: Config, feature_sets: list[str]) -> pd.DataFrame:
    """Load and merge the requested feature sets.

    ``feature_sets`` is a list of {"baseline", "phase", "space"}.
    All sets must share the (match_id, action_id) key. Missing files raise.
    """
    files = {
        "baseline": cfg.paths.features_dir / "features_baseline.parquet",
        "phase":    cfg.paths.features_dir / "features_phase.parquet",
        "space":    cfg.paths.features_dir / "space_features.parquet",
    }
    parts: list[pd.DataFrame] = []
    for name in feature_sets:
        path = files[name]
        if not path.exists():
            raise FileNotFoundError(
                f"Feature set '{name}' not found at {path}. "
                f"Run the corresponding pipeline stage first."
            )
        df = pd.read_parquet(path)
        log.info("Loaded %s features: %d rows × %d cols", name, len(df), df.shape[1])
        parts.append(df)

    out = parts[0]
    for nxt in parts[1:]:
        out = out.merge(nxt, on=["match_id", "action_id"], how="inner")
    return out


def load_labels(cfg: Config, k: int) -> pd.DataFrame:
    """Load the VAEP score/concede labels at horizon ``k``."""
    path = cfg.paths.labels_dir / f"vaep_labels_k{k}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Labels file missing: {path}")
    df = pd.read_parquet(path)
    log.info("Loaded labels k=%d: %d rows", k, len(df))
    return df


def filter_to_corpus(df: pd.DataFrame, match_ids: set[int], col: str = "match_id") -> pd.DataFrame:
    return df[df[col].isin(match_ids)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

NON_FEATURE_COLS = {
    "match_id", "action_id", "game_id", "original_event_id", "team_id",
    "player_id", "type_name", "result_name", "bodypart_name", "play_pattern",
    "score_label", "concede_label", "k",
}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that aren't identifiers or labels."""
    cols: list[str] = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def train_one_head(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    early_stopping_rounds: int,
) -> Any:
    """Train one LightGBM binary classifier with early stopping on the val fold."""
    import lightgbm as lgb

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=params.get("n_estimators", 1500),
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return booster


def predict_proba(booster: Any, X: pd.DataFrame) -> np.ndarray:
    """Wrap LightGBM predict for clarity (returns positive-class probabilities)."""
    return booster.predict(X, num_iteration=booster.best_iteration)


def match_level_split(
    match_ids: np.ndarray,
    val_frac: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, val_mask) splits at the *match* granularity."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_seed)
    idx_train, idx_val = next(splitter.split(np.zeros(len(match_ids)), groups=match_ids))
    train_mask = np.zeros(len(match_ids), dtype=bool)
    val_mask = np.zeros(len(match_ids), dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Top-level training routine
# ---------------------------------------------------------------------------

def train_model(
    cfg: Config,
    model_id: str,
    feature_sets: list[str],
    train_corpora: list[str],
    eval_corpora: list[str],
    k: int,
) -> dict[str, Any]:
    """Train scoring + conceding heads and produce predictions on the eval corpus."""

    # 1. Resolve match ids per corpus.
    train_matches = set(select_matches(cfg, competitions_group=train_corpora)["match_id"].astype(int))
    eval_matches = set(select_matches(cfg, competitions_group=eval_corpora)["match_id"].astype(int))
    overlap = train_matches & eval_matches
    if overlap:
        raise ValueError(
            f"Train and eval corpora overlap by {len(overlap)} matches. "
            "Pick disjoint groups (e.g. fine-tune on WC2023+W-Euro2022, eval on W-Euro2025)."
        )
    log.info("Train matches: %d, eval matches: %d", len(train_matches), len(eval_matches))

    # 2. Load features and labels, merge.
    features = load_features(cfg, feature_sets)
    labels = load_labels(cfg, k=k)
    data = features.merge(labels[["match_id", "action_id", "score_label", "concede_label"]],
                         on=["match_id", "action_id"], how="inner")
    log.info("Joined feature+label table: %d rows × %d cols", len(data), data.shape[1])

    feat_cols = _feature_columns(data)
    log.info("Using %d feature columns", len(feat_cols))

    # 3. Split.
    train_df = filter_to_corpus(data, train_matches).copy()
    eval_df = filter_to_corpus(data, eval_matches).copy()
    log.info("Train rows: %d, eval rows: %d", len(train_df), len(eval_df))

    if len(train_df) == 0:
        raise RuntimeError("Empty training set — check that features were generated for the train corpus.")

    train_mask, val_mask = match_level_split(
        train_df["match_id"].to_numpy(),
        val_frac=cfg.split.val_frac,
        random_seed=cfg.split.random_seed,
    )
    X_train = train_df.loc[train_mask, feat_cols]
    X_val = train_df.loc[val_mask, feat_cols]
    y_train_score = train_df.loc[train_mask, "score_label"].to_numpy()
    y_val_score = train_df.loc[val_mask, "score_label"].to_numpy()
    y_train_conc = train_df.loc[train_mask, "concede_label"].to_numpy()
    y_val_conc = train_df.loc[val_mask, "concede_label"].to_numpy()

    log.info("Train fold: %d rows | Val fold: %d rows", len(X_train), len(X_val))

    # 4. Train the two heads.
    params = dict(cfg.modelling.lightgbm_params)
    early_stop = int(cfg.modelling.early_stopping_rounds)

    log.info("Training scoring head ...")
    booster_score = train_one_head(X_train, y_train_score, X_val, y_val_score, params, early_stop)

    log.info("Training conceding head ...")
    booster_conc = train_one_head(X_train, y_train_conc, X_val, y_val_conc, params, early_stop)

    # 5. Predict on eval.
    X_eval = eval_df[feat_cols]
    eval_df["p_score"] = predict_proba(booster_score, X_eval)
    eval_df["p_concede"] = predict_proba(booster_conc, X_eval)

    # 6. Persist artifacts.
    model_dir = cfg.paths.models_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster_score, model_dir / f"{model_id}_score.pkl")
    joblib.dump(booster_conc, model_dir / f"{model_id}_concede.pkl")
    (model_dir / "feature_columns.json").write_text(json.dumps(feat_cols, indent=2))
    (model_dir / "training_metadata.json").write_text(json.dumps({
        "model_id": model_id,
        "feature_sets": feature_sets,
        "train_corpora": train_corpora,
        "eval_corpora": eval_corpora,
        "k": k,
        "n_train_rows": int(train_mask.sum()),
        "n_val_rows": int(val_mask.sum()),
        "n_eval_rows": int(len(eval_df)),
        "best_iter_score": booster_score.best_iteration,
        "best_iter_concede": booster_conc.best_iteration,
    }, indent=2))

    pred_path = cfg.paths.model_outputs_dir / f"{model_id}_predictions.parquet"
    cfg.paths.model_outputs_dir.mkdir(parents=True, exist_ok=True)
    pred_cols = ["match_id", "action_id", "score_label", "concede_label", "p_score", "p_concede"]
    eval_df[pred_cols].to_parquet(pred_path, index=False)
    log.info("Saved predictions to %s", pred_path)

    return {
        "model_id": model_id,
        "predictions_path": str(pred_path),
        "model_dir": str(model_dir),
        "n_eval_rows": len(eval_df),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
@click.option("--model-id", required=True, help="Identifier for this model (becomes the artifact subdir name).")
@click.option(
    "--features", "feature_sets", multiple=True, default=["baseline"],
    type=click.Choice(["baseline", "phase", "space"]),
    help="Which feature sets to include. Pass multiple times to combine.",
)
@click.option(
    "--train-corpus", "train_corpora", multiple=True, required=True,
    help="Competition group(s) to train on. Pass multiple times to union.",
)
@click.option(
    "--eval-corpus", "eval_corpora", multiple=True, required=True,
    help="Competition group(s) to evaluate on. Must be disjoint from --train-corpus.",
)
@click.option("--k", default=None, type=int, help="Label horizon (default: cfg.labels.k_main).")
def main(
    config_path: str | None,
    model_id: str,
    feature_sets: tuple[str, ...],
    train_corpora: tuple[str, ...],
    eval_corpora: tuple[str, ...],
    k: int | None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("models_dir", "model_outputs_dir", "tables_dir")

    k_used = k if k is not None else int(cfg.labels.k_main)
    result = train_model(
        cfg,
        model_id=model_id,
        feature_sets=list(feature_sets),
        train_corpora=list(train_corpora),
        eval_corpora=list(eval_corpora),
        k=k_used,
    )
    log.info("Training complete:\n%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
