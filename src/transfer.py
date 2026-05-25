"""Stage 13 — men-to-women transfer-learning experiment.

For one feature set, this module trains and evaluates the three transfer
variants defined in ``configs/default.yaml`` under ``transfer_learning``:

  * ``men_only``    — train on the men's 360 corpus, evaluate on women's
                      EURO 2025 with no target-domain adaptation (zero-shot).
  * ``women_only``  — control: train on the women's fine-tune corpus only.
  * ``fine_tuned``  — pre-train on the men's corpus, then continue training
                      (LightGBM ``init_model``) on the women's fine-tune
                      corpus with a reduced learning rate.

All three are evaluated on the same held-out women's corpus
(``women_360_evaluation``). A fixed slice of ``women_360_finetune`` is carved
off as a *calibration fold* — held out from training in every variant — and
used to fit isotonic target-domain calibrators, so each variant is reported
both raw and calibrated. This keeps the comparison fair across variants.

The men's corpus is trained once: that booster is the ``men_only`` model and
also the starting point for ``fine_tuned``.

Outputs
-------
* ``models/<model_id>_<variant>/``                       — boosters + metadata
* ``data/processed/model_outputs/<model_id>_<variant>_predictions.parquet``
* ``outputs/tables/transfer_comparison_<model_id>.csv``  — tidy metrics table

Run (one feature set per invocation, mirroring ``src.modelling``):

    # Model A feature set — event-only.
    python -m src.transfer --model-id model_A --features baseline

    # Model B feature set — baseline + phase.
    python -m src.transfer --model-id model_B --features baseline --features phase

    # Model C feature set — baseline + phase + space.
    python -m src.transfer --model-id model_C --features baseline --features phase --features space
"""
from __future__ import annotations

import json
import logging
from typing import Any

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .config import Config, load_config
from .modelling import (
    load_features,
    load_labels,
    match_level_split,
    predict_proba,
    train_one_head,
    _feature_columns,
)
from .preprocessing import select_matches

log = logging.getLogger(__name__)

VARIANTS = ("men_only", "women_only", "fine_tuned")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15) -> float:
    """Count-weighted mean gap between predicted and observed frequency."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred, edges[1:-1]), 0, n_bins - 1)
    total = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        conf = float(y_pred[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)


def _head_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metric bundle for one head, unprefixed (suitable for a tidy table)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(np.unique(y_true)) < 2:
        log.warning("Head has only one class in the evaluation set; metrics ill-defined.")
        return {
            "auc": float("nan"),
            "avg_precision": float("nan"),
            "brier": float(np.mean((y_pred - y_true) ** 2)),
            "log_loss": float("nan"),
            "ece": float("nan"),
            "pos_rate_pct": float(y_true.mean() * 100),
        }
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "avg_precision": float(average_precision_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, np.clip(y_pred, 1e-9, 1 - 1e-9))),
        "ece": _expected_calibration_error(y_true, y_pred),
        "pos_rate_pct": float(y_true.mean() * 100),
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(cfg: Config, feature_sets: list[str], k: int) -> tuple[pd.DataFrame, list[str]]:
    """Load features + labels, merge, and return (data, feature_columns)."""
    features = load_features(cfg, feature_sets)
    labels = load_labels(cfg, k=k)
    data = features.merge(
        labels[["match_id", "action_id", "score_label", "concede_label"]],
        on=["match_id", "action_id"],
        how="inner",
    )
    feat_cols = _feature_columns(data)
    log.info("Joined feature+label table: %d rows x %d cols (%d feature columns)",
             len(data), data.shape[1], len(feat_cols))
    return data, feat_cols


def _corpus_match_ids(cfg: Config, group: str) -> set[int]:
    df = select_matches(cfg, competitions_group=[group])
    return set(int(m) for m in df["match_id"].astype(int))


def _split_match_ids(match_ids: set[int], frac: float, seed: int) -> tuple[set[int], set[int]]:
    """Deterministically split a set of match ids into (major, minor) by ``frac``."""
    ids = sorted(int(m) for m in match_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_minor = max(1, int(round(len(ids) * frac)))
    return set(ids[n_minor:]), set(ids[:n_minor])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _continue_train_head(
    init_booster: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    n_more: int,
    early_stopping_rounds: int,
) -> Any:
    """Continue boosting from ``init_booster`` on a new (target-domain) fold."""
    import lightgbm as lgb

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=n_more,
        valid_sets=[val_set],
        valid_names=["val"],
        init_model=init_booster,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return booster


def _train_two_heads(
    df: pd.DataFrame,
    feat_cols: list[str],
    params: dict[str, Any],
    early_stop: int,
    seed: int,
    val_frac: float,
) -> tuple[Any, Any]:
    """Train scoring + conceding heads from scratch on ``df`` (match-level split)."""
    if df.empty:
        raise RuntimeError("Empty training corpus — check feature coverage for this group.")
    tr_mask, val_mask = match_level_split(df["match_id"].to_numpy(), val_frac, seed)
    X_tr, X_val = df.loc[tr_mask, feat_cols], df.loc[val_mask, feat_cols]
    log.info("  train fold: %d rows | val fold: %d rows", len(X_tr), len(X_val))
    booster_score = train_one_head(
        X_tr, df.loc[tr_mask, "score_label"].to_numpy(),
        X_val, df.loc[val_mask, "score_label"].to_numpy(), params, early_stop)
    booster_conc = train_one_head(
        X_tr, df.loc[tr_mask, "concede_label"].to_numpy(),
        X_val, df.loc[val_mask, "concede_label"].to_numpy(), params, early_stop)
    return booster_score, booster_conc


def _finetune_two_heads(
    df: pd.DataFrame,
    feat_cols: list[str],
    ft_params: dict[str, Any],
    n_more: int,
    early_stop: int,
    seed: int,
    val_frac: float,
    init_score: Any,
    init_conc: Any,
) -> tuple[Any, Any]:
    """Continue-train both heads on the women's fold from men-pretrained boosters."""
    if df.empty:
        raise RuntimeError("Empty fine-tune corpus — check feature coverage.")
    tr_mask, val_mask = match_level_split(df["match_id"].to_numpy(), val_frac, seed)
    X_tr, X_val = df.loc[tr_mask, feat_cols], df.loc[val_mask, feat_cols]
    log.info("  fine-tune fold: %d rows | val fold: %d rows", len(X_tr), len(X_val))
    booster_score = _continue_train_head(
        init_score, X_tr, df.loc[tr_mask, "score_label"].to_numpy(),
        X_val, df.loc[val_mask, "score_label"].to_numpy(), ft_params, n_more, early_stop)
    booster_conc = _continue_train_head(
        init_conc, X_tr, df.loc[tr_mask, "concede_label"].to_numpy(),
        X_val, df.loc[val_mask, "concede_label"].to_numpy(), ft_params, n_more, early_stop)
    return booster_score, booster_conc


def _fit_isotonic(y_true: np.ndarray, y_pred: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic calibrator mapping raw probabilities to observed frequency."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(y_pred, y_true)
    return iso


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_variant(
    cfg: Config,
    model_id: str,
    variant: str,
    booster_score: Any,
    booster_conc: Any,
    feat_cols: list[str],
    eval_df: pd.DataFrame,
    p_score: np.ndarray,
    p_concede: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Persist boosters, feature list, metadata, and the eval-set predictions."""
    variant_id = f"{model_id}_{variant}"
    model_dir = cfg.paths.models_dir / variant_id
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster_score, model_dir / f"{variant_id}_score.pkl")
    joblib.dump(booster_conc, model_dir / f"{variant_id}_concede.pkl")
    (model_dir / "feature_columns.json").write_text(json.dumps(feat_cols, indent=2))
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    out = eval_df[["match_id", "action_id", "score_label", "concede_label"]].copy()
    out["p_score"] = p_score
    out["p_concede"] = p_concede
    pred_path = cfg.paths.model_outputs_dir / f"{variant_id}_predictions.parquet"
    out.to_parquet(pred_path, index=False)
    log.info("  saved boosters to %s and predictions to %s", model_dir, pred_path)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_transfer(cfg: Config, model_id: str, feature_sets: list[str], k: int) -> pd.DataFrame:
    """Run the three transfer variants for one feature set; return the tidy table."""
    data, feat_cols = prepare_data(cfg, feature_sets, k)

    # --- resolve the three corpora -----------------------------------------
    men_ids = _corpus_match_ids(cfg, "men_360_source")
    women_ft_ids = _corpus_match_ids(cfg, "women_360_finetune")
    eval_ids = _corpus_match_ids(cfg, "women_360_evaluation")
    if not men_ids or not women_ft_ids or not eval_ids:
        raise RuntimeError("One of the transfer corpora resolved to zero matches.")
    if (men_ids | women_ft_ids) & eval_ids:
        raise ValueError("Training corpora overlap with the evaluation corpus.")

    seed = int(cfg.split.random_seed)
    val_frac = float(cfg.split.val_frac)

    # A fixed calibration fold, carved from the women's fine-tune corpus and
    # held out from training in EVERY variant so the calibrators are fair.
    women_train_ids, calib_ids = _split_match_ids(women_ft_ids, val_frac, seed)
    log.info("Men matches: %d | women train: %d | women calib: %d | eval: %d",
             len(men_ids), len(women_train_ids), len(calib_ids), len(eval_ids))

    def subset(ids: set[int]) -> pd.DataFrame:
        return data[data["match_id"].astype(int).isin(ids)].reset_index(drop=True)

    men_df = subset(men_ids)
    women_df = subset(women_train_ids)
    calib_df = subset(calib_ids)
    eval_df = subset(eval_ids)
    log.info("Row counts -- men: %d | women: %d | calib: %d | eval: %d",
             len(men_df), len(women_df), len(calib_df), len(eval_df))
    for name, frame in [("men", men_df), ("women", women_df),
                        ("calibration", calib_df), ("evaluation", eval_df)]:
        if frame.empty:
            raise RuntimeError(f"The {name} subset is empty for feature set {feature_sets}.")

    params = dict(cfg.modelling.lightgbm_params)
    early_stop = int(cfg.modelling.early_stopping_rounds)
    ft_mult = float(cfg.transfer_learning.fine_tune_learning_rate_mult)
    ft_n_more = int(cfg.transfer_learning.fine_tune_n_estimators)
    calib_method = str(cfg.transfer_learning.calibration_method).lower()

    # --- train the three variants ------------------------------------------
    log.info("[%s] Pre-training on the men's corpus (== men_only variant) ...", model_id)
    men_score, men_conc = _train_two_heads(men_df, feat_cols, params, early_stop, seed, val_frac)

    log.info("[%s] Training the women-only control ...", model_id)
    women_score, women_conc = _train_two_heads(women_df, feat_cols, params, early_stop, seed, val_frac)

    log.info("[%s] Fine-tuning the men's model on the women's corpus ...", model_id)
    ft_params = dict(params)
    ft_params["learning_rate"] = float(params.get("learning_rate", 0.05)) * ft_mult
    ft_score, ft_conc = _finetune_two_heads(
        women_df, feat_cols, ft_params, ft_n_more, early_stop, seed, val_frac,
        men_score, men_conc)

    boosters = {
        "men_only": (men_score, men_conc),
        "women_only": (women_score, women_conc),
        "fine_tuned": (ft_score, ft_conc),
    }

    # --- evaluate ----------------------------------------------------------
    X_eval = eval_df[feat_cols]
    X_calib = calib_df[feat_cols]
    y_eval_score = eval_df["score_label"].to_numpy()
    y_eval_conc = eval_df["concede_label"].to_numpy()
    y_calib_score = calib_df["score_label"].to_numpy()
    y_calib_conc = calib_df["concede_label"].to_numpy()

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        b_score, b_conc = boosters[variant]
        raw_score = predict_proba(b_score, X_eval)
        raw_conc = predict_proba(b_conc, X_eval)

        _save_variant(
            cfg, model_id, variant, b_score, b_conc, feat_cols, eval_df,
            raw_score, raw_conc,
            metadata={
                "model_id": model_id, "variant": variant,
                "feature_sets": feature_sets, "k": k,
                "n_men_rows": len(men_df), "n_women_rows": len(women_df),
                "n_calib_rows": len(calib_df), "n_eval_rows": len(eval_df),
                "best_iter_score": int(getattr(b_score, "best_iteration", 0) or 0),
                "best_iter_concede": int(getattr(b_conc, "best_iteration", 0) or 0),
            },
        )

        def _emit(calibration: str, head: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
            m = _head_metrics(y_true, y_pred)
            rows.append({
                "model_id": model_id, "variant": variant,
                "calibration": calibration, "head": head,
                "n_eval_rows": len(eval_df), **m,
            })

        _emit("raw", "score", y_eval_score, raw_score)
        _emit("raw", "concede", y_eval_conc, raw_conc)

        if calib_method == "isotonic":
            iso_score = _fit_isotonic(y_calib_score, predict_proba(b_score, X_calib))
            iso_conc = _fit_isotonic(y_calib_conc, predict_proba(b_conc, X_calib))
            _emit("isotonic", "score", y_eval_score, iso_score.predict(raw_score))
            _emit("isotonic", "concede", y_eval_conc, iso_conc.predict(raw_conc))

    table = pd.DataFrame(rows)
    out_path = cfg.paths.tables_dir / f"transfer_comparison_{model_id}.csv"
    table.round(6).to_csv(out_path, index=False)
    log.info("Wrote transfer comparison to %s", out_path)
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default=None, help="Override config YAML path.")
@click.option("--model-id", required=True,
              help="Feature-set identifier (model_A / model_B / model_C). "
                   "Variant artifacts are named <model_id>_<variant>.")
@click.option("--features", "feature_sets", multiple=True, default=["baseline"],
              type=click.Choice(["baseline", "phase", "space"]),
              help="Feature sets to include. Pass multiple times to combine.")
@click.option("--k", default=None, type=int, help="Label horizon (default: cfg.labels.k_main).")
def main(config_path: str | None, model_id: str, feature_sets: tuple[str, ...],
         k: int | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs("models_dir", "model_outputs_dir", "tables_dir")

    k_used = k if k is not None else int(cfg.labels.k_main)
    table = run_transfer(cfg, model_id, list(feature_sets), k_used)

    # Compact console summary: total VAEP heads side by side.
    summary = table[table["head"] == "score"].pivot_table(
        index=["variant", "calibration"], values=["auc", "brier", "log_loss", "ece"])
    log.info("Transfer experiment complete for %s.\nScoring-head summary:\n%s",
             model_id, summary.to_string())


if __name__ == "__main__":
    main()
