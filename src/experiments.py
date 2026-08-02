"""Training-dependent experiments for the MLSA 2026 revision (paper 306).

Each experiment answers a specific promise made in the manuscript or the
response letter. All of them need models to be refit, which is why they live
here rather than in ``rebuttal_analysis.py`` (that module is purely post-hoc).

    transfer      Size-matched and domain-balanced training, learning curves,
                  international-vs-club source composition, and A-distance
                  domain divergence.            (R1-C6/Q5, R1-Q2, R2-C3)

    groups        Leave-one-group-out and add-one-group-in over the four
                  spatial feature groups and the phase layer, with grouped
                  SHAP attributions.                          (R1-C4/Q4)

    pseudospace   Predict each spatial feature from event data alone, rebuild
                  VAEP on the predicted features, and measure how much of the
                  PS-VAEP effect survives without 360 coverage.  (R1-C4/Q4)

    sensitivity   Ten seeds per cell, a small hyperparameter grid, and
                  logistic-regression and XGBoost baselines.    (R1-C8/Q7)

Run:
    python -m src.experiments all
    python -m src.experiments transfer --n-seeds 3     # quick pass

Results are written to outputs/tables/experiments/. Every experiment skips
itself if its output already exists, so a long run can be resumed after a
crash; pass --force to recompute.
"""
from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence

import click
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .config import Config, load_config
from .modelling import load_features, load_labels, _feature_columns
from .preprocessing import select_matches

log = logging.getLogger(__name__)

HEADS = ("score", "concede")

# The four interpretable groups the seventeen spatial features fall into.
# Used by the grouped ablation and by the grouped SHAP attribution.
SPATIAL_GROUPS: dict[str, tuple[str, ...]] = {
    "pressure_support": (
        "nearest_opponent_distance",
        "nearest_teammate_distance",
    ),
    "defensive_density": (
        "defensive_density_around_ball_5m",
        "defensive_density_around_ball_10m",
        "defensive_density_around_ball_15m",
        "defensive_density_in_front_of_ball",
    ),
    "defensive_structure": (
        "n_opponents_ahead_of_ball",
        "n_opponents_between_ball_and_goal",
        "n_opponents_visible",
        "n_teammates_visible",
    ),
    "zone_context": (
        "is_central_zone",
        "is_half_space_left",
        "is_half_space_right",
        "is_wide_zone",
        "is_box_action",
        "is_box_entry",
        "is_final_third_entry",
    ),
}


# ---------------------------------------------------------------------------
# Shared fitting core
# ---------------------------------------------------------------------------

def metrics_for(y: np.ndarray, p: np.ndarray, prefix: str) -> dict[str, float]:
    if y.sum() == 0 or y.sum() == len(y):
        return {f"{prefix}_auc": float("nan"), f"{prefix}_ap": float("nan"),
                f"{prefix}_logloss": float("nan")}
    return {
        f"{prefix}_auc": float(roc_auc_score(y, p)),
        f"{prefix}_ap": float(average_precision_score(y, p)),
        f"{prefix}_logloss": float(log_loss(y, p, labels=[0, 1])),
    }


def _fit_lightgbm(X_tr, y_tr, X_val, y_val, params: dict, early_stop: int):
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    return lgb.train(
        params, dtrain,
        num_boost_round=int(params.get("n_estimators", 1500)),
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stop, verbose=False)],
    )


def _fit_xgboost(X_tr, y_tr, X_val, y_val, params: dict, early_stop: int):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=int(params.get("n_estimators", 1500)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=6,
        subsample=float(params.get("bagging_fraction", 0.8)),
        colsample_bytree=float(params.get("feature_fraction", 0.8)),
        eval_metric="logloss",
        early_stopping_rounds=early_stop,
        tree_method="hist",
        random_state=int(params.get("seed", 42)),
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _fit_logreg(X_tr, y_tr, X_val, y_val, params: dict, early_stop: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    )
    return model.fit(np.nan_to_num(X_tr.to_numpy(dtype=float)), y_tr)


LEARNERS: dict[str, Callable] = {
    "lightgbm": _fit_lightgbm,
    "xgboost": _fit_xgboost,
    "logreg": _fit_logreg,
}


def _predict(model, X: pd.DataFrame, learner: str) -> np.ndarray:
    if learner == "lightgbm":
        return model.predict(X, num_iteration=model.best_iteration)
    if learner == "xgboost":
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(np.nan_to_num(X.to_numpy(dtype=float)))[:, 1]


def fit_and_eval(
    data: pd.DataFrame,
    feat_cols: Sequence[str],
    train_matches: set[int],
    eval_matches: set[int],
    cfg: Config,
    seed: int = 42,
    param_overrides: dict[str, Any] | None = None,
    learner: str = "lightgbm",
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Fit both heads on a match subset and score them on the eval matches.

    This mirrors ``modelling.train_model`` but takes the feature columns and the
    training match ids as arguments, which is what lets the experiments below
    vary the feature set (grouped ablation), the training sample (size-matched
    transfer, learning curves) and the learner (sensitivity) without touching
    the production training path.
    """
    feat_cols = list(feat_cols)
    train_df = data[data["match_id"].isin(train_matches)]
    eval_df = data[data["match_id"].isin(eval_matches)]
    if len(train_df) == 0 or len(eval_df) == 0:
        raise RuntimeError("Empty train or eval split.")

    groups = train_df["match_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.split.val_frac, random_state=seed)
    tr_idx, val_idx = next(splitter.split(train_df, groups=groups))

    params = dict(cfg.modelling.lightgbm_params)
    params.update(param_overrides or {})
    params["seed"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed
    early_stop = int(cfg.modelling.early_stopping_rounds)

    X_tr = train_df.iloc[tr_idx][feat_cols]
    X_val = train_df.iloc[val_idx][feat_cols]
    X_eval = eval_df[feat_cols]

    out: dict[str, Any] = {
        "n_train_matches": len(train_matches),
        "n_train_rows": len(tr_idx),
        "n_eval_rows": len(eval_df),
        "n_features": len(feat_cols),
        "seed": seed,
        "learner": learner,
    }
    preds: dict[str, np.ndarray] = {}
    for head in HEADS:
        y_tr = train_df.iloc[tr_idx][f"{head}_label"].to_numpy()
        y_val = train_df.iloc[val_idx][f"{head}_label"].to_numpy()
        model = LEARNERS[learner](X_tr, y_tr, X_val, y_val, params, early_stop)
        p = _predict(model, X_eval, learner)
        preds[head] = p
        out.update(metrics_for(eval_df[f"{head}_label"].to_numpy(), p, head))

    if return_predictions:
        out["_predictions"] = eval_df[["match_id", "action_id"]].assign(
            p_score=preds["score"], p_concede=preds["concede"],
            score_label=eval_df["score_label"].to_numpy(),
            concede_label=eval_df["concede_label"].to_numpy(),
        )
    return out


def build_dataset(cfg: Config, feature_sets: list[str], k: int) -> pd.DataFrame:
    """Feature + label table, restricted to actions that have all feature sets."""
    features = load_features(cfg, feature_sets)
    labels = load_labels(cfg, k=k)
    data = features.merge(
        labels[["match_id", "action_id", "score_label", "concede_label"]],
        on=["match_id", "action_id"], how="inner",
    )
    log.info("Dataset for %s: %d rows x %d cols", feature_sets, len(data), data.shape[1])
    return data


def corpus_matches(cfg: Config, groups: list[str]) -> set[int]:
    return set(select_matches(cfg, competitions_group=groups)["match_id"].astype(int))


# ---------------------------------------------------------------------------
# Experiment 1: transfer  (R1-C6/Q5, R1-Q2, R2-C3)
# ---------------------------------------------------------------------------

def a_distance(
    data: pd.DataFrame,
    feat_cols: Sequence[str],
    matches_a: set[int],
    matches_b: set[int],
    cfg: Config,
    seed: int = 42,
    max_rows: int = 60000,
) -> float:
    """Proxy A-distance between two corpora.

    A domain classifier is trained to tell the two corpora apart from their
    action feature vectors; the proxy A-distance is 2(1 - 2*error). A value near
    0 means the domains are indistinguishable, near 2 that they are trivially
    separable. This is the standard empirical stand-in for the divergence term
    in domain-adaptation bounds (Ben-David et al.).
    """
    rng = np.random.default_rng(seed)
    a = data[data["match_id"].isin(matches_a)]
    b = data[data["match_id"].isin(matches_b)]
    n = min(len(a), len(b), max_rows // 2)
    if n < 100:
        return float("nan")
    a = a.sample(n=n, random_state=seed)
    b = b.sample(n=n, random_state=seed)

    pooled = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    groups = pooled["match_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(splitter.split(pooled, groups=groups))

    params = dict(cfg.modelling.lightgbm_params)
    params["n_estimators"] = 200
    model = _fit_lightgbm(
        pooled.iloc[tr][list(feat_cols)], y[tr],
        pooled.iloc[te][list(feat_cols)], y[te],
        params, 20,
    )
    p = model.predict(pooled.iloc[te][list(feat_cols)], num_iteration=model.best_iteration)
    err = float(np.mean((p > 0.5).astype(int) != y[te]))
    return float(2.0 * (1.0 - 2.0 * err))


def run_transfer(cfg: Config, k: int, n_seeds: int, out_dir: Path) -> None:
    """Separate the value of more data from the value of cross-domain data."""
    data = build_dataset(cfg, ["baseline", "phase", "space"], k)
    feat_cols = _feature_columns(data)

    men = corpus_matches(cfg, ["men_360_source"])
    women_tr = corpus_matches(cfg, ["women_360_finetune"])
    eval_m = corpus_matches(cfg, ["women_360_evaluation"])

    # Competition-type split of the men's source.
    inv = select_matches(cfg, competitions_group=["men_360_source"])
    comp_col = "competition_name" if "competition_name" in inv.columns else None
    if comp_col:
        club_mask = inv[comp_col].str.contains("Bundesliga", case=False, na=False)
        men_club = set(inv.loc[club_mask, "match_id"].astype(int))
        men_intl = set(inv.loc[~club_mask, "match_id"].astype(int))
    else:
        log.warning("No competition_name in inventory — skipping competition-type split.")
        men_club, men_intl = set(), set()

    rows: list[dict] = []
    rng = np.random.default_rng(cfg.split.random_seed)
    n_women = len(women_tr)

    def add(label: str, train: set[int], seed: int) -> None:
        if len(train) < 5:
            return
        res = fit_and_eval(data, feat_cols, train, eval_m, cfg, seed=seed)
        rows.append({"experiment": label, **res})
        log.info("%s (seed %d): concede AUC %.4f", label, seed, res["concede_auc"])

    for seed in range(n_seeds):
        s = cfg.split.random_seed + seed
        r = np.random.default_rng(s)

        # Size-matched: subsample men's to the women's match count, so pooled
        # and single-domain strategies see the same number of matches.
        men_sub = set(r.choice(sorted(men), size=min(n_women, len(men)), replace=False).tolist())
        add("women_only", women_tr, s)
        add("men_only_full", men, s)
        add("men_only_size_matched", men_sub, s)
        add("pooled_full", men | women_tr, s)
        add("pooled_size_matched", men_sub | women_tr, s)

        # Domain-balanced pooling: equal match counts per domain, and a
        # women-only control at the same total size, which isolates whether the
        # benefit is cross-domain or simply more data.
        half = n_women
        men_half = set(r.choice(sorted(men), size=min(half, len(men)), replace=False).tolist())
        add("pooled_domain_balanced", men_half | women_tr, s)

        if men_intl and men_club:
            n_club = len(men_club)
            intl_sub = set(r.choice(sorted(men_intl), size=min(n_club, len(men_intl)),
                                    replace=False).tolist())
            add("men_international_size_matched", intl_sub, s)
            add("men_club_size_matched", men_club, s)

    pd.DataFrame(rows).to_csv(out_dir / "transfer_controls.csv", index=False)
    log.info("Wrote transfer_controls.csv")

    # Learning curves: target metrics against the number of training matches,
    # separately per source composition.
    curve: list[dict] = []
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    for name, pool in (("men", men), ("women", women_tr), ("pooled", men | women_tr)):
        for frac in fractions:
            size = max(5, int(round(frac * len(pool))))
            for seed in range(max(1, n_seeds - 1)):
                s = cfg.split.random_seed + seed
                r = np.random.default_rng(s + size)
                sub = set(r.choice(sorted(pool), size=min(size, len(pool)), replace=False).tolist())
                try:
                    res = fit_and_eval(data, feat_cols, sub, eval_m, cfg, seed=s)
                except RuntimeError:
                    continue
                curve.append({"source": name, "frac": frac, **res})
                log.info("curve %s n=%d seed=%d: concede AUC %.4f",
                         name, size, s, res["concede_auc"])
    pd.DataFrame(curve).to_csv(out_dir / "transfer_learning_curves.csv", index=False)
    log.info("Wrote transfer_learning_curves.csv")

    # Domain divergence between the corpora we actually transfer between.
    div: list[dict] = []
    pairs = [("men_all", men, "women_target", women_tr)]
    if men_intl and men_club:
        pairs += [
            ("men_international", men_intl, "women_target", women_tr),
            ("men_club", men_club, "women_target", women_tr),
            ("men_international", men_intl, "men_club", men_club),
        ]
    for na, ma, nb, mb in pairs:
        d = a_distance(data, feat_cols, ma, mb, cfg)
        div.append({"domain_a": na, "domain_b": nb, "n_matches_a": len(ma),
                    "n_matches_b": len(mb), "proxy_a_distance": d})
        log.info("A-distance %s vs %s: %.3f", na, nb, d)
    pd.DataFrame(div).to_csv(out_dir / "domain_divergence.csv", index=False)
    log.info("Wrote domain_divergence.csv")


# ---------------------------------------------------------------------------
# Experiment 2: grouped ablation  (R1-C4/Q4)
# ---------------------------------------------------------------------------

def run_groups(cfg: Config, k: int, out_dir: Path) -> None:
    """Leave-one-group-out and add-one-group-in over phase and spatial groups."""
    data = build_dataset(cfg, ["baseline", "phase", "space"], k)
    all_feats = _feature_columns(data)

    phase_cols = [c for c in all_feats if c.startswith("phase_")]
    spatial_cols = {g: [c for c in cols if c in all_feats]
                    for g, cols in SPATIAL_GROUPS.items()}
    grouped = set(phase_cols) | {c for cols in spatial_cols.values() for c in cols}
    baseline_cols = [c for c in all_feats if c not in grouped]

    groups: dict[str, list[str]] = {"phase": phase_cols, **spatial_cols}
    train = corpus_matches(cfg, ["men_360_source", "women_360_finetune"])
    eval_m = corpus_matches(cfg, ["women_360_evaluation"])

    rows: list[dict] = []

    def run(label: str, cols: list[str]) -> dict:
        res = fit_and_eval(data, cols, train, eval_m, cfg)
        rows.append({"variant": label, **res})
        log.info("%s: concede AUC %.4f (%d feats)", label, res["concede_auc"], len(cols))
        return res

    run("baseline_only", baseline_cols)
    run("all_groups", all_feats)
    for name, cols in groups.items():
        if not cols:
            continue
        run(f"add_only_{name}", baseline_cols + cols)          # add-one-group-in
        run(f"drop_{name}", [c for c in all_feats if c not in cols])  # leave-one-out

    pd.DataFrame(rows).to_csv(out_dir / "grouped_ablation.csv", index=False)
    log.info("Wrote grouped_ablation.csv")

    # Grouped SHAP, via LightGBM's own contribution output so that no extra
    # dependency is required. Contributions are summed within each group.
    import lightgbm as lgb
    res = fit_and_eval(data, all_feats, train, eval_m, cfg)
    eval_df = data[data["match_id"].isin(eval_m)]
    tr_df = data[data["match_id"].isin(train)]
    params = dict(cfg.modelling.lightgbm_params)
    shap_rows: list[dict] = []
    for head in HEADS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.split.val_frac,
                                     random_state=cfg.split.random_seed)
        tr_idx, val_idx = next(splitter.split(tr_df, groups=tr_df["match_id"].to_numpy()))
        booster = _fit_lightgbm(
            tr_df.iloc[tr_idx][all_feats], tr_df.iloc[tr_idx][f"{head}_label"].to_numpy(),
            tr_df.iloc[val_idx][all_feats], tr_df.iloc[val_idx][f"{head}_label"].to_numpy(),
            params, int(cfg.modelling.early_stopping_rounds),
        )
        sample = eval_df.sample(n=min(20000, len(eval_df)), random_state=42)
        contrib = booster.predict(sample[all_feats], pred_contrib=True,
                                  num_iteration=booster.best_iteration)
        contrib = np.abs(contrib[:, :-1]).mean(axis=0)  # drop the bias column
        by_feat = dict(zip(all_feats, contrib))
        for gname, cols in {"baseline": baseline_cols, **groups}.items():
            shap_rows.append({
                "head": head, "group": gname,
                "mean_abs_shap": float(sum(by_feat.get(c, 0.0) for c in cols)),
                "n_features": len(cols),
            })
    pd.DataFrame(shap_rows).to_csv(out_dir / "grouped_shap.csv", index=False)
    log.info("Wrote grouped_shap.csv")


# ---------------------------------------------------------------------------
# Experiment 3: Pseudo-Space VAEP  (R1-C4/Q4)
# ---------------------------------------------------------------------------

def run_pseudospace(cfg: Config, k: int, out_dir: Path) -> None:
    """Can the spatial features be approximated from event data alone?

    For each spatial feature we fit a model predicting it from the event
    features, then rebuild VAEP on the predicted values. If the PS-VAEP effect
    survives, the approach transfers to competitions without 360 coverage.
    """
    import lightgbm as lgb

    data = build_dataset(cfg, ["baseline", "phase", "space"], k)
    all_feats = _feature_columns(data)
    spatial_cols = [c for cols in SPATIAL_GROUPS.values() for c in cols if c in all_feats]
    event_cols = [c for c in all_feats if c not in spatial_cols]

    train = corpus_matches(cfg, ["men_360_source", "women_360_finetune"])
    eval_m = corpus_matches(cfg, ["women_360_evaluation"])
    tr_df = data[data["match_id"].isin(train)]
    ev_df = data[data["match_id"].isin(eval_m)]

    quality: list[dict] = []
    predicted = {}
    for col in spatial_cols:
        binary = set(np.unique(data[col].dropna())) <= {0, 1}
        params = dict(cfg.modelling.lightgbm_params)
        params["objective"] = "binary" if binary else "regression"
        params["metric"] = "binary_logloss" if binary else "l2"
        params["n_estimators"] = 400

        splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.split.val_frac,
                                     random_state=cfg.split.random_seed)
        tr_idx, val_idx = next(splitter.split(tr_df, groups=tr_df["match_id"].to_numpy()))
        booster = _fit_lightgbm(
            tr_df.iloc[tr_idx][event_cols], tr_df.iloc[tr_idx][col].to_numpy(),
            tr_df.iloc[val_idx][event_cols], tr_df.iloc[val_idx][col].to_numpy(),
            params, 30,
        )
        pred_tr = booster.predict(tr_df[event_cols], num_iteration=booster.best_iteration)
        pred_ev = booster.predict(ev_df[event_cols], num_iteration=booster.best_iteration)
        predicted[col] = (pred_tr, pred_ev)

        actual = ev_df[col].to_numpy()
        if binary:
            score = roc_auc_score(actual, pred_ev) if 0 < actual.sum() < len(actual) else float("nan")
            quality.append({"feature": col, "kind": "binary", "metric": "roc_auc", "value": score})
        else:
            ss_res = float(np.sum((actual - pred_ev) ** 2))
            ss_tot = float(np.sum((actual - actual.mean()) ** 2))
            quality.append({"feature": col, "kind": "continuous", "metric": "r2",
                            "value": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")})
        log.info("approximated %s: %s", col, quality[-1]["value"])

    pd.DataFrame(quality).to_csv(out_dir / "pseudospace_feature_quality.csv", index=False)

    # Rebuild the model on predicted spatial features.
    pseudo = data.copy()
    idx_tr = pseudo.index[pseudo["match_id"].isin(train)]
    idx_ev = pseudo.index[pseudo["match_id"].isin(eval_m)]
    for col, (ptr, pev) in predicted.items():
        pseudo.loc[idx_tr, col] = ptr
        pseudo.loc[idx_ev, col] = pev

    rows = []
    real = fit_and_eval(data, all_feats, train, eval_m, cfg, return_predictions=True)
    fake = fit_and_eval(pseudo, all_feats, train, eval_m, cfg, return_predictions=True)
    for label, res in (("PS-VAEP (observed 360)", real), ("Pseudo-Space VAEP (predicted)", fake)):
        rows.append({"variant": label,
                     **{kk: vv for kk, vv in res.items() if not kk.startswith("_")}})
    pd.DataFrame(rows).to_csv(out_dir / "pseudospace_metrics.csv", index=False)

    real["_predictions"].to_parquet(out_dir / "pseudospace_pred_real.parquet", index=False)
    fake["_predictions"].to_parquet(out_dir / "pseudospace_pred_fake.parquet", index=False)
    log.info("Wrote pseudospace_*.csv/parquet")


# ---------------------------------------------------------------------------
# Experiment 4: sensitivity  (R1-C8/Q7)
# ---------------------------------------------------------------------------

def run_sensitivity(cfg: Config, k: int, n_seeds: int, out_dir: Path) -> None:
    """Seed noise, hyperparameter sensitivity and learner sensitivity.

    The point is to let the reader compare between-variant differences against
    the variation induced by arbitrary choices, so we report the same three
    feature sets under each perturbation.
    """
    variants = {
        "E-VAEP": ["baseline"],
        "P-VAEP": ["baseline", "phase"],
        "PS-VAEP": ["baseline", "phase", "space"],
        "ES-VAEP": ["baseline", "space"],
    }
    train = corpus_matches(cfg, ["men_360_source", "women_360_finetune"])
    eval_m = corpus_matches(cfg, ["women_360_evaluation"])

    # Every variant must be scored on the same action set, so restrict to the
    # freeze-frame subset that the space models are defined on.
    space_data = build_dataset(cfg, ["baseline", "phase", "space"], k)
    matched_keys = space_data[["match_id", "action_id"]].drop_duplicates()

    seed_rows: list[dict] = []
    grid_rows: list[dict] = []
    learner_rows: list[dict] = []

    for name, fsets in variants.items():
        data = build_dataset(cfg, fsets, k).merge(matched_keys, on=["match_id", "action_id"])
        cols = _feature_columns(data)

        for seed in range(n_seeds):
            res = fit_and_eval(data, cols, train, eval_m, cfg, seed=cfg.split.random_seed + seed)
            seed_rows.append({"variant": name, **res})
            log.info("[seed] %s seed=%d concede AUC %.4f", name, seed, res["concede_auc"])

        for leaves, lr, ff in product([31, 63, 127], [0.03, 0.05, 0.1], [0.6, 0.8]):
            res = fit_and_eval(data, cols, train, eval_m, cfg,
                               param_overrides={"num_leaves": leaves, "learning_rate": lr,
                                                "feature_fraction": ff})
            grid_rows.append({"variant": name, "num_leaves": leaves, "learning_rate": lr,
                              "feature_fraction": ff, **res})
            log.info("[grid] %s leaves=%d lr=%.2f ff=%.1f concede AUC %.4f",
                     name, leaves, lr, ff, res["concede_auc"])

        for learner in ("logreg", "xgboost"):
            try:
                res = fit_and_eval(data, cols, train, eval_m, cfg, learner=learner)
            except ImportError:
                log.warning("%s unavailable — skipping.", learner)
                continue
            learner_rows.append({"variant": name, **res})
            log.info("[learner] %s %s concede AUC %.4f", name, learner, res["concede_auc"])

    pd.DataFrame(seed_rows).to_csv(out_dir / "sensitivity_seeds.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(out_dir / "sensitivity_grid.csv", index=False)
    pd.DataFrame(learner_rows).to_csv(out_dir / "sensitivity_learners.csv", index=False)
    log.info("Wrote sensitivity_*.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "transfer": lambda cfg, k, n, d: run_transfer(cfg, k, n, d),
    "groups": lambda cfg, k, n, d: run_groups(cfg, k, d),
    "pseudospace": lambda cfg, k, n, d: run_pseudospace(cfg, k, d),
    "sensitivity": lambda cfg, k, n, d: run_sensitivity(cfg, k, n, d),
}

SENTINELS = {
    "transfer": "domain_divergence.csv",
    "groups": "grouped_shap.csv",
    "pseudospace": "pseudospace_metrics.csv",
    "sensitivity": "sensitivity_learners.csv",
}


@click.command()
@click.argument("which", type=click.Choice([*EXPERIMENTS, "all"]))
@click.option("--config", "config_path", default=None)
@click.option("--k", default=None, type=int, help="Label horizon (default cfg.labels.k_main).")
@click.option("--n-seeds", default=10, type=int,
              help="Seeds/repeats for the transfer and sensitivity experiments.")
@click.option("--force", is_flag=True, default=False,
              help="Recompute even if the experiment's output already exists.")
def main(which: str, config_path: str | None, k: int | None, n_seeds: int, force: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    k_used = k if k is not None else int(cfg.labels.k_main)

    out_dir = Path(cfg.paths.tables_dir) / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = list(EXPERIMENTS) if which == "all" else [which]
    for name in todo:
        sentinel = out_dir / SENTINELS[name]
        if sentinel.exists() and not force:
            log.info("SKIP %s — %s already exists (use --force to redo).", name, sentinel.name)
            continue
        log.info("=== RUNNING %s ===", name)
        EXPERIMENTS[name](cfg, k_used, n_seeds, out_dir)

    log.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
