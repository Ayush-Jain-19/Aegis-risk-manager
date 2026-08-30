"""
src/tune.py

Optuna hyperparameter search for the LightGBM fraud model, evaluated with
time-series cross-validation and scored on PR-AUC.

WHY TIME-SERIES CV, NOT RANDOM K-FOLD, FOR HYPERPARAMETER SEARCH
---------------------------------------------------------------------
Every leakage-avoidance decision made elsewhere in this project (the
chronological fraudTrain/fraudTest split, the time-ordered fit/validation
split in train.py) exists to answer one question honestly: how will this
model perform on transactions that happen AFTER the ones it was trained
on? A random K-fold split during hyperparameter search would silently
undo all of that — it would let each fold's "validation" rows sit
chronologically BEFORE some of that fold's "training" rows, so trials
would end up selecting hyperparameters that look great at predicting the
past from the future, a capability the deployed model will never have.
sklearn's TimeSeriesSplit instead produces folds where every validation
set is strictly later in time than everything it's evaluated against —
fold 1 trains on the earliest slice and validates on the next slice, fold
2 trains on everything up through that point and validates on the slice
after THAT, and so on. This is slower to reason about than K-fold (folds
aren't symmetric — later folds train on more data) but it is the only
splitting strategy consistent with how this model will actually be used.

WHY PR-AUC, NOT ACCURACY OR ROC-AUC
---------------------------------------
Accuracy on a ~2-3% fraud dataset is nearly meaningless — a model that
predicts "not fraud" on every single transaction scores >97% accuracy
while catching zero fraud. ROC-AUC is more informative but is still
computed against the (huge) true-negative count, which makes it
insensitive to exactly the failure mode this system cares most about:
ranking the tiny number of real fraud cases above the enormous number of
legitimate ones. Average Precision (the area under the precision-recall
curve) is driven almost entirely by how the model handles the minority
class, which is the right thing to optimize a fraud model's
hyperparameters against.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from src.config_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Silence Optuna's own per-trial logging in favor of our own summary lines —
# at 50 trials x 3 folds, Optuna's default verbosity is noisy enough to bury
# the signal (which trial is currently running, what it scored) in repeated
# framework boilerplate.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_and_sort_data(config: dict) -> pd.DataFrame:
    """
    Loads train_cleaned.parquet and ensures it's in chronological order
    before any CV split touches it.

    WHY THIS FUNCTION CHECKS FOR THE TIMESTAMP COLUMN RATHER THAN ASSUMING
    IT EXISTS: data_preprocessing.py deliberately drops the raw
    `trans_date_trans_time` column after deriving hour/day-of-week/month
    features from it (see that file's `run_pipeline`) — so by the time
    train_cleaned.parquet reaches this script, there usually is NO
    timestamp column to sort by. That isn't a bug to route around
    silently: this function explicitly checks for the configured
    timestamp column, and if it isn't present, falls back to the
    dataframe's EXISTING row order — which is still correct, because
    nothing in data_preprocessing.py ever reorders rows; it only drops
    and adds columns. pandas preserves row order through every
    transformation in that pipeline, and parquet preserves row order on
    disk, so "the order the rows are already in" IS the chronological
    order, even with the explicit timestamp column gone. This function
    logs which path it took so that fact is never silently assumed.
    """
    train_path = Path(config["paths"]["train_processed"])
    df = pd.read_parquet(train_path)

    timestamp_col = config.get("columns", {}).get("timestamp_col")
    if timestamp_col and timestamp_col in df.columns:
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        logger.info("Sorted %d rows chronologically by '%s'.", len(df), timestamp_col)
    else:
        logger.info(
            "No timestamp column ('%s') found in %s — relying on existing row order, "
            "which data_preprocessing.py preserves as chronological (it only drops/adds "
            "columns, never reorders rows).",
            timestamp_col, train_path,
        )
    return df


def compute_scale_pos_weight_bounds(y: pd.Series, tightness: float = 0.3) -> tuple[float, float]:
    """
    Computes a search range for `scale_pos_weight` centered on the TRUE
    class imbalance ratio, rather than either hardcoding it or letting
    Optuna search a wide, uninformed range.

    WHY A TIGHT RANGE AROUND THE TRUE RATIO, NOT A FREE SEARCH: the
    mathematically "correct" scale_pos_weight for balancing a loss
    function is n_negative / n_positive — that's not a hyperparameter to
    discover from scratch, it's a computable fact about the training
    data. But the theoretically optimal value for balancing the LOSS
    isn't always the empirically best value for the metric we actually
    care about (PR-AUC) — slightly over- or under-weighting the minority
    class can trade precision against recall in ways that move PR-AUC in
    either direction. So: center the search on the true ratio (where we
    have strong prior reason to expect the optimum sits) and let Optuna
    explore a modest band around it (`tightness`, e.g. +/-30%) rather
    than search blindly from 1 to 1000, which would waste trials on
    values with no principled justification.
    """
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    true_ratio = max(n_neg / max(n_pos, 1), 1.0)
    low = true_ratio * (1 - tightness)
    high = true_ratio * (1 + tightness)
    logger.info(
        "Class imbalance ratio = %.2f (%d legit / %d fraud) -> scale_pos_weight search "
        "range [%.2f, %.2f]",
        true_ratio, n_neg, n_pos, low, high,
    )
    return low, high


def make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    tscv: TimeSeriesSplit,
    spw_low: float,
    spw_high: float,
    n_estimators: int,
    random_state: int,
):
    """
    Returns an Optuna objective closure over the fixed CV splitter and
    data, so `study.optimize()` only needs to pass in `trial`.

    WHY n_estimators IS FIXED (NOT TUNED) AND THERE'S NO EARLY STOPPING
    INSIDE THIS LOOP: early-stopping a fold's model using that SAME
    fold's validation set, and then reporting that fold's score, subtly
    inflates the reported metric — you'd be picking the best iteration
    USING the exact data you then score on. The regularization
    hyperparameters actually being searched here (num_leaves, max_depth,
    min_child_samples, subsample, colsample_bytree, learning_rate) are
    themselves the mechanism for controlling overfitting in this search;
    a moderately large, fixed `n_estimators` lets the search find the
    regularization strength that generalizes, rather than letting early
    stopping paper over an under-regularized configuration. If you want
    n_estimators tuned too, add `trial.suggest_int("n_estimators", ...)`
    to the params dict below — kept out here to keep the search space
    focused on exactly the parameters the requirements specify.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # LightGBM requires bagging_freq > 0 for `subsample` (bagging
            # fraction) to actually take effect — without it, `subsample`
            # is silently ignored, which would make part of the search
            # space a no-op.
            "subsample_freq": 1,
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", spw_low, spw_high),
        }

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)

            val_probs = model.predict_proba(X_val)[:, 1]
            pr_auc = average_precision_score(y_val, val_probs)
            fold_scores.append(pr_auc)

            # Pruning: report this fold's score and let Optuna decide
            # whether this trial is even worth finishing. With 50 trials x
            # 3 folds = up to 150 model fits, a trial whose first fold
            # already scores far below the running median is very
            # unlikely to recover — pruning it after fold 1 or 2 instead
            # of always running all 3 folds meaningfully speeds up the
            # search without biasing which hyperparameters get selected.
            trial.report(pr_auc, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def run_study(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    n_trials: int,
    n_estimators: int,
    random_state: int,
) -> optuna.Study:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    spw_low, spw_high = compute_scale_pos_weight_bounds(y)

    objective = make_objective(X, y, tscv, spw_low, spw_high, n_estimators, random_state)

    study = optuna.create_study(
        direction="maximize",  # maximizing PR-AUC (average precision)
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    logger.info("Starting Optuna search: %d trials, %d-split TimeSeriesSplit CV.", n_trials, n_splits)
    for trial_num in range(n_trials):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        best_so_far = study.best_value
        logger.info(
            "Trial %d/%d complete. This trial: %.4f | Best so far: %.4f",
            trial_num + 1, n_trials,
            study.trials[-1].value if study.trials[-1].value is not None else float("nan"),
            best_so_far,
        )

    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for the fraud LightGBM model.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=300,
                         help="Fixed boosting rounds per trial (see make_objective's docstring "
                              "for why this isn't tuned or early-stopped).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=str, default="models/best_hyperparameters.json",
                         help="Where to save the best params as JSON, in addition to printing them.")
    args = parser.parse_args()

    config = load_config(args.config)
    label_col = config["columns"]["label_col"]

    df = load_and_sort_data(config)
    X = df.drop(columns=[label_col])
    y = df[label_col]

    study = run_study(
        X, y,
        n_splits=args.n_splits,
        n_trials=args.n_trials,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS FOUND")
    print("=" * 70)
    print(f"Best PR-AUC (mean across {args.n_splits} time-series folds): {study.best_value:.4f}")
    print("\nBest params (copy into train.py's TrainConfig.lgb_params):")
    print(json.dumps(study.best_params, indent=2))
    print("=" * 70)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "best_pr_auc": study.best_value,
                "best_params": study.best_params,
                "n_trials": args.n_trials,
                "n_splits": args.n_splits,
                "n_estimators": args.n_estimators,
            },
            f, indent=2,
        )
    logger.info("Also saved best params to %s", output_path)


if __name__ == "__main__":
    main()