"""
src/data_preprocessing.py

Cleans and feature-engineers the Sparkov "Credit Card Transactions Fraud
Detection" dataset (fraudTrain.csv / fraudTest.csv) into model-ready
parquet files — now including per-customer time-series velocity features
(rolling transaction counts, rolling spend, time-since-last-transaction,
and a spend-anomaly ratio), in addition to the Haversine distance, age,
and frequency-encoding logic from the previous version.

WHY THIS REWRITE PROCESSES TRAIN AND TEST AS ONE COMBINED STREAM FOR
VELOCITY FEATURES (BUT STILL FITS THE ENCODER TRAIN-ONLY)
------------------------------------------------------------------------
fraudTrain.csv and fraudTest.csv are not two independent datasets — they
are one continuous per-cardholder transaction history, sliced at a point
in time. A transaction in the first hour of the test period has real
transaction history in the last hours of the train period, and a live
system scoring that transaction would see that history. If velocity
features were computed on each file separately, every test-period
transaction near the boundary would look artificially "quiet" (low rolling
counts/sums) simply because its recent history got cut off by our file
split — a systematic, avoidable distortion, not a subtle one.

This is NOT the same thing as label leakage. Velocity features are pure
counts and sums of transaction amounts; they never touch `is_fraud`, so
letting a test-period row's rolling window reach back into train-period
rows does not leak any label information forward in time — it just
correctly reconstructs the transaction history a live system would
actually have. The categorical encoder is a completely different kind of
statistic (it's fit ON the label for target encoding, or reflects category
frequency that could shift over time for frequency encoding) — that one
must stay fit on train rows only, and still is, further down this file.

WHY THE VELOCITY-FEATURE IMPLEMENTATION LOOKS THE WAY IT DOES (READ THIS
BEFORE MODIFYING IT)
------------------------------------------------------------------------
`df.groupby(cc_num).rolling(window, on=timestamp_col).count()` is the
"obvious" pandas API for this, and it silently produces WRONG results
here: the returned Series is ordered by GROUP first (every row for
customer A, then every row for customer B, ...), not by the original
chronological row order. If you assign that result back with `.values`,
every row's velocity features get misaligned to the wrong transaction the
moment two customers' transactions interleave in time — which, in a
1M+-row multi-customer dataset, is virtually guaranteed. This was
verified directly (not assumed) while building this file. The fix used
below — `groupby(...).apply(...)` on a function that leaves each group's
original row index untouched — lets pandas realign the result back onto
the dataframe by INDEX LABEL rather than position, which is correct
regardless of how groupby reorders internally.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.feature_encoding import CategoricalEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_config(config_path: str | Path) -> dict:
    """Load the YAML config. Centralizing this means every script (this one,
    train.py, tune.py via src.config_utils, inference.py) parses column
    names / paths identically — no risk of one script's hardcoded string
    drifting from another's."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


# --------------------------------------------------------------------------- #
# Feature engineering: time-series / velocity (NEW)
# --------------------------------------------------------------------------- #
def engineer_velocity_features(df: pd.DataFrame, cols: dict, velocity_cfg: dict) -> pd.DataFrame:
    """
    Adds, per row, features describing THIS card's recent activity leading
    up to (but not including) this transaction:
      - seconds_since_last_txn: time gap since this card's previous transaction.
      - txn_count_1h / txn_count_24h: how many transactions this card made
        in the preceding 1h / 24h.
      - amt_sum_24h: how much this card spent in the preceding 24h.
      - amt_vs_24h_avg_ratio: this transaction's amount relative to this
        card's recent (24h) average spend — a spike in this ratio is one of
        the more reliable classic fraud signals (a sudden, out-of-pattern
        transaction size).

    WHY THE ROLLING WINDOWS EXCLUDE THE CURRENT TRANSACTION
    (`closed="left"`, meaning the window is [t - window, t), never
    including t itself): two independent reasons converge on the same
    answer here. First, leakage-avoidance discipline used throughout this
    project — a feature describing "recent behavior" should describe
    behavior strictly BEFORE the event being scored. Second, and more
    concretely: `amt_vs_24h_avg_ratio` divides the current amount by the
    24h rolling average. If that average included the current
    transaction, a single unusually large transaction would inflate its
    OWN baseline and shrink its own ratio — diluting the exact signal this
    feature exists to capture, right when it matters most.
    """
    cc_num_col = cols["cc_num_col"]
    timestamp_col = cols["timestamp_col"]
    amount_col = cols["amount_col"]

    short_window = velocity_cfg.get("short_window", "1h")
    long_window = velocity_cfg.get("long_window", "24h")
    no_history_seconds = velocity_cfg.get("no_history_time_delta_seconds", 999999)
    ratio_epsilon = velocity_cfg.get("ratio_epsilon", 1e-6)
    no_history_ratio_fill = velocity_cfg.get("no_history_ratio_fill", 1.0)

    # Defensive re-sort: this function's correctness depends entirely on
    # row order, so it re-asserts that order itself rather than trusting a
    # caller's promise — cheap on an already-sorted frame, and prevents a
    # very hard-to-diagnose bug if this is ever called from a different
    # context than run_pipeline().
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    def _per_customer_group(g: pd.DataFrame) -> pd.DataFrame:
        # `g` keeps its ORIGINAL row index from the outer dataframe
        # throughout this function — we never call .set_index() or
        # otherwise swap it for the timestamp. `on=timestamp_col` uses the
        # timestamps purely to size the rolling windows; the result stays
        # indexed the same way `g` is, which is what makes the
        # index-label-based reassembly below (`df.join(...)`) correct.
        short_roll = g.rolling(window=short_window, on=timestamp_col, closed="left")
        long_roll = g.rolling(window=long_window, on=timestamp_col, closed="left")
        return pd.DataFrame({
            "txn_count_1h": short_roll[amount_col].count(),
            "txn_count_24h": long_roll[amount_col].count(),
            "amt_sum_24h": long_roll[amount_col].sum(),
            "amt_mean_24h": long_roll[amount_col].mean(),
            "seconds_since_last_txn": g[timestamp_col].diff().dt.total_seconds(),
        })

    # group_keys=False: don't add cc_num as an extra index level on the
    # result. include_groups=False: don't pass the cc_num column itself
    # into _per_customer_group (it isn't needed there, and recent pandas
    # versions warn/error on this by default anyway).
    velocity_feats = df.groupby(cc_num_col, group_keys=False).apply(
        _per_customer_group, include_groups=False
    )
    df = df.join(velocity_feats)

    # --- Fill "no prior history" cases --------------------------------------
    # A card's first-ever transaction (or first in a very long time) has an
    # EMPTY rolling window, which pandas represents as NaN, not 0. These
    # rows genuinely have zero prior activity — fill explicitly so that
    # semantics are unambiguous rather than leaving NaN to be interpreted
    # implicitly downstream.
    df["txn_count_1h"] = df["txn_count_1h"].fillna(0)
    df["txn_count_24h"] = df["txn_count_24h"].fillna(0)
    df["amt_sum_24h"] = df["amt_sum_24h"].fillna(0)
    df["seconds_since_last_txn"] = df["seconds_since_last_txn"].fillna(no_history_seconds)

    # --- Ratio feature: current amt vs 24h rolling average ------------------
    # amt_mean_24h is NaN in exactly the same "no history" cases — a
    # rolling mean over zero points is undefined. Filling the RATIO with a
    # neutral 1.0 here (rather than leaving it NaN) keeps the feature
    # well-defined regardless of model type. The model can still tell a
    # genuine "ratio=1.0, plenty of history" row apart from a "ratio=1.0,
    # no history" row via txn_count_24h (0 in the latter case) — LightGBM
    # is perfectly capable of learning that interaction from two features.
    df["amt_vs_24h_avg_ratio"] = np.where(
        df["amt_mean_24h"].isna(),
        no_history_ratio_fill,
        df[amount_col] / (df["amt_mean_24h"] + ratio_epsilon),
    )
    df = df.drop(columns=["amt_mean_24h"])  # only needed as an intermediate for the ratio

    logger.info(
        "Engineered velocity features (seconds_since_last_txn, txn_count_1h/24h, "
        "amt_sum_24h, amt_vs_24h_avg_ratio) grouped by '%s' over %d rows.",
        cc_num_col, len(df),
    )
    return df


# --------------------------------------------------------------------------- #
# Feature engineering: temporal
# --------------------------------------------------------------------------- #
def engineer_temporal_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Extract hour / day-of-week / month from the transaction timestamp.

    WHY cyclical (sin/cos) hour AND raw hour: trees split on raw integer
    hour perfectly well, but we keep both because (a) raw hour is directly
    interpretable in SHAP output — "hour=3am contributed +0.4" is legible
    to a fraud analyst in a way "hour_sin=-0.97" is not, and (b) sin/cos
    still helps LightGBM avoid an artificial split boundary at the 23->0
    wraparound that a raw integer would otherwise encode as "maximally
    different".
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df["txn_hour"] = df[timestamp_col].dt.hour
    df["txn_day_of_week"] = df[timestamp_col].dt.dayofweek  # 0=Mon .. 6=Sun
    df["txn_month"] = df[timestamp_col].dt.month
    df["txn_is_weekend"] = (df["txn_day_of_week"] >= 5).astype(int)

    df["txn_hour_sin"] = np.sin(2 * np.pi * df["txn_hour"] / 24)
    df["txn_hour_cos"] = np.cos(2 * np.pi * df["txn_hour"] / 24)

    logger.info("Engineered temporal features from '%s'.", timestamp_col)
    return df


# --------------------------------------------------------------------------- #
# Feature engineering: spatial (Haversine distance) — retained, unchanged
# --------------------------------------------------------------------------- #
def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Great-circle distance in kilometers between two lat/long points.

    WHY Haversine over plain Euclidean distance on raw lat/long: degrees of
    longitude represent a shrinking physical distance as you move away from
    the equator. Naive Euclidean distance on raw degrees would systematically
    distort distance for customers at different latitudes — Haversine
    accounts for the Earth's curvature and gives a physically meaningful
    "how far is this purchase from home" feature, one of the strongest
    classic fraud signals (card-present-far-from-home).

    Vectorized with numpy, not a python loop / .apply(), for the same
    reason the velocity features above avoid a naive per-row loop: this
    dataset has ~1.85M rows.
    """
    R = 6371.0  # Earth's radius in km
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # clip guards float error
                                                     # pushing `a` fractionally
                                                     # above 1 -> NaN in sqrt
    return R * c


def engineer_spatial_features(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Adds `distance_from_home_km`: distance between the customer's home
    (lat, long) and the merchant's location (merch_lat, merch_long)."""
    df = df.copy()
    df["distance_from_home_km"] = haversine_distance(
        df[cols["customer_lat_col"]].values,
        df[cols["customer_long_col"]].values,
        df[cols["merchant_lat_col"]].values,
        df[cols["merchant_long_col"]].values,
    )
    logger.info("Engineered 'distance_from_home_km' via Haversine formula.")
    return df


# --------------------------------------------------------------------------- #
# Feature engineering: demographic — retained, unchanged
# --------------------------------------------------------------------------- #
def engineer_demographic_features(
    df: pd.DataFrame, dob_col: str, timestamp_col: str
) -> pd.DataFrame:
    """
    Converts date-of-birth into a numeric `age` feature, computed relative
    to the TRANSACTION time (not "today"), since this dataset's
    transactions span 2019-2020 and age-as-of-today would silently corrupt
    the feature with years of drift.
    """
    df = df.copy()
    dob = pd.to_datetime(df[dob_col])
    txn_time = pd.to_datetime(df[timestamp_col])
    df["age"] = ((txn_time - dob).dt.days / 365.25).astype(float)
    logger.info("Engineered 'age' from '%s' relative to '%s'.", dob_col, timestamp_col)
    return df


# --------------------------------------------------------------------------- #
# Feature dropping
# --------------------------------------------------------------------------- #
def drop_unused_columns(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    """
    Drops high-cardinality free-text / identifier columns. `cc_num` is in
    this list — safe to drop ONLY because engineer_velocity_features()
    above has already consumed it to build the per-card behavioral
    features; run_pipeline() calls these functions in that order
    deliberately, per the requirement that grouped features be computed
    BEFORE cc_num is dropped.
    """
    present = [c for c in drop_columns if c in df.columns]
    missing = set(drop_columns) - set(present)
    if missing:
        logger.warning("Some configured drop_columns not found in df, skipping: %s", missing)
    df = df.drop(columns=present)
    logger.info("Dropped %d columns: %s", len(present), present)
    return df


# CategoricalEncoder now lives in feature_encoding.py (imported above) —
# see that file's docstring for why it isn't defined in this script.


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_raw_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = config["paths"]["train_raw"]
    test_path = config["paths"]["test_raw"]
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logger.info("Loaded raw train=%s rows, test=%s rows.", len(train_df), len(test_df))
    return train_df, test_df


def engineer_stateless_features(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Runs the feature-engineering steps that are pure, per-row functions
    of already-known values (temporal, spatial, demographic) — these never
    depend on the label and never depend on cross-row grouping, so unlike
    the velocity features and the categorical encoder, it's safe to run
    them on train and test data identically, in any order."""
    df = engineer_temporal_features(df, cols["timestamp_col"])
    df = engineer_spatial_features(df, cols)
    df = engineer_demographic_features(df, cols["dob_col"], cols["timestamp_col"])
    return df


def run_pipeline(config: dict) -> None:
    cols = config["columns"]
    prep_cfg = config["preprocessing"]
    velocity_cfg = config.get("velocity_features", {})

    train_df, test_df = load_raw_data(config)

    # Parse the timestamp to datetime up front — every step below
    # (velocity features, temporal features, age) depends on it, and
    # requirement #1 demands sorting happen before any groupby touches
    # the data at all.
    train_df[cols["timestamp_col"]] = pd.to_datetime(train_df[cols["timestamp_col"]])
    test_df[cols["timestamp_col"]] = pd.to_datetime(test_df[cols["timestamp_col"]])

    # --- Combine into one continuous chronological stream for velocity
    # feature computation only. See this file's module docstring for why:
    # in short, fraudTrain/fraudTest are one continuous per-card history
    # artificially cut at a point in time, and computing velocity features
    # on each file separately would understate every test-period card's
    # recent activity near that cut point. `_split_origin` lets us put
    # train and test right back where they came from afterward.
    train_df["_split_origin"] = "train"
    test_df["_split_origin"] = "test"
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # --- Requirement #1: sort chronologically BEFORE any groupby -----------
    combined = combined.sort_values(cols["timestamp_col"]).reset_index(drop=True)
    logger.info("Sorted %d combined rows chronologically by '%s'.", len(combined), cols["timestamp_col"])

    # --- Velocity / time-delta features (grouped by cc_num) -----------------
    combined = engineer_velocity_features(combined, cols, velocity_cfg)

    # --- Remaining, stateless feature engineering ---------------------------
    combined = engineer_stateless_features(combined, cols)

    # --- Drop high-cardinality / now-unneeded identifier columns -----------
    # cc_num is dropped here via drop_columns (config-driven) — AFTER
    # engineer_velocity_features has already consumed it, per requirement #5.
    combined = drop_unused_columns(combined, config["drop_columns"])

    # Raw timestamp and dob are dropped explicitly (not via drop_columns)
    # once every feature that needed them (velocity, temporal, age) has
    # been derived — keeping this as its own explicit step, rather than
    # folding it into drop_columns, makes the "why" (it's a derived-from,
    # not a never-useful, column) visible at the call site.
    for extra_col in (cols["timestamp_col"], cols["dob_col"]):
        combined = combined.drop(columns=[extra_col], errors="ignore")

    # --- Split back into train/test using the origin marker -----------------
    train_df = (
        combined[combined["_split_origin"] == "train"]
        .drop(columns=["_split_origin"])
        .reset_index(drop=True)
    )
    test_df = (
        combined[combined["_split_origin"] == "test"]
        .drop(columns=["_split_origin"])
        .reset_index(drop=True)
    )

    # --- Categorical encoding: FIT ON TRAIN, TRANSFORM BOTH -----------------
    # This is the one piece of statistics in this file that must NOT be
    # computed on the combined stream — unlike velocity features (pure
    # counts/sums, never touch the label), frequency/target encoding
    # reflects category statistics (and, for target encoding, the label
    # itself) that must only ever be learned from train rows. See
    # feature_encoding.py for the full leakage-safety discussion.
    encoder = CategoricalEncoder(
        method=prep_cfg["encoding_method"],
        smoothing=prep_cfg.get("target_encoding_smoothing", 20.0),
    )
    encoder.fit(
        train_df,
        categorical_columns=config["categorical_columns"],
        label_col=cols["label_col"],
    )
    train_df = encoder.transform(train_df)
    test_df = encoder.transform(test_df)  # reuses train-fitted mapping — no refit

    # Persist the fitted encoder so the FastAPI inference layer can apply
    # the exact same train-time mapping to a single live transaction.
    encoder_path = Path(config["paths"]["encoder_path"])
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, encoder_path)
    logger.info("Saved fitted encoder to %s", encoder_path)

    # --- Persist cleaned datasets -------------------------------------------
    out_train = Path(config["paths"]["train_processed"])
    out_test = Path(config["paths"]["test_processed"])
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out_train, index=False)
    test_df.to_parquet(out_test, index=False)

    logger.info(
        "Saved cleaned data -> train: %s (%d rows, %d cols), test: %s (%d rows, %d cols)",
        out_train, len(train_df), train_df.shape[1],
        out_test, len(test_df), test_df.shape[1],
    )
    logger.info(
        "Train fraud rate: %.4f%% | Test fraud rate: %.4f%%",
        100 * train_df[cols["label_col"]].mean(),
        100 * test_df[cols["label_col"]].mean(),
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess the Sparkov fraud dataset.")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()