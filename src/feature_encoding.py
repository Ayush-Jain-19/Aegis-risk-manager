"""
src/feature_encoding.py

WHY THIS CLASS LIVES IN ITS OWN FILE
---------------------------------------
`CategoricalEncoder` instances are serialized with joblib in
data_preprocessing.py and deserialized in api/main.py — two different
Python processes, run two different ways (one as a direct script, one as
an imported ASGI app). Pickle records a class by its "module path" — e.g.
"data_preprocessing.CategoricalEncoder" — and Python sets a script's
`__name__` (and therefore every class defined in it) to `"__main__"`
whenever that script is the ENTRY POINT of a process, regardless of
whether it's invoked as `python file.py` or `python -m package.module`.

Concretely: if `CategoricalEncoder` were still defined inside
data_preprocessing.py, running `python src/data_preprocessing.py` would
pickle it as `__main__.CategoricalEncoder`. The API process's `__main__` is
a completely different module (uvicorn's entry point, or this API's own
main.py) — so `joblib.load()` would raise `AttributeError: Can't get
attribute 'CategoricalEncoder' on <module '__main__'>`, exactly the
failure this refactor exists to prevent. By keeping the class in a module
that is NEVER the entry point of any process, its pickled module path is
always the stable `"feature_encoding"`, regardless of how
data_preprocessing.py or the API happen to be launched.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CategoricalEncoder:
    """
    Leakage-safe frequency / target encoder for high-cardinality categoricals
    (merchant, category, job, city, state, gender).

    WHY NOT ONE-HOT: `merchant` alone has ~700 unique values in the Sparkov
    dataset and `job` has ~500. One-hot encoding both would add ~1,200
    mostly-zero columns, bloating memory, slowing LightGBM's histogram
    building, and diluting each individual split's signal. A single dense
    numeric column per categorical (its frequency, or its smoothed
    fraud-rate) captures nearly all of the useful signal a tree needs to
    split on, at 1/1200th of the width.

    WHY fit()/transform() ARE SPLIT: callers are forced to fit on train and
    transform on both train and test (and, at inference time, on a single
    live transaction) using the SAME learned mapping — never refit on data
    the model shouldn't be allowed to peek at.
    """

    def __init__(self, method: str = "frequency", smoothing: float = 20.0):
        if method not in ("frequency", "target"):
            raise ValueError(f"Unknown encoding method: {method}")
        self.method = method
        self.smoothing = smoothing
        self.mappings_: dict[str, pd.Series] = {}
        self.global_fallback_: dict[str, float] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, categorical_columns: list[str], label_col: Optional[str] = None):
        if self.method == "target" and label_col is None:
            raise ValueError("label_col is required for target encoding.")

        for col in categorical_columns:
            if self.method == "frequency":
                # Normalized value counts -> each category's share of all
                # TRAIN rows. Purely a function of category frequency, never
                # touches the label, so it cannot leak fraud-rate information
                # even if accidentally computed on a mixed set.
                mapping = df[col].value_counts(normalize=True)
                self.mappings_[col] = mapping
                self.global_fallback_[col] = 0.0  # unseen category -> "never seen before"

            else:  # target encoding, smoothed toward the global fraud rate
                global_rate = df[label_col].mean()
                agg = df.groupby(col)[label_col].agg(["sum", "count"])
                smoothed = (agg["sum"] + self.smoothing * global_rate) / (
                    agg["count"] + self.smoothing
                )
                self.mappings_[col] = smoothed
                self.global_fallback_[col] = global_rate  # unseen category -> global prior

        self._is_fitted = True
        logger.info(
            "Fit %s-encoding on columns %s (train rows only).",
            self.method, categorical_columns,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "CategoricalEncoder.transform() called before fit(). This "
                "guard exists to stop a caller from accidentally fitting "
                "fresh statistics on the test set instead of reusing the "
                "train-fitted mapping."
            )
        df = df.copy()
        for col, mapping in self.mappings_.items():
            new_col = f"{col}_{self.method}_enc"
            df[new_col] = (
                df[col].map(mapping).fillna(self.global_fallback_[col]).astype(float)
            )
            df = df.drop(columns=[col])  # replace raw string column with its encoding
        return df

    def fit_transform(self, df, categorical_columns, label_col=None):
        return self.fit(df, categorical_columns, label_col).transform(df)