"""
Self-contained deployment bundle.

`DeployablePipeline` wraps a fitted sklearn/imblearn pipeline together with the
raw-DataFrame preprocessing that `loader.py` performs at training time
(column-name sanitisation, jalali/gregorian date expansion, categorical
mapping) and a tuned classification threshold. The bundle is serialised with
`save_deployable`, which registers every framework module with cloudpickle's
by-value protocol so the resulting `.pkl` file carries its own class
definitions. Deployment environments therefore need only the standard ML
stack (numpy, pandas, scikit-learn, imblearn, the booster libraries the
winning classifier depends on, cloudpickle, jdatetime) plus a short driver
script — no other file from this repo.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import jdatetime  # only required when jalali dates are present
except Exception:  # pragma: no cover
    jdatetime = None


# ---------------------------------------------------------------------------
# Helpers inlined from loader.py so the bundle does not depend on that file
# at deployment time. Kept behaviourally identical.
# ---------------------------------------------------------------------------
def _sanitize_column_names(df: pd.DataFrame, y_column: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        col if col == y_column else re.sub(r'[^\w]', '_', str(col))
        for col in df.columns
    ]
    return df


def _identify_date_columns(df: pd.DataFrame) -> List[str]:
    date_cols = []
    for col in df.select_dtypes(include=['object', 'string']).columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        try:
            pd.to_datetime(non_null, errors='raise')
            date_cols.append(col)
        except Exception:
            continue
    return date_cols


def _detect_calendar_type(date_str: Any) -> str:
    try:
        s = str(date_str).strip()
        parts = s.split('/') if '/' in s else (s.split('-') if '-' in s else [])
        if len(parts) != 3:
            return 'unknown'
        year = int(parts[0])
        if 1200 <= year <= 1500:
            return 'jalali'
        if 1800 <= year <= 2200:
            return 'gregorian'
        return 'unknown'
    except Exception:
        return 'unknown'


def _convert_jalali_to_datetime(series: pd.Series) -> pd.Series:
    if jdatetime is None:
        raise ImportError(
            'jdatetime is required to parse jalali dates at inference time.'
        )
    out = []
    for val in series:
        try:
            parts = str(val).strip().split('/')
            if len(parts) == 3:
                y, m, d = map(int, parts)
                out.append(pd.Timestamp(jdatetime.date(y, m, d).togregorian()))
            else:
                out.append(pd.NaT)
        except Exception:
            out.append(pd.NaT)
    return pd.Series(out, index=series.index)


def _expand_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as loader.process_date_columns but operating
    on a copy so it is safe to call on raw incoming DataFrames."""
    df = df.copy()
    date_cols = _identify_date_columns(df)
    if not date_cols:
        return df
    sample = df[date_cols[0]].dropna()
    cal = _detect_calendar_type(sample.iloc[0]) if len(sample) else 'gregorian'
    if cal == 'jalali':
        for c in date_cols:
            df[c] = _convert_jalali_to_datetime(df[c])
    else:
        for c in date_cols:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    for c in date_cols:
        df[f'{c}_year']        = df[c].dt.year.astype(float)
        df[f'{c}_month']       = df[c].dt.month.astype(float)
        df[f'{c}_day']         = df[c].dt.day.astype(float)
        df[f'{c}_day_of_week'] = df[c].dt.dayofweek.astype(float)
        df[f'{c}_hour']        = df[c].dt.hour.astype(float)
        df[f'{c}_day_of_year'] = df[c].dt.dayofyear.astype(float)
        df[f'{c}_is_weekend']  = (df[c].dt.weekday >= 5).astype(float)
    return df.drop(columns=date_cols)


# ---------------------------------------------------------------------------
# RawPreprocessor — sklearn-compatible transformer that turns a raw incoming
# DataFrame into the numeric matrix the trained pipeline was fitted on.
# ---------------------------------------------------------------------------
class RawPreprocessor(BaseEstimator, TransformerMixin):
    """Apply train-time raw-data preprocessing to unseen DataFrames.

    Steps (mirroring loader.py + the notebook):
      1. drop the target column if present;
      2. sanitise column names (non-word chars -> '_');
      3. expand date columns into year/month/day/... features;
      4. apply the learned categorical mappings — unseen categories become
         NaN so the downstream imputer can absorb them;
      5. reindex to the exact training feature order (missing columns are
         created as NaN, extra columns are dropped);
      6. cast to float32 to match the train-time dtype.
    """

    def __init__(
        self,
        y_column: str,
        all_mappings: Dict[str, Dict[Any, int]],
        feature_columns: List[str],
    ):
        self.y_column = y_column
        self.all_mappings = all_mappings
        self.feature_columns = list(feature_columns)

    def fit(self, X, y=None):  # noqa: D401
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)
        df = X.copy()
        if self.y_column in df.columns:
            df = df.drop(columns=[self.y_column])
        df = _sanitize_column_names(df, self.y_column)
        df = _expand_date_columns(df)
        for col, mapping in self.all_mappings.items():
            if col == self.y_column or col not in df.columns:
                continue
            df[col] = df[col].map(lambda v: mapping.get(str(v), np.nan))
        df = df.reindex(columns=self.feature_columns)
        return df.astype('float32')


# ---------------------------------------------------------------------------
# DeployablePipeline — bundle of (raw preprocessor, sklearn pipeline,
# operating-point threshold, y-label inverse mapping). Exposed methods are
# the only API a deployment script needs.
# ---------------------------------------------------------------------------
class DeployablePipeline:
    def __init__(
        self,
        raw_preprocessor: RawPreprocessor,
        sklearn_pipeline,
        threshold: float = 0.5,
        y_inverse_mapping: Optional[Dict[int, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.raw_preprocessor = raw_preprocessor
        self.sklearn_pipeline = sklearn_pipeline
        self.threshold = float(threshold)
        self.y_inverse_mapping = y_inverse_mapping or {}
        self.metadata = metadata or {}

    def _prepare(self, raw_df):
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError('DeployablePipeline expects a pandas DataFrame.')
        return self.raw_preprocessor.transform(raw_df)

    def predict_proba(self, raw_df) -> np.ndarray:
        X = self._prepare(raw_df)
        return self.sklearn_pipeline.predict_proba(X)

    def predict(self, raw_df) -> np.ndarray:
        """Binary 0/1 predictions using the embedded threshold."""
        proba = self.predict_proba(raw_df)
        if proba.shape[1] == 2:
            return (proba[:, 1] >= self.threshold).astype(int)
        return np.argmax(proba, axis=1)

    def predict_labels(self, raw_df) -> List[Any]:
        """Return predictions decoded back to the original class labels."""
        preds = self.predict(raw_df)
        if not self.y_inverse_mapping:
            return preds.tolist()
        return [self.y_inverse_mapping.get(int(p), int(p)) for p in preds]

    def describe(self) -> str:
        steps = []
        if hasattr(self.sklearn_pipeline, 'named_steps'):
            for name, step in self.sklearn_pipeline.named_steps.items():
                steps.append(f'  {name:11s} -> {type(step).__name__}')
        return (
            f'DeployablePipeline(threshold={self.threshold:.4f}, '
            f'features={len(self.raw_preprocessor.feature_columns)})\n'
            + '\n'.join(steps)
        )


# ---------------------------------------------------------------------------
# Serialisation — cloudpickle by-value so deployment does not need any
# framework .py file.
# ---------------------------------------------------------------------------
_FRAMEWORK_MODULES = (
    'deployable',
    'clean_data',
    'tame_outlier',
    'normalization',
    'feature_selection',
    'balance',
    'model_training',
)


def _register_by_value():
    import importlib
    import cloudpickle
    for mod_name in _FRAMEWORK_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            cloudpickle.register_pickle_by_value(mod)
        except Exception:
            # A module that the winning pipeline doesn't reference can be
            # skipped — the goal is to embed only what's needed.
            continue


def save_deployable(bundle: DeployablePipeline, out_path: str) -> str:
    """Serialise `bundle` to `out_path` using cloudpickle by-value.

    The resulting file can be unpickled in any environment that has the
    standard ML stack and cloudpickle installed — no framework `.py` files
    are required.
    """
    import cloudpickle
    _register_by_value()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        cloudpickle.dump(bundle, f)
    return out_path


def load_deployable(path: str) -> DeployablePipeline:
    import cloudpickle
    with open(path, 'rb') as f:
        return cloudpickle.load(f)
