import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

_MIN_UNIQUE_FOR_NUMERIC = 10


# ---------- helpers ---------------------------------------------------------

def _coerce_dataframe(X, feature_names=None):
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    cols = (
        list(feature_names)
        if feature_names is not None and len(feature_names) == arr.shape[1]
        else [f'c{i}' for i in range(arr.shape[1])]
    )
    return pd.DataFrame(arr, columns=cols, index=pd.RangeIndex(arr.shape[0]))


def _select_numeric_cols(X):
    return [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c])
        and X[c].nunique(dropna=True) > _MIN_UNIQUE_FOR_NUMERIC
    ]


def _mad_scale(X_num):
    med = X_num.median()
    mad = X_num.sub(med).abs().median() * 1.4826
    return mad.replace(0.0, 1.0)


def _normalize_scores_unit(scores):
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(scores)), index=scores.index)
    return (scores - lo) / (hi - lo)


# ---------- Stage 1 — detectors --------------------------------------------

class _StatisticalDetector:
    def __init__(self, z_threshold):
        self.z_threshold = z_threshold

    def fit(self, X_num, stats):
        return self

    def score(self, X_num, stats):
        z = (X_num - stats['medians_']) / stats['mads_']
        max_z = z.abs().max(axis=1)
        return pd.DataFrame(
            {'score': max_z.to_numpy(),
             'flag': (max_z > self.z_threshold).astype(int).to_numpy()},
            index=X_num.index,
        )


class _ClusterDetector:
    def __init__(self, n_clusters, contamination, random_state):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.model_ = None

    def fit(self, X_num, stats):
        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init='auto',
        )
        self.model_.fit(X_num.values)
        d = self.model_.transform(X_num.values).min(axis=1)
        labels = pd.Series(self.model_.labels_, index=X_num.index)
        stats['cluster_centroids_'] = pd.DataFrame(
            self.model_.cluster_centers_,
            columns=X_num.columns,
            index=pd.RangeIndex(self.n_clusters),
        )
        stats['cluster_medians_'] = (
            X_num.assign(_cluster=labels.values).groupby('_cluster').median()
        )
        stats['cluster_threshold_'] = float(
            np.percentile(d, 100 * (1 - self.contamination))
        )
        return self

    def score(self, X_num, stats):
        d = self.model_.transform(X_num.values).min(axis=1)
        labels = self.model_.predict(X_num.values)
        flag = (d > stats['cluster_threshold_']).astype(int)
        return pd.DataFrame(
            {'score': d, 'flag': flag, 'cluster': labels},
            index=X_num.index,
        )


class _IForestDetector:
    def __init__(self, contamination, random_state):
        self.contamination = contamination
        self.random_state = random_state
        self.model_ = None

    def fit(self, X_num, stats):
        self.model_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_num.values)
        return self

    def score(self, X_num, stats):
        raw = -self.model_.score_samples(X_num.values)
        flag = (self.model_.predict(X_num.values) == -1).astype(int)
        return pd.DataFrame({'score': raw, 'flag': flag}, index=X_num.index)


# ---------- Stage 2 — explanations -----------------------------------------

def _per_feature_deviation(X_num, stats):
    return (X_num - stats['medians_']) / stats['mads_']


def _dominant_features(deviation_df, flags, top_k=3):
    flagged = deviation_df.loc[flags == 1]
    if flagged.empty:
        return pd.DataFrame(
            columns=['row', 'rank', 'feature', 'signed_mad_z'],
            index=pd.RangeIndex(0),
        )
    records = []
    for row_id, row in flagged.iterrows():
        order = row.abs().sort_values(ascending=False).head(top_k).index
        for rank, feat in enumerate(order, 1):
            records.append((row_id, rank, feat, float(row[feat])))
    return pd.DataFrame(
        records,
        columns=['row', 'rank', 'feature', 'signed_mad_z'],
        index=pd.RangeIndex(len(records)),
    )


def _cluster_deviation_profile(deviation_df, cluster_labels, flags):
    abs_dev = deviation_df.abs()
    base = abs_dev.assign(_cluster=cluster_labels.values, _flag=flags.values)
    means = base.groupby('_cluster')[abs_dev.columns.tolist()].mean()
    n_members = base.groupby('_cluster').size().rename('n_members')
    flag_rate = base.groupby('_cluster')['_flag'].mean().rename('flag_rate')
    out = means.join(n_members).join(flag_rate)
    out.index.name = 'cluster'
    return out


# ---------- Stage 3 — remediation ------------------------------------------

def _remediate_winsorize(X_num, stats, det_out, mad_threshold):
    dev = (X_num - stats['medians_']) / stats['mads_']
    anomalous = dev.abs() > mad_threshold
    lo, hi = stats['quantiles_lo_'], stats['quantiles_hi_']
    too_low = anomalous & X_num.lt(lo, axis=1)
    too_high = anomalous & X_num.gt(hi, axis=1)
    out = X_num.mask(too_low, lo, axis=1)
    out = out.mask(too_high, hi, axis=1)
    return out


def _remediate_percentile_clip(X_num, stats, det_out, mad_threshold):
    return X_num.clip(lower=stats['quantiles_lo_'],
                      upper=stats['quantiles_hi_'], axis=1)


def _remediate_mad_replace(X_num, stats, det_out, mad_threshold):
    dev = (X_num - stats['medians_']) / stats['mads_']
    mask = dev.abs() > mad_threshold
    return X_num.mask(mask, stats['medians_'], axis=1)


def _remediate_cluster_local_median(X_num, stats, det_out, mad_threshold):
    dev = (X_num - stats['medians_']) / stats['mads_']
    deviant = dev.abs() > mad_threshold
    labels = det_out['cluster']
    cluster_meds = stats['cluster_medians_']
    out = X_num.copy()
    for c_id, sub_idx in labels.groupby(labels).groups.items():
        if c_id not in cluster_meds.index:
            continue
        block = out.loc[sub_idx]
        sub_mask = deviant.loc[sub_idx]
        out.loc[sub_idx] = block.mask(sub_mask, cluster_meds.loc[c_id], axis=1)
    return out


def _remediate_shrink(X_num, stats, det_out, shrinkage_lambda):
    lam = float(np.clip(shrinkage_lambda, 0.0, 1.0))
    score_norm = _normalize_scores_unit(det_out['score']).to_numpy()[:, None]
    diff = X_num.to_numpy() - stats['medians_'].to_numpy()[None, :]
    return pd.DataFrame(
        X_num.to_numpy() - lam * score_norm * diff,
        index=X_num.index,
        columns=X_num.columns,
    )


# ---------- public transformer ---------------------------------------------

class OutlierTamer(BaseEstimator, TransformerMixin):
    """Detect → explain → remediate. Output shape, column order, and index match input.
    Explanations (when enabled) live on ``self.metadata_``; nothing is appended to X."""

    def __init__(self,
                 detector='iforest',
                 remediation='winsorize',
                 contamination=0.05,
                 z_threshold=3.0,
                 mad_threshold=3.5,
                 lower_quantile=0.01,
                 upper_quantile=0.99,
                 n_clusters=3,
                 shrinkage_lambda=0.5,
                 store_metadata=False,
                 random_state=42):
        self.detector = detector
        self.remediation = remediation
        self.contamination = contamination
        self.z_threshold = z_threshold
        self.mad_threshold = mad_threshold
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.n_clusters = n_clusters
        self.shrinkage_lambda = shrinkage_lambda
        self.store_metadata = store_metadata
        self.random_state = random_state

    # -- internals ----------------------------------------------------------
    def _build_detector(self):
        if self.detector == 'iforest':
            return _IForestDetector(self.contamination, self.random_state)
        if self.detector == 'cluster':
            return _ClusterDetector(self.n_clusters, self.contamination,
                                    self.random_state)
        if self.detector == 'statistical':
            return _StatisticalDetector(self.z_threshold)
        raise ValueError(f"unknown detector: {self.detector!r}")

    def _init_statistics(self, X_num):
        return {
            'medians_':      X_num.median(),
            'mads_':         _mad_scale(X_num),
            'quantiles_lo_': X_num.quantile(self.lower_quantile),
            'quantiles_hi_': X_num.quantile(self.upper_quantile),
        }

    def _build_aux_cluster_if_needed(self, X_num):
        if self.remediation != 'cluster_local_median' or self.detector == 'cluster':
            self._aux_cluster_ = None
            return
        aux = _ClusterDetector(self.n_clusters, self.contamination, self.random_state)
        aux.fit(X_num, self.statistics_)
        self._aux_cluster_ = aux

    # -- sklearn API --------------------------------------------------------
    def fit(self, X, y=None):
        X_df = _coerce_dataframe(X, getattr(self, 'feature_names_in_', None))
        self.feature_names_in_ = list(X_df.columns)
        self.numeric_cols_ = _select_numeric_cols(X_df)
        self.statistics_ = {}
        self.metadata_ = {}
        self._aux_cluster_ = None

        if not self.numeric_cols_:
            self.detector_ = None
            return self

        X_num = X_df[self.numeric_cols_].astype('float64')
        X_num = X_num.fillna(X_num.median())
        self.statistics_.update(self._init_statistics(X_num))

        self.detector_ = self._build_detector()
        self.detector_.fit(X_num, self.statistics_)
        self._build_aux_cluster_if_needed(X_num)

        if self.store_metadata:
            det_out = self.detector_.score(X_num, self.statistics_)
            deviation = _per_feature_deviation(X_num, self.statistics_)
            self.metadata_['per_feature_deviation_'] = deviation
            self.metadata_['detector_output_'] = det_out
            self.metadata_['dominant_features_'] = _dominant_features(
                deviation, det_out['flag']
            )
            if 'cluster' in det_out.columns:
                self.metadata_['cluster_deviation_profile_'] = _cluster_deviation_profile(
                    deviation, det_out['cluster'], det_out['flag']
                )
        return self

    def transform(self, X):
        X_df = _coerce_dataframe(X, self.feature_names_in_)
        out = X_df.copy()

        if not self.numeric_cols_ or self.detector_ is None:
            return out.reindex(columns=self.feature_names_in_, copy=False)

        X_num = out[self.numeric_cols_].astype('float64')
        X_num_filled = X_num.fillna(self.statistics_['medians_'])

        det_out = self.detector_.score(X_num_filled, self.statistics_)
        if self.remediation == 'cluster_local_median' and 'cluster' not in det_out.columns:
            aux_out = self._aux_cluster_.score(X_num_filled, self.statistics_)
            det_out = det_out.assign(cluster=aux_out['cluster'].to_numpy())

        rem = self.remediation
        if rem == 'none':
            repaired = X_num_filled
        elif rem == 'winsorize':
            repaired = _remediate_winsorize(X_num_filled, self.statistics_,
                                            det_out, self.mad_threshold)
        elif rem == 'percentile_clip':
            repaired = _remediate_percentile_clip(X_num_filled, self.statistics_,
                                                  det_out, self.mad_threshold)
        elif rem == 'mad_replace':
            repaired = _remediate_mad_replace(X_num_filled, self.statistics_,
                                              det_out, self.mad_threshold)
        elif rem == 'cluster_local_median':
            repaired = _remediate_cluster_local_median(X_num_filled, self.statistics_,
                                                       det_out, self.mad_threshold)
        elif rem == 'shrink':
            repaired = _remediate_shrink(X_num_filled, self.statistics_,
                                         det_out, self.shrinkage_lambda)
        else:
            raise ValueError(f"unknown remediation: {rem!r}")

        out.loc[:, self.numeric_cols_] = repaired.to_numpy()
        return out.reindex(columns=self.feature_names_in_, copy=False)
