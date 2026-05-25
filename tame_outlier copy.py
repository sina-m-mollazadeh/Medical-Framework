import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

class BaseOutlierTamer(BaseEstimator, TransformerMixin):
    """Base class providing shared utilities for medical-grade outlier handlers."""
    def _get_numeric_columns(self, X):
        return [
            col for col in X.columns 
            if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 10
        ]

    def _get_clean_numeric_data(self, X, numeric_cols):
        X_numeric = X[numeric_cols]
        # Impute missing values with column medians temporarily to safely fit/transform models
        fill_values = X_numeric.median()
        return X_numeric.fillna(fill_values)


class IsolationForestTamer(BaseOutlierTamer):
    """
    Article 2 (Samariya et al.): Isolation Forest Ensemble
    Evaluates multivariate feature interactions (Subspace / Outlying Aspect Mining)
    to identify complex anomalies without relying on biased variance metrics.
    """
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.numeric_cols_ = []
        self.model_instance_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.numeric_cols_ = self._get_numeric_columns(X_df)
        
        if not self.numeric_cols_:
            return self
            
        X_clean = self._get_clean_numeric_data(X_df, self.numeric_cols_)
        
        self.model_instance_ = IsolationForest(
            contamination=self.contamination, 
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_instance_.fit(X_clean)
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        if not self.numeric_cols_ or self.model_instance_ is None:
            return X_out
            
        X_clean = self._get_clean_numeric_data(X_out, self.numeric_cols_)
        
        # Continuous isolation scores (Lower = more anomalous)
        X_out['anomaly_score_iforest'] = self.model_instance_.score_samples(X_clean)
        
        # Binary flags (-1: anomaly, 1: normal) mapped to (1: anomaly, 0: normal)
        flags = self.model_instance_.predict(X_clean)
        X_out['anomaly_flag_iforest'] = np.where(flags == -1, 1, 0)
        return X_out


class ClusterScoreTamer(BaseOutlierTamer):
    """
    Article 1 (Christy et al.): Distance-to-Centroid Clustering
    Fits a K-Means algorithm and calculates continuous euclidean distances to the 
    nearest cluster centroid, flagging anomalies without truncating high-risk patient rows.
    """
    def __init__(self, n_clusters=3, contamination=0.05, random_state=42):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.numeric_cols_ = []
        self.model_instance_ = None
        self.threshold_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.numeric_cols_ = self._get_numeric_columns(X_df)
        
        if not self.numeric_cols_:
            return self
            
        X_clean = self._get_clean_numeric_data(X_df, self.numeric_cols_)
        
        self.model_instance_ = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init='auto'
        )
        self.model_instance_.fit(X_clean)
        
        # Establish the cutoff threshold based on training contamination parameters
        distances = self.model_instance_.transform(X_clean)
        min_distances = np.min(distances, axis=1)
        self.threshold_ = np.percentile(min_distances, 100 * (1 - self.contamination))
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        if not self.numeric_cols_ or self.model_instance_ is None:
            return X_out
            
        X_clean = self._get_clean_numeric_data(X_out, self.numeric_cols_)
        
        distances = self.model_instance_.transform(X_clean)
        min_distances = np.min(distances, axis=1)
        
        X_out['anomaly_score_cluster'] = min_distances
        X_out['anomaly_flag_cluster'] = (min_distances > self.threshold_).astype(int)
        return X_out


class StatisticalZTamer(BaseOutlierTamer):
    """
    Article 3 (Gaspar et al.): Parametric Baseline Z-Scoring
    Calculates statistical standard deviations across clinical distributions, 
    matching standard administrative healthcare baselines.
    """
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.numeric_cols_ = []
        self.means_ = None
        self.stds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.numeric_cols_ = self._get_numeric_columns(X_df)
        
        if not self.numeric_cols_:
            return self
            
        X_clean = self._get_clean_numeric_data(X_df, self.numeric_cols_)
        self.means_ = X_clean.mean()
        self.stds_ = X_clean.std().replace(0, 1.0) # Prevent zero-division errors
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        if not self.numeric_cols_ or self.means_ is None:
            return X_out
            
        X_clean = self._get_clean_numeric_data(X_out, self.numeric_cols_)
        
        z_scores = np.abs((X_clean - self.means_) / self.stds_)
        max_z_per_row = z_scores.max(axis=1)
        
        X_out['max_clinical_z_score'] = max_z_per_row
        X_out['anomaly_flag_z'] = (max_z_per_row > self.threshold).astype(int)
        return X_out
