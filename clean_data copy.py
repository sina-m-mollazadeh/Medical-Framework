import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class DropRowsImputer(BaseEstimator, TransformerMixin):
    """
    WARNING: Standard Scikit-Learn Pipelines do not support row-dropping in `transform`.
    If this step drops rows from X, the pipeline will crash because y is not automatically 
    truncated to match. Only use this outside of a standard Pipeline or with imblearn's FunctionSampler.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        mask = X_df.notna().all(axis=1)
        return X_df[mask]

class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="mean")

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_imputed = self.imputer.transform(X_df)
        return pd.DataFrame(X_imputed, columns=X_df.columns, index=X_df.index)

class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_imputed = self.imputer.transform(X_df)
        return pd.DataFrame(X_imputed, columns=X_df.columns, index=X_df.index)

class ClassMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.means_dict = {}
        self.overall_mean = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.overall_mean = X_df.mean()
        if y is not None:
            self.means_dict = X_df.groupby(np.array(y)).mean().to_dict()
        return self

    def transform(self, X):
        # Inference phase: y is unknown, use overall mean
        X_df = pd.DataFrame(X).copy()
        return X_df.fillna(self.overall_mean)

    def fit_transform(self, X, y=None, **fit_params):
        # Training phase: y is known, use class-specific means
        self.fit(X, y, **fit_params)
        X_df = pd.DataFrame(X).copy()
        
        if y is not None and self.means_dict:
            y_series = pd.Series(np.array(y), index=X_df.index)
            for col in X_df.columns:
                if col in self.means_dict:
                    X_df[col] = X_df[col].fillna(y_series.map(self.means_dict[col]))
                    
        return X_df.fillna(self.overall_mean)

class ClassMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians_dict = {}
        self.overall_median = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.overall_median = X_df.median()
        if y is not None:
            self.medians_dict = X_df.groupby(np.array(y)).median().to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.fillna(self.overall_median)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        X_df = pd.DataFrame(X).copy()
        
        if y is not None and self.medians_dict:
            y_series = pd.Series(np.array(y), index=X_df.index)
            for col in X_df.columns:
                if col in self.medians_dict:
                    X_df[col] = X_df[col].fillna(y_series.map(self.medians_dict[col]))
                    
        return X_df.fillna(self.overall_median)

class FFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).ffill().bfill()

class BFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).bfill().ffill()

class InterpolateImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).interpolate(method="linear", limit_direction="both")

class IterativeModelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.imputer = None

    def fit(self, X, y=None):
        self.imputer = IterativeImputer(random_state=self.random_state)
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_imputed = self.imputer.transform(X_df)
        return pd.DataFrame(X_imputed, columns=X_df.columns, index=X_df.index)

class KNNImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.imputer = None

    def fit(self, X, y=None):
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_imputed = self.imputer.transform(X_df)
        return pd.DataFrame(X_imputed, columns=X_df.columns, index=X_df.index)

class ModelImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.models = {}
        self.fill_values = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.fill_values = X_df.mean()
        
        for col in X_df.columns:
            missing_mask = X_df[col].isna()
            known_X = X_df[~missing_mask].drop(columns=[col])
            known_y = X_df.loc[~missing_mask, col]
            
            if not known_X.empty and known_y.nunique() > 1:
                model = LinearRegression()
                model.fit(known_X.fillna(self.fill_values.drop(col)), known_y)
                self.models[col] = model
                
        return self

    def transform(self, X):
        X_copy = pd.DataFrame(X).copy()
        
        for col in X_copy.columns:
            mask = X_copy[col].isna()
            if mask.any():
                if col in self.models:
                    X_subset = X_copy.loc[mask].drop(columns=[col])
                    X_copy.loc[mask, col] = self.models[col].predict(X_subset.fillna(self.fill_values))
                else:
                    X_copy.loc[mask, col] = self.fill_values[col]
                    
        return X_copy.fillna(self.fill_values)