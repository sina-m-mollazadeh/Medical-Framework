import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, PowerTransformer

class MinMaxScalerNorm(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return pd.DataFrame(self.scaler.transform(X_df), columns=X_df.columns, index=X_df.index)


class RobustScalerNorm(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return pd.DataFrame(self.scaler.transform(X_df), columns=X_df.columns, index=X_df.index)


class ZScoreNormalizationNorm(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return pd.DataFrame(self.scaler.transform(X_df), columns=X_df.columns, index=X_df.index)


class PowerTransformerNorm(BaseEstimator, TransformerMixin):
    def __init__(self, method='yeo-johnson'):
        self.method = method
        self.scaler = PowerTransformer(method=self.method, standardize=True)

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return pd.DataFrame(self.scaler.transform(X_df), columns=X_df.columns, index=X_df.index)

