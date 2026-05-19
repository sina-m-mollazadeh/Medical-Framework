from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, PowerTransformer
from test_models import train_and_evaluate
import pandas as pd
import numpy as np

class Normalizer:
    def __init__(self):
        self.best_algo = None
        self.best_accuracy = 0
        self.acc_holder = {}
        self.fitted_params = {}
        self.is_fitted = False
    
    def _PowerTransformerNorm(self, X, Y, fit=False):
        """Yeo-Johnson transformation to make data more Gaussian-like."""
        if fit:
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            scaler.fit(X)
            self.fitted_params['PowerTransformerNorm'] = {'scaler': scaler}
        
        scaler = self.fitted_params.get('PowerTransformerNorm', {}).get('scaler')
        if scaler is None:
            scaler = PowerTransformer(method='yeo-johnson', standardize=True).fit(X)
        
        X_transformed = scaler.transform(X)
        return pd.DataFrame(X_transformed, columns=X.columns, index=X.index), Y

    def _MinMaxScalerNorm(self, X, Y, fit=False):
        if fit:
            scaler = MinMaxScaler()
            scaler.fit(X)
            self.fitted_params['MinMaxScalerNorm'] = {'scaler': scaler}
        scaler = self.fitted_params.get('MinMaxScalerNorm', {}).get('scaler')
        if scaler is None: scaler = MinMaxScaler().fit(X)
        return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index), Y
    
    def _RobustScalerNorm(self, X, Y, fit=False):
        if fit:
            transformer = RobustScaler(with_centering=True, with_scaling=True)
            transformer.fit(X)
            X_transformed = transformer.transform(X)
            denom = X_transformed.max(axis=0) - X_transformed.min(axis=0)
            denom[denom == 0] = 1
            min_vals = X_transformed.min(axis=0)
            self.fitted_params['RobustScalerNorm'] = {'scaler': transformer, 'min_vals': min_vals, 'denom': denom}
        params = self.fitted_params.get('RobustScalerNorm', {})
        X_transformed = params['scaler'].transform(X)
        X_scaled = (X_transformed - params['min_vals']) / params['denom']
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), Y
    
    def _ZScoreNormalizationNorm(self, X, Y, fit=False):
        if fit:
            scaler = StandardScaler()
            scaler.fit(X)
            X_transformed = scaler.transform(X)
            denom = X_transformed.max(axis=0) - X_transformed.min(axis=0)
            denom[denom == 0] = 1
            min_vals = X_transformed.min(axis=0)
            self.fitted_params['ZScoreNormalizationNorm'] = {'scaler': scaler, 'min_vals': min_vals, 'denom': denom}
        params = self.fitted_params.get('ZScoreNormalizationNorm', {})
        X_transformed = params['scaler'].transform(X)
        X_scaled = (X_transformed - params['min_vals']) / params['denom']
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), Y
    
    def fit(self, X_train, y_train, weights):
        algorithm_functions = {
            "MinMaxScalerNorm": self._MinMaxScalerNorm,
            "RobustScalerNorm": self._RobustScalerNorm,
            "ZScoreNormalizationNorm": self._ZScoreNormalizationNorm,
            "PowerTransformerNorm": self._PowerTransformerNorm, # Added Strategy
        }
        print("Selecting best normalization algorithm...")
        for name, func in algorithm_functions.items():
            X_adjusted, _ = func(X_train.copy(), y_train.copy(), fit=True)
            accuracy = train_and_evaluate(X_adjusted, y_train, weights, expand=False)
            self.acc_holder[name] = accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_algo = name
        self.is_fitted = True
        return self
    
    def transform(self, X, y=None):
        if not self.is_fitted: raise ValueError("Not fitted.")
        algorithm_functions = {
            "MinMaxScalerNorm": self._MinMaxScalerNorm,
            "RobustScalerNorm": self._RobustScalerNorm,
            "ZScoreNormalizationNorm": self._ZScoreNormalizationNorm,
            "PowerTransformerNorm": self._PowerTransformerNorm,
        }
        return algorithm_functions[self.best_algo](X, y, fit=False)

    def fit_transform(self, X_train, y_train, weights):
        self.fit(X_train, y_train, weights)
        return self.transform(X_train, y_train)

def normalization(x_train_clean, y_train_clean, x_val_clean, y_val_clean, weights):
    normalizer = Normalizer()
    x_train_norm, y_train_norm = normalizer.fit_transform(x_train_clean, y_train_clean, weights)
    x_val_norm, y_val_norm = normalizer.transform(x_val_clean, y_val_clean)
    return (x_train_norm, y_train_norm, x_val_norm, y_val_norm, normalizer.best_algo, normalizer.best_accuracy, normalizer.acc_holder, normalizer)