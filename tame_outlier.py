import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierTamer(BaseEstimator, TransformerMixin):
    """
    Medical-Grade Outlier Handler:
    - Log Transformation for severe skew (Ratio > 20)
    - Flag and Leave for moderate skew (Ratio 5-20)
    - Preserves biological signal for clinical integrity.
    """
    
    def __init__(self):
        self.log_cols = []
        self.flag_thresholds = {}  # {column_name: threshold_value}
        self.best_algo = "Medical_Log_Flag"
        self.acc_holder = {"Medical_Log_Flag": 1.0}
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """Analyze skewness and determine the strategy per column based on Training Data."""
        self.log_cols = []
        self.flag_thresholds = {}
        
        # Only process numeric columns with enough variance
        numeric_cols = [col for col in X.columns if X[col].nunique() > 10]
        
        for col in numeric_cols:
            median_val = X[col].median()
            max_val = X[col].max()
            
            # Division by zero safety
            ratio = max_val / median_val if median_val != 0 else 0
            
            # 1. SEVERE SKEW (> 20x): Log Transform
            if ratio > 20:
                self.log_cols.append(col)
                
            # 2. MODERATE SKEW (5x - 20x): Flag and Leave
            elif ratio > 5:
                mean = X[col].mean()
                std = X[col].std()
                # Store 3-Sigma threshold
                self.flag_thresholds[col] = mean + (3 * std)
                
        self.is_fitted = True
        print(f"Fit Complete: {len(self.log_cols)} columns for Log, {len(self.flag_thresholds)} for Flagging.")
        return self
    
    def transform(self, X, y=None):
        """Apply the saved strategies to any dataset (Train or Val)."""
        if not self.is_fitted:
            raise ValueError("OutlierTamer must be fitted before transforming.")
            
        X_transformed = X.copy()
        
        # Apply Log1p to squelch extreme variance while preserving order
        for col in self.log_cols:
            X_transformed[col] = np.log1p(X_transformed[col])
            
        # Add 'Danger Zone' Flags but LEAVE original values as they are
        for col, threshold in self.flag_thresholds.items():
            flag_name = f"{col}_extreme_flag"
            X_transformed[flag_name] = (X_transformed[col] > threshold).astype(int)
            
        return X_transformed, y
    
def taming_outliers(x_train, y_train, x_val, y_val, weights):
    """
    Standardizes the outlier logic for the pipeline.
    Ensures validation data uses training-set thresholds.
    """
    tamer = OutlierTamer()
    
    # 1. Fit only on Training Data
    tamer.fit(x_train)
    
    # 2. Transform both sets
    x_train_tame, y_train_tame = tamer.transform(x_train, y_train)
    x_val_tame, y_val_tame = tamer.transform(x_val, y_val)
    
    # Metadata for framework compatibility
    best_algo = tamer.best_algo
    acc_holder = tamer.acc_holder
    best_acc = 0.0 # Will be calculated by the training/eval phase
    changes = [] # We added columns/transformed distribution, didn't clip rows
    
    print(f"Outlier Phase: Created {len(tamer.flag_thresholds)} flags and logged {len(tamer.log_cols)} columns.")
    
    return (x_train_tame, y_train_tame, 
            x_val_tame, y_val_tame, 
            best_algo, best_acc, acc_holder, 
            changes, tamer)