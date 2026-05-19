from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def is_imbalanced(y, threshold=0.1):
    """Check if dataset is imbalanced beyond a threshold."""
    class_counts = Counter(y)
    if len(class_counts) < 2: return False
    total = sum(class_counts.values())
    minority_ratio = min(class_counts.values()) / total
    return minority_ratio < threshold

def apply_balancing(x_train, y_train, method='smote'):
    """Apply a selected balancing method safely."""
    samplers = {
        'smote': SMOTE(random_state=42, k_neighbors=3),
        'random_over': RandomOverSampler(random_state=42),
        'adasyn': ADASYN(random_state=42, n_neighbors=3),
        'borderline_smote': BorderlineSMOTE(random_state=42, k_neighbors=3),
        'smote_tomek': SMOTETomek(random_state=42),
        'smote_enn': SMOTEENN(random_state=42),
        'random_under': RandomUnderSampler(random_state=42),
    }

    if method == 'none':
        return x_train, y_train
        
    sampler = samplers.get(method)
    if not sampler:
        return x_train, y_train

    try:
        x_res, y_res = sampler.fit_resample(x_train, y_train)
        return x_res, y_res
    except Exception:
        return x_train, y_train

def _internal_selection_score(x_train_bal, y_train_bal, x_val, y_val):
    """
    Evaluates a balancing method correctly:
    Train on Balanced Data -> Test on PURE Unbalanced Data.
    """
    # We use a simple but robust model to 'probe' the quality of the balancing
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    
    try:
        model.fit(x_train_bal, y_train_bal)
        y_pred = model.predict(x_val)
        # Use F2 score to prioritize clinical Recall
        return fbeta_score(y_val, y_pred, beta=2.0, zero_division=0)
    except:
        return 0

def check_and_balance(x_train, y_train, x_val, y_val, weights=None):
    """
    Compares balancing methods by testing their performance on a 
    pure (unbalanced) validation set.
    """
    if not is_imbalanced(y_train):
        print("Data is already balanced. Skipping balancing step.")
        return x_train, y_train, 'none'

    print("Class imbalance detected. Selecting best balancing strategy...")

    methods = ['none', 'random_over', 'smote', 'adasyn', 
               'borderline_smote', 'smote_tomek', 'smote_enn', 
               'random_under']

    best_score = -1
    best_method = 'none'

    for method in methods:
        # 1. Balance the training data
        x_t_res, y_t_res = apply_balancing(x_train, y_train, method)
        
        # 2. Score it against the PURE validation set (No Leakage)
        score = _internal_selection_score(x_t_res, y_t_res, x_val, y_val)
        
        print(f"  Method: {method:16} | Val F2-Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_method = method

    print(f"\nFinal Selection: {best_method} (Best Val Score: {best_score:.4f})")
    
    # Return the balanced training set of the winner
    best_x, best_y = apply_balancing(x_train, y_train, best_method)
    return best_x, best_y, best_method