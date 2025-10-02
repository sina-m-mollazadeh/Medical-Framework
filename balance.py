from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

def is_imbalanced(y, threshold=0.1):
    """Check if dataset is imbalanced beyond a threshold."""
    class_counts = Counter(y)
    majority = max(class_counts.values())
    for count in class_counts.values():
        if (majority - count) / majority > threshold:
            return True
    return False


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

    if method not in samplers:
        raise ValueError(f"Unsupported balancing method: {method}")

    sampler = samplers[method]

    try:
        x_res, y_res = sampler.fit_resample(x_train, y_train)
        return x_res, y_res
    except ValueError as e:
        print(f"{method} failed: {e}")
        return x_train, y_train


def check_and_balance(x_train, y_train, method_selection_fn):
    """Test multiple balancing methods and select the best one."""
    if not is_imbalanced(y_train):
        print("Data appears to be balanced.")
        return x_train, y_train, 'none'

    print("Class imbalance detected. Testing balancing methods...")

    methods = ['none', 'random_over', 'smote', 'adasyn', 
               'borderline_smote', 'smote_tomek', 'smote_enn', 
               'random_under']

    best_score = -np.inf
    best_method = 'none'
    best_x, best_y = x_train, y_train

    for method in methods:
        if method == "none":
            x_try, y_try = x_train, y_train
        else:
            x_try, y_try = apply_balancing(x_train, y_train, method)

        score = method_selection_fn(x_try, y_try)
        print(f"Method: {method}, Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_method = method
            best_x, best_y = x_try, y_try

    print(f"Best balancing method: {best_method} with score: {best_score:.4f}")
    return best_x, best_y, best_method
