from collections import Counter # Balanced Learning and Model Training
from imblearn.over_sampling import SMOTE, RandomOverSampler # Balanced Learning
import numpy as np

def is_imbalanced(y, threshold=0.1):
    class_counts = Counter(y)
    majority = max(class_counts.values())
    for count in class_counts.values():
        if (majority - count) / majority > threshold:
            return True
    return False

def apply_balancing(x_train, y_train, method='smote'):
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'random':
        sampler = RandomOverSampler(random_state=42)
    else:
        raise ValueError("Unsupported balancing method. Use 'smote' or 'random'.")
    
    x_res, y_res = sampler.fit_resample(x_train, y_train)
    return x_res, y_res

def check_and_balance(x_train, y_train, method_selection_fn):
    if not is_imbalanced(y_train):
        print("Data appears to be balanced.")
        return x_train, y_train, 'none'
    
    print("Class imbalance detected. Testing balancing methods...")

    methods = ['random', 'smote',"none"]
    best_score = -np.inf
    best_method = 'random'
    best_x, best_y = x_train, y_train

    for method in methods:
        if(method=="none"):
            x_try, y_try = x_train,y_train
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
