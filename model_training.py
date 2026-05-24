import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV #Model Training
from sklearn.neural_network import MLPClassifier #Model Training
from sklearn.svm import SVC #Model Training
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #Model Training
from sklearn.naive_bayes import GaussianNB #Model Training
from xgboost import XGBClassifier #Model Training
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve #Model Training
from lightgbm import LGBMClassifier #Model Training
from catboost import CatBoostClassifier #Model Training
from collections import Counter # Balanced Learning and Model Training
from sklearn.linear_model import LogisticRegression # Feature Selection
from sklearn.neighbors import KNeighborsClassifier #Weak Model Test
from sklearn.metrics import make_scorer #Scoring
from sklearn.metrics import average_precision_score, fbeta_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV

def is_imbalanced(y, threshold=0.25):
    counter = Counter(y)
    values = list(counter.values())
    if len(values) != 2:
        return False
    minority_ratio = min(values) / sum(values)
    return minority_ratio < threshold

# def custom_score(y_true, y_pred, num_classes,weights):
#     average = "weighted" if num_classes != 2 else "binary"
#     acc = accuracy_score(y_true, y_pred)
#     rec = recall_score(y_true, y_pred, average=average,zero_division=0,pos_label=1)
#     prec = precision_score(y_true, y_pred, average=average,zero_division=0,pos_label=1)
#     f1 = f1_score(y_true, y_pred, average=average,zero_division=0,pos_label=1)
#     weighted_sum = acc*weights[0]+rec*weights[1]+prec*weights[2]+f1*weights[3]
#     return weighted_sum / sum(weights)

def custom_score(y_true, y_pred_or_proba, num_classes=2, metric_type="f2"):
    if metric_type == "pr_auc" and num_classes == 2:
        return average_precision_score(y_true, y_pred_or_proba)
        
    elif metric_type == "f2":
        average = "weighted" if num_classes > 2 else "binary"
        return fbeta_score(y_true, y_pred_or_proba, beta=2.0, average=average, zero_division=0)
        
    elif metric_type == "f1":
        average = "weighted" if num_classes > 2 else "binary"
        return f1_score(y_true, y_pred_or_proba, average=average, zero_division=0)
    


def run_model_with_grid_search(model_name, model, param_grid, x_train, x_test, y_train, y_test, return_model, num_classes, weights):
    if is_imbalanced(y_train):
        if "class_weight" in model.get_params().keys():
            param_grid["class_weight"] = ["balanced"]
        if model_name == "XGBoostBased" and num_classes == 2:
            neg, pos = np.bincount(y_train)
            param_grid["scale_pos_weight"] = [neg / pos]

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Running {model_name} with {total_combinations} model configurations...")

    # Switch to ROC-AUC for intuitive 0.5-1.0 scoring
    scoring_metric = 'roc_auc' if num_classes == 2 else 'f1_weighted'

    grid = GridSearchCV(model, param_grid, cv=5, scoring=scoring_metric)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    
    # Calculate final output score based on probabilities
    if num_classes == 2 and hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(x_test)[:, 1]
        score = roc_auc_score(y_test, y_prob)
        print(f"{model_name} ROC-AUC Score: {score:.4f}")
    else:
        score = fbeta_score(y_test, y_pred, beta=2.0, average='weighted' if num_classes > 2 else 'binary', zero_division=0)
        print(f"{model_name} F2 Score: {score:.4f}")

    return (score, best_model, y_pred, total_combinations) if return_model else (score, None, None, total_combinations)

def NeuralNetworkBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50),(100,50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'max_iter': [300]
    }
    return run_model_with_grid_search("NeuralNetworkBased", MLPClassifier(random_state=42), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

def SVMBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    return run_model_with_grid_search("SVMBased", SVC(probability=True), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

def RandomForestBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'n_estimators': [50, 100, 200, 500,1000,2000],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    return run_model_with_grid_search("RandomForestBased", RandomForestClassifier(random_state=42), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

def XGBoostBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'n_estimators': [200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5]
    }
    return run_model_with_grid_search("XGBoostBased", XGBClassifier(use_label_encoder=True, eval_metric='mlogloss'), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

def LogisticRegressionBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['saga'],
        'max_iter': [1000],
        'l1_ratio': [0, 0.5, 1]
    }
    return run_model_with_grid_search("LogisticRegression", LogisticRegression(), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

def KNNBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    }
    return run_model_with_grid_search("KNeighbors", KNeighborsClassifier(), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

def NaiveBayesBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
    return run_model_with_grid_search("GaussianNB", GaussianNB(), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

def LightGBMBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [-1, 3, 5, 10, 20]
    }
    return run_model_with_grid_search("LightGBM", LGBMClassifier(random_state=42,class_weight="balanced", verbose=-1), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

def CatBoostBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 6, 8]
    }
    return run_model_with_grid_search("CatBoost", CatBoostClassifier(silent=True,allow_writing_files=False), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

def GradientBoostingBased(x_train, x_test, y_train, y_test, return_model, num_classes,weights):
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    return run_model_with_grid_search("GradientBoosting", GradientBoostingClassifier(), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes,weights)

algorithm_functions_model_training = {
    "LogisticRegression": LogisticRegressionBased,
    "KNN": KNNBased,
    "NaiveBayes": NaiveBayesBased,
    "RandomForest": RandomForestBased,
    "XGBoost": XGBoostBased,
    # "LightGBM": LightGBMBased,
    # "CatBoost": CatBoostBased,
    # "GradientBoosting": GradientBoostingBased,
    # "NeuralNetwork": NeuralNetworkBased,
    # "SVM": SVMBased,
}


def train_stacked_model(top_model_funcs, x_train, x_test, y_train, y_test, num_classes, weights):
    base_estimators = []
    
    # 1. Fit and Calibrate Base Models
    for name, func in top_model_funcs:
        _, best_uncalibrated_model, _, _ = func(x_train, x_test, y_train, y_test, True, num_classes, weights)
        
        calibrated_model = CalibratedClassifierCV(
            estimator=best_uncalibrated_model, 
            method='isotonic', 
            cv='prefit' # We use prefit because the model was already trained inside 'func'
        )
        calibrated_model.fit(x_train, y_train) # Calibrate on the hold-out/validation data
        
        base_estimators.append((name, calibrated_model))

    # 2. Prepare Meta-features using Out-of-Fold predictions

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_train = np.zeros((x_train.shape[0], len(base_estimators)))
    
    for i, (name, model) in enumerate(base_estimators):
        probs = cross_val_predict(model.estimator, x_train, y_train, cv=cv, method='predict_proba')
        meta_train[:, i] = probs[:, 1]

    # 3. Meta-Learner with Ridge Penalty (C=0.1)
    meta_learner = LogisticRegression(C=0.1, penalty='l2', solver='saga', max_iter=2000, class_weight='balanced')
    meta_learner.fit(meta_train, y_train)

    # 4. Generate Meta-features for the test set
    meta_test = np.zeros((x_test.shape[0], len(base_estimators)))
    for i, (name, model) in enumerate(base_estimators):
        meta_test[:, i] = model.predict_proba(x_test)[:, 1]

    y_prob = meta_learner.predict_proba(meta_test)[:, 1]
    y_pred = meta_learner.predict(meta_test)
    
    return meta_learner, base_estimators, roc_auc_score(y_test, y_prob), y_pred, y_prob

# ... [Individual model functions like SVMBased, XGBoostBased, etc., remain the same] ...

def model_training(x_train, x_test, y_train, y_test, weights):
    # Step 1: Evaluate individual models
    acc_holder, model_func_map = {}, {}
    num_classes = pd.concat([y_train, y_test]).nunique()
    tot_combs = 0

    for name, func in algorithm_functions_model_training.items():
        # Evaluate to find which architectures work best
        accuracy, _, _, combs = func(x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy(), False, num_classes, weights)
        acc_holder[name], model_func_map[name], tot_combs = accuracy, func, tot_combs + combs

    # Step 2: Stack Top 3-5 Models
    # Sort by the ROC-AUC score we found in Step 1
    sorted_algos = sorted(acc_holder.items(), key=lambda x: x[1], reverse=True)
    top_names = [name for name, score in sorted_algos[:min(5, len(algorithm_functions_model_training))]]
    top_funcs = [(n, model_func_map[n]) for n in top_names]
    
    # Run the new Calibrated Stack
    meta_model, base_estimators, stacked_acc, y_pred, y_prob = train_stacked_model(
        top_funcs, x_train, x_test, y_train, y_test, num_classes, weights
    )

    # Final Output
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob) if num_classes == 2 else (None, None, None)

    stack_info = {
        "meta_model": meta_model, 
        "base_models": base_estimators, 
        "top_algos": top_names,
        "calibration_method": "isotonic"
    }
    print(f"FINAL CALIBRATED STACKED ROC-AUC: {stacked_acc:.4f}")
    
    return stack_info, "StackedGeneralization", stacked_acc, acc_holder, cm, fpr, tpr, tot_combs