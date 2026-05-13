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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #Weak Model Test
from sklearn.linear_model import LogisticRegression # Feature Selection
from sklearn.neighbors import KNeighborsClassifier #Weak Model Test
import pandas as pd
import numpy as np


def is_imbalanced(y, threshold=0.25):
    counter = Counter(y)
    values = list(counter.values())
    if len(values) != 2:
        return False
    minority_ratio = min(values) / sum(values)
    return minority_ratio < threshold

def custom_score(y_true, y_pred, num_classes):
    average = "weighted" if num_classes != 2 else "binary"
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average=average,zero_division=0)
    prec = precision_score(y_true, y_pred, average=average,zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average)
    weighted_sum = rec+prec+acc+f1
    return weighted_sum / 4

def run_model_with_grid_search(model_name, model, param_grid, x_train, x_test, y_train, y_test, return_model, num_classes):
    if is_imbalanced(y_train):
        if "class_weight" in model.get_params().keys():
            param_grid["class_weight"] = ["balanced"]
        if model_name == "XGBoostBased":
            if(num_classes==2):
                neg, pos = np.bincount(y_train)
                param_grid["scale_pos_weight"] = [neg / pos]

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Running {model_name} with {total_combinations} model configurations...")
    grid = GridSearchCV(model, param_grid, cv=2)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    score = custom_score(y_test, y_pred, num_classes)
    print(f"{model_name} custom score: {score:.4f}")
    return (score, best_model,y_pred,total_combinations) if return_model else (score, None,None,total_combinations)

def NeuralNetworkBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50),(100,50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'max_iter': [300]
    }
    return run_model_with_grid_search("NeuralNetworkBased", MLPClassifier(random_state=42), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

def SVMBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    return run_model_with_grid_search("SVMBased", SVC(probability=True), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

def RandomForestBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    return run_model_with_grid_search("RandomForestBased", RandomForestClassifier(random_state=42), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

def XGBoostBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    return run_model_with_grid_search("XGBoostBased", XGBClassifier(use_label_encoder=True, eval_metric='mlogloss'), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

def LogisticRegressionBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['saga'],
        'max_iter': [1000],
        'l1_ratio': [0, 0.5, 1]
    }
    return run_model_with_grid_search("LogisticRegression", LogisticRegression(), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

def KNNBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    }
    return run_model_with_grid_search("KNeighbors", KNeighborsClassifier(), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

def NaiveBayesBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
    return run_model_with_grid_search("GaussianNB", GaussianNB(), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

def LightGBMBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [-1, 3, 5, 10, 20]
    }
    return run_model_with_grid_search("LightGBM", LGBMClassifier(random_state=42,class_weight="balanced", verbose=-1), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

def CatBoostBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 6, 8]
    }
    return run_model_with_grid_search("CatBoost", CatBoostClassifier(silent=True,allow_writing_files=False), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

def GradientBoostingBased(x_train, x_test, y_train, y_test, return_model, num_classes):
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    return run_model_with_grid_search("GradientBoosting", GradientBoostingClassifier(), param_grid, x_train, x_test, y_train, y_test, return_model, num_classes)

algorithm_functions_model_training = {
    "LogisticRegression": LogisticRegressionBased,
    # "KNN": KNNBased,
    # "NaiveBayes": NaiveBayesBased,
    "RandomForest": RandomForestBased,
    "XGBoost": XGBoostBased,
    "LightGBM": LightGBMBased,
    # "CatBoost": CatBoostBased,
    # "GradientBoosting": GradientBoostingBased,
    # "NeuralNetwork": NeuralNetworkBased,
    # "SVM": SVMBased,
}


def model_training(x_train,x_test,y_train,y_test):
    acc_holder = {}
    best_algo = None
    best_accuracy = 0
    num_classes = pd.concat([y_train, y_test]).nunique()
    cm=None
    fpr=None
    tpr=None
    tot_combs_all_models=0
    for name, func in algorithm_functions_model_training.items():
        print(name)
        accuracy,model,y_pred,total_combinations = func(x_train.copy(),x_test.copy(),y_train.copy(),y_test.copy(),return_model=False,num_classes=num_classes)
        tot_combs_all_models+=total_combinations
        acc_holder[name] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_algo = name
        print()


    if best_algo:
        accuracy,model,y_pred,_ = algorithm_functions_model_training[best_algo](x_train.copy(),x_test.copy(),y_train.copy(),y_test.copy(),return_model=True,num_classes=num_classes)
        cm = confusion_matrix(y_test, y_pred)
        if num_classes == 2:
            y_prob = model.predict_proba(x_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
        else:
            y_test_bin = label_binarize(
                y_test,
                classes=np.unique(y_train)
            )
            y_prob = model.predict_proba(x_test)

            fpr = {}
            tpr = {}

            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(
                    y_test_bin[:, i],
                    y_prob[:, i]
                )


    return model, best_algo, best_accuracy, acc_holder, cm, fpr, tpr,tot_combs_all_models