import numpy as np
from sklearn.neighbors import LocalOutlierFactor #Tame Outlier
from sklearn.ensemble import IsolationForest #Tame Outlier
from scipy.stats import zscore #Tame Outlier
from test_models import train_and_evaluate

def IQR(X, Y):
    """Adjust outliers based on IQR min-max whiskers."""
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    min_whisker = Q1 - 2.2 * IQR
    max_whisker = Q3 + 2.2 * IQR
    X_adjusted = X.clip(lower=min_whisker, upper=max_whisker, axis=1)
    return X_adjusted, Y

def LOF(X, Y):
    """Adjust only the detected outliers based on LOF anomaly score."""
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    X_scores = clf.fit_predict(X)
    adjustment_factor = np.abs(clf.negative_outlier_factor_) / np.max(np.abs(clf.negative_outlier_factor_))

    X_adjusted = X.copy()
    outlier_mask = X_scores == -1
    X_adjusted[outlier_mask] = X[outlier_mask] * (1 - adjustment_factor[outlier_mask, np.newaxis])
    return X_adjusted, Y

def IsolationForestOutlier(X, Y):
    """Only adjust outliers based on Isolation Forest isolation score."""
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    preds = iso.fit_predict(X)
    scores = iso.decision_function(X)

    adjustment_factor = (scores - scores.min()) / (scores.max() - scores.min())
    X_adjusted = X.copy()

    for i, pred in enumerate(preds):
        if pred == -1:
            X_adjusted.iloc[i] = X.iloc[i] * (1 - adjustment_factor[i])

    return X_adjusted, Y



def SP(X, Y):
    """Adjust only outliers beyond 3 std dev using Standardization Projection."""
    X_adjusted = X.copy()
    Z_scores = np.abs(zscore(X, nan_policy='omit'))
    Z_scores = np.nan_to_num(Z_scores, nan=0)

    for col in X.columns:
        col_idx = X.columns.get_loc(col)
        for i in range(len(X)):
            if Z_scores[i, col_idx] > 3:
                factor = 3 / Z_scores[i, col_idx]
                X_adjusted.iloc[i, col_idx] = X.iloc[i, col_idx] * factor

    return X_adjusted, Y


def IsolationNNe(X, Y):
    """Only adjust outliers based on Isolation Forest distance score."""
    iso = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
    preds = iso.fit_predict(X)
    dist = iso.decision_function(X)
    adjustment_factor = np.abs(dist) / np.max(np.abs(dist))

    X_adjusted = X.copy()
    for i, pred in enumerate(preds):
        if pred == -1:
            X_adjusted.iloc[i] = X.iloc[i] * (1 - adjustment_factor[i])

    return X_adjusted, Y


algorithm_functions_tame_outlier = {
    "IQR": IQR,
    "LOF": LOF,
    "iForest": IsolationForestOutlier,
    "SP": SP,
    "iNNe": IsolationNNe,
}
def track_changes(original_X, new_X):
    changes = []
    for col_idx, col in enumerate(original_X.columns):
        modified = original_X[col] != new_X[col]
        for row_idx in modified[modified].index:
            changes.append((row_idx, col_idx))
    return changes

def taming_outliers(X,Y):
    acc_holder = {}
    x_copy = X.copy()
    y_copy = Y.copy()
    best_algo = None
    best_accuracy = 0


    for name, func in algorithm_functions_tame_outlier.items():
        print(name)
        print()

        X_Adjusted,Y_Adjusted = func(x_copy.copy(), y_copy.copy())
        accuracy = train_and_evaluate(X_Adjusted, Y_Adjusted,expand=False)
        acc_holder[name] = accuracy


        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_algo = name
    if best_algo:
        x_copy,y_copy = algorithm_functions_tame_outlier[best_algo](x_copy.copy(), y_copy.copy())

    changes=track_changes(x_copy,X)
    return x_copy, y_copy, best_algo, best_accuracy,acc_holder,changes