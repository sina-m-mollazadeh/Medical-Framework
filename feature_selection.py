from sklearn.feature_selection import SelectKBest,f_classif,SelectFromModel,RFE,chi2,VarianceThreshold, mutual_info_classif # Feature Selection
from sklearn.svm import LinearSVC # Feature Selection
from sklearn.ensemble import ExtraTreesClassifier # Feature Selection
from sklearn.linear_model import LogisticRegression # Feature Selection
from test_models import train_and_evaluate
import pandas as pd
import numpy as np

def SelectK(X, Y, k):
    selector = SelectKBest(f_classif, k="all")
    selector.fit(X.values, Y)
    selected_X = selector.transform(X.values)
    scores = selector.scores_
    return pd.DataFrame(selected_X, columns=X.columns[selector.get_support()]), Y, scores


def L1_Based(X, Y, k):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=2000).fit(X.values, Y)
    model = SelectFromModel(lsvc, prefit=True)
    selected_X = model.transform(X.values)
    coefs = np.abs(lsvc.coef_).mean(axis=0)
    return pd.DataFrame(selected_X, columns=X.columns[model.get_support()]), Y, coefs


def TreeBased(X, Y, k):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(X.values, Y)
    model = SelectFromModel(clf, prefit=True)
    selected_X = model.transform(X.values)
    importances = clf.feature_importances_
    return pd.DataFrame(selected_X, columns=X.columns[model.get_support()]), Y, importances

def RFE_FeatureSelection(X, Y, k):
    model = LogisticRegression(max_iter=1000)
    selector = RFE(model, n_features_to_select=int(X.shape[1] * 0.5), step=1)
    selector.fit(X, Y)
    selected_X = selector.transform(X)
    return pd.DataFrame(selected_X, columns=X.columns[selector.get_support()]), Y, selector.ranking_

def Chi2_Selection(X, Y, k):
    if (X < 0).any().any():
        print("Negative values detected in X. Skipping chi2.")
        zero_scores = np.zeros(X.shape[1])
        return X, Y, zero_scores

    scores, _ = chi2(X, Y)
    score_series = pd.Series(scores, index=X.columns)
    top_k_features = score_series.nlargest(k).index
    selected_X = X[top_k_features]
    return selected_X, Y, scores

def VarianceThresh(X, Y, k):
    selector = VarianceThreshold(threshold=0.01)
    selected_X = selector.fit_transform(X)
    return pd.DataFrame(selected_X, columns=X.columns[selector.get_support()]), Y, selector.variances_

def MutualInfo(X, Y, k):
    scores = mutual_info_classif(X, Y)
    score_series = pd.Series(scores, index=X.columns)
    top_k_features = score_series.nlargest(k).index
    selected_X = X[top_k_features]

    return selected_X, Y, score_series

algorithm_functions_feature_selection = {
    "SelectK": SelectK,
    "L1_Based": L1_Based,
    "TreeBased": TreeBased,
    "RFE_FeatureSelection":RFE_FeatureSelection,
    "Chi2_Selection":Chi2_Selection,
    "VarianceThresh":VarianceThresh
}

def feature_selection(X, Y, categories):
    """Removes features that have the same value in all samples."""

    def remove_constant_features(X):
        return X.loc[:, X.nunique(dropna=False) > 1]
    
    def ReturnNumSelectColumns(sorted_cols):
        sorted_cols=pd.Series(sorted_cols)
        sorted_cols = sorted_cols.apply(lambda x: x[1])
        q1=pd.Series(sorted_cols).quantile(0.25)
        q3=pd.Series(sorted_cols).quantile(0.75)
        min_cut=q1-1.5*(q3-q1)
        count=0
        for num in sorted_cols:
            if num >= min_cut:
                count+=1
        return count
    
    acc_holder = {}
    X = remove_constant_features(X)
    y_copy = Y.copy()
    best_algo = None
    best_accuracy = 0

    selected_columns = []
    category_score_maps = {} 

    for cat in categories:
        cat_name = cat["name"]
        cat_cols = [col for col in cat["columns"] if col in X.columns]
        num_to_select = cat["num"]

        x_cat = X[cat_cols].copy()
        cat_scores_accumulator = np.zeros(len(x_cat.columns))
        for name, func in algorithm_functions_feature_selection.items():
            k = min(len(cat_cols), len(x_cat.columns))
            X_Adjusted, Y_Adjusted, Scores = func(x_cat.copy(), y_copy.copy(), k)

            if Scores.min() == Scores.max():
                accuracy = 0
            else:
                norm_scores = (Scores - Scores.min()) / (Scores.max() - Scores.min())
                accuracy = train_and_evaluate(X_Adjusted, Y_Adjusted, expand=False)
                cat_scores_accumulator += norm_scores * accuracy

            acc_holder[f"{cat_name}_{name}"] = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_algo = name

        score_map = dict(zip(x_cat.columns, cat_scores_accumulator))
        category_score_maps[cat_name] = score_map
        sorted_cols = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        
        if(num_to_select==0 and cat_name=="ALL_COLS"):
            num_to_select=ReturnNumSelectColumns(sorted_cols)

        top_n_cols = [col for col, _ in sorted_cols[:num_to_select]]
        selected_columns.extend(top_n_cols)

    x_selected = X[selected_columns]
    final_acc = train_and_evaluate(x_selected, y_copy, expand=False)

    final_result = f"Selected {len(selected_columns)} features: {selected_columns}"
    print(final_result)

    return x_selected, y_copy, best_algo, best_accuracy, acc_holder, final_acc, final_result