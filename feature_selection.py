import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest,f_classif,SelectFromModel,RFE,chi2,VarianceThreshold, mutual_info_classif, RFECV # Feature Selection

from sklearn.svm import LinearSVC # Feature Selection

from sklearn.ensemble import ExtraTreesClassifier # Feature Selection

from sklearn.linear_model import LogisticRegression # Feature Selection

from test_models import train_and_evaluate

from itertools import combinations

from sklearn.feature_selection import mutual_info_classif

from sklearn.cluster import KMeans

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
    "SelectK": SelectK, "L1_Based": L1_Based, "TreeBased": TreeBased,
    "RFE_FeatureSelection": RFE_FeatureSelection, "Chi2_Selection": Chi2_Selection, "VarianceThresh": VarianceThresh
}

import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from itertools import combinations

def feature_selection(x_train, y_train, x_val, y_val, categories, weights):
    num_classes = len(np.unique(y_train))
    
    def get_optimal_k_rfecv(X, Y, num_classes):
        scoring_method = 'roc_auc' if num_classes == 2 else 'f1_weighted'
        model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
        selector = RFECV(estimator=model, step=1, cv=5, scoring=scoring_method, n_jobs=-1)
        selector.fit(X, Y)
        return selector.estimator_

    def remove_constant_features(X): 
        return X.loc[:, X.nunique() > 1]

    # --- 1. Cleanup & Phenotype Generation ---
    x_train = remove_constant_features(x_train)
    x_val = x_val[x_train.columns] 

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(x_train)
    val_clusters = kmeans.predict(x_val)
    
    train_cluster_df = pd.get_dummies(train_clusters, prefix='Patient_Phenotype')
    val_cluster_df = pd.get_dummies(val_clusters, prefix='Patient_Phenotype')
    
    x_train = pd.concat([x_train.reset_index(drop=True), train_cluster_df.reset_index(drop=True)], axis=1)
    x_val = pd.concat([x_val.reset_index(drop=True), val_cluster_df.reset_index(drop=True)], axis=1)
    
    forced_features = [c for c in x_train.columns if 'Patient_Phenotype' in c]
    expert_opinion_features = [] 
    forced_features.extend(expert_opinion_features)

    # --- 2. Categorical Selection Logic ---
    selected_from_categories = []
    
    for cat in categories:
        name = cat.get("name")
        num_to_pick = cat.get("num", 0)
        cols = cat.get("columns", [])
        
        available_cols = [c for c in cols if c in x_train.columns]
        if not available_cols:
            continue
            
        x_subset = x_train[available_cols]
        
        if num_to_pick > 0:
            k_target = min(num_to_pick, len(available_cols))
            print(f"Category '{name}': Picking top {k_target} features (User Defined)")
        else:
            k_target = get_optimal_k_rfecv(x_subset, y_train, num_classes)
            print(f"Category '{name}': Picking top {k_target} features (RFECV Optimized)")
            
        scores, _ = f_classif(x_subset, y_train)
        ranked_cols = pd.Series(scores, index=available_cols).sort_values(ascending=False)
        selected_from_categories.extend(ranked_cols.head(k_target).index.tolist())

    # (Interaction block completely removed)

    # Final Selection: only categories + clusters + expert
    final_feature_list = list(set(selected_from_categories + forced_features))
    
    print(f"Final Selection Complete: {len(final_feature_list)} features total.")

    return (x_train[final_feature_list], y_train, 
            x_val[final_feature_list], y_val, 
            "Hybrid_Category_RFECV", 0, {}, 0, "Optimized Selection", kmeans)