import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE, RFECV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

class SelectKBestFilter(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
        self.selector = None
        self.scores_ = None

    def fit(self, X, y):
        # Convert k to integer or valid limit
        k_val = min(self.k, X.shape[1]) if isinstance(self.k, int) else "all"
        self.selector = SelectKBest(score_func=f_classif, k=k_val)
        self.selector.fit(X, y)
        self.scores_ = self.selector.scores_
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        selected = self.selector.transform(X_df)
        return pd.DataFrame(selected, columns=X_df.columns[self.selector.get_support()], index=X_df.index)


class L1BasedSelection(BaseEstimator, TransformerMixin):
    def __init__(self, C=0.01, max_iter=2000):
        self.C = C
        self.max_iter = max_iter
        self.selector = None
        self.coefs_ = None

    def fit(self, X, y):
        lsvc = LinearSVC(C=self.C, penalty="l1", dual=False, max_iter=self.max_iter, random_state=42)
        lsvc.fit(X, y)
        self.selector = SelectFromModel(lsvc, prefit=True)
        self.coefs_ = np.abs(lsvc.coef_).mean(axis=0) if lsvc.coef_.ndim > 1 else np.abs(lsvc.coef_)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        selected = self.selector.transform(X_df)
        return pd.DataFrame(selected, columns=X_df.columns[self.selector.get_support()], index=X_df.index)


class TreeBasedSelection(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=50, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.selector = None
        self.feature_importances_ = None

    def fit(self, X, y):
        clf = ExtraTreesClassifier(n_estimators=self.n_estimators, max_features=self.max_features, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        self.selector = SelectFromModel(clf, prefit=True)
        self.feature_importances_ = clf.feature_importances_
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        selected = self.selector.transform(X_df)
        return pd.DataFrame(selected, columns=X_df.columns[self.selector.get_support()], index=X_df.index)


class RFERecursiveSelection(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=10):
        self.n_features_to_select = n_features_to_select
        self.selector = None

    def fit(self, X, y):
        estimator = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1)
        n_select = min(self.n_features_to_select, X.shape[1])
        self.selector = RFE(estimator=estimator, n_features_to_select=n_select, step=1)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        selected = self.selector.transform(X_df)
        return pd.DataFrame(selected, columns=X_df.columns[self.selector.get_support()], index=X_df.index)