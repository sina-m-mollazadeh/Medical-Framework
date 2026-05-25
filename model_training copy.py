from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class LogisticRegressionEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, y):
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight='balanced', random_state=42, n_jobs=-1)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SVCEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf'):
        self.C = C
        self.kernel = kernel
        self.model = None

    def fit(self, X, y):
        self.model = SVC(C=self.C, kernel=self.kernel, class_weight='balanced', probability=True, random_state=42, max_iter=2000)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class RandomForestEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None

    def fit(self, X, y):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, class_weight='balanced', random_state=42, n_jobs=-1)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBoostEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None

    def fit(self, X, y):
        self.model = XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, random_state=42, n_jobs=-1, eval_metric='logloss')
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBMEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X, y):
        self.model = LGBMClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CatBoostEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=100, learning_rate=0.1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X, y):
        self.model = CatBoostClassifier(iterations=self.iterations, learning_rate=self.learning_rate, random_seed=42, verbose=0)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MLPEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), max_iter=500):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, y):
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter, random_state=42)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
