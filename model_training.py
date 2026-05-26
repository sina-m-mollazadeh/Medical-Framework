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

class LogisticRegressionEstimator(ClassifierMixin, BaseEstimator):
    # _estimator_type belt-and-suspenders: also covered by ClassifierMixin
    # being first in the MRO so __sklearn_tags__() reports classifier in
    # sklearn >=1.6. Without this, sklearn's scorer treats us as a regressor
    # and predict_proba scoring raises ValueError → CV scores become NaN.
    _estimator_type = "classifier"
    def __init__(self, C=1.0, max_iter=1000, penalty='l2', class_weight='balanced'):
        self.C = C
        self.max_iter = max_iter
        self.penalty = penalty
        self.class_weight = class_weight
        self.model = None

    def fit(self, X, y):
        solver = 'liblinear' if self.penalty == 'l1' else 'lbfgs'
        n_jobs = None if solver == 'liblinear' else -1
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, penalty=self.penalty,
            solver=solver, class_weight=self.class_weight,
            random_state=42, n_jobs=n_jobs,
        )
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SVCEstimator(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
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


class RandomForestEstimator(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1,
                 min_samples_split=2, max_features='sqrt', class_weight='balanced'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.class_weight = class_weight
        self.model = None

    def fit(self, X, y):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            class_weight=self.class_weight, random_state=42, n_jobs=1,
        )
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBoostEstimator(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=5,
                 min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                 reg_alpha=0.0, reg_lambda=1.0, scale_pos_weight=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.gamma = gamma
        self.model = None

    def fit(self, X, y):
        self.model = XGBClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, min_child_weight=self.min_child_weight,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight, gamma=self.gamma,
            random_state=42, n_jobs=1, eval_metric='logloss',
            use_label_encoder=False, tree_method='hist',
        )
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBMEstimator(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    def __init__(self, n_estimators=200, learning_rate=0.1, num_leaves=31,
                 min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0,
                 subsample=1.0, colsample_bytree=1.0, class_weight='balanced',
                 scale_pos_weight=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.model = None

    def fit(self, X, y):
        kwargs = dict(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            random_state=42, n_jobs=1, verbose=-1,
        )
        if self.scale_pos_weight is not None:
            kwargs['scale_pos_weight'] = self.scale_pos_weight
        else:
            kwargs['class_weight'] = self.class_weight
        self.model = LGBMClassifier(**kwargs)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CatBoostEstimator(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    def __init__(self, iterations=300, learning_rate=0.1, depth=6,
                 l2_leaf_reg=3.0, auto_class_weights='Balanced'):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.model = None

    def fit(self, X, y):
        self.model = CatBoostClassifier(
            iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, l2_leaf_reg=self.l2_leaf_reg,
            auto_class_weights=self.auto_class_weights,
            random_seed=42, verbose=0, allow_writing_files=False,
        )
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MLPEstimator(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
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
