import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer #Handle Missing
from sklearn.linear_model import LinearRegression #Handle Missing
from sklearn.experimental import enable_iterative_imputer #Handle Missing(Imported Because of API Change)
from sklearn.impute import IterativeImputer #Handle Missing
import signal

class TimeoutError(Exception):
    pass
def handler(signum, frame):
    raise TimeoutError("Imputer timed out after 5 minutes")


class BaseImputer:
    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        return X, Y

def track_changes(original_X, new_X):
    changes = []
    original_X, new_X = original_X.align(new_X)

    for col_idx, col in enumerate(original_X.columns):
        modified = original_X[col] != new_X[col]
        for row_idx in modified[modified].index:
            changes.append((row_idx, col_idx))

    return changes

def handling_missing_data(X, Y, train_and_evaluate):

    if X.isna().sum().sum() == 0:
        return X, Y, "No Null Data", 100, {"No Null Data": 100}, [],None

    Y = Y.dropna(inplace=False)
    X = X.loc[Y.index]

    threshold = 2
    valid_cols = X.columns[X.notna().sum() >= threshold]
    only_nan = list(set(X.columns) - set(valid_cols))
    X = X.drop(columns=only_nan)

    acc_holder = {}
    best_algo = None
    best_accuracy = -1
    best_imputer = None
    
    for name, imputer in algorithm_functions_clean_data.items():
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(300)  # 5-minute timeout
        try:
            imputer_instance = imputer.__class__()
            imputer_instance.fit(X.copy(), Y.copy())
            X_transformed, Y_transformed = imputer_instance.transform(X.copy(), Y.copy())
            accuracy = train_and_evaluate(X_transformed, Y_transformed, expand=False)
            acc_holder[name] = accuracy
            print(f"{name}: {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_algo = name
                best_imputer = imputer_instance
                
        except TimeoutError:
            acc_holder[name] = -1
        finally:
            signal.alarm(0)

    if best_algo:
        X_final, Y_final = best_imputer.transform(X.copy(), Y.copy())
        changes = track_changes(X, X_final)
    else:
        X_final, Y_final, changes = X.copy(), Y.copy(), []
    return X_final, Y_final, best_algo, best_accuracy, acc_holder, changes,best_imputer

class DropRowsImputer(BaseImputer):
    def transform(self, X, Y=None):
        mask = X.notna().all(axis=1)
        return X[mask], Y[mask] if Y is not None else None

class MeanImputer(BaseImputer):
    def fit(self, X, Y=None):
        self.imputer = SimpleImputer(strategy="mean")
        self.imputer.fit(X)
        return self

    def transform(self, X, Y=None):
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns), Y

class MedianImputer(BaseImputer):
    def fit(self, X, Y=None):
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(X)
        return self

    def transform(self, X, Y=None):
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns), Y

class ClassMeanImputer(BaseImputer):
    def fit(self, X, Y):
        self.class_means = X.groupby(Y).mean()
        return self

    def transform(self, X, Y):
        X_copy = X.copy()
        for col in X.columns:
            X_copy[col] = X.groupby(Y)[col].transform(lambda x: x.fillna(x.mean()))
            X_copy[col] = X_copy[col].fillna(X[col].mean())
        return X_copy, Y

class ClassMedianImputer(BaseImputer):
    def fit(self, X, Y):
        self.class_medians = X.groupby(Y).median()
        return self

    def transform(self, X, Y):
        X_copy = X.copy()
        for col in X.columns:
            X_copy[col] = X.groupby(Y)[col].transform(lambda x: x.fillna(x.median()))
            X_copy[col]=X_copy[col].fillna(X[col].mean())
        return X_copy, Y

class FFillImputer(BaseImputer):
    def transform(self, X, Y=None):
        X_filled = X.ffill().bfill()
        return X_filled, Y

class BFillImputer(BaseImputer):
    def transform(self, X, Y=None):
        X_filled = X.bfill().ffill()
        return X_filled, Y

class InterpolateImputer(BaseImputer):
    def transform(self, X, Y=None):
        X_filled = X.interpolate(method="linear", limit_direction="both")
        return X_filled, Y

class ModelImputer(BaseImputer):
    def fit(self, X, Y=None):
        self.models = {}
        self.means = {}  

        for col in X.columns:
            missing_mask = X[col].isna()
            if missing_mask.sum() > 0:
                known_X = X[~missing_mask].drop(columns=[col])
                known_y = X.loc[~missing_mask, col]
                if not known_X.empty:
                    col_means = known_X.mean(numeric_only=True)
                    self.means[col] = col_means 
                    model = LinearRegression()
                    model.fit(known_X.fillna(col_means), known_y)
                    self.models[col] = model
        return self

    def transform(self, X, Y=None):
        X_copy = X.copy()
        for col, model in self.models.items():
            missing_mask = X_copy[col].isna()
            if missing_mask.sum() > 0:
                missing_X = X_copy.loc[missing_mask].drop(columns=[col])
                missing_X_filled = missing_X.fillna(self.means[col])
                X_copy.loc[missing_mask, col] = model.predict(missing_X_filled)
        return X_copy, Y

class IterativeModelImputer(BaseImputer):
    def fit(self, X, Y=None):
        self.imputer = IterativeImputer()
        self.imputer.fit(X)
        return self

    def transform(self, X, Y=None):
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns), Y

class KNNImputerWrapper(BaseImputer):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, Y=None):
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer.fit(X)
        return self

    def transform(self, X, Y=None):
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns), Y

algorithm_functions_clean_data = {
    "drop_rows": DropRowsImputer(),
    "impute_mean": MeanImputer(),
    "impute_median": MedianImputer(),
    "impute_class_mean": ClassMeanImputer(),
    "impute_class_median": ClassMedianImputer(),
    "ffill": FFillImputer(),
    "bfill": BFillImputer(),
    "interpolate": InterpolateImputer(),
    "Iterative_model_Imputation": IterativeModelImputer(),
    "KNN_Imputation": KNNImputerWrapper()
}
 # "Model_imputation": ModelImputer(),
