import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer #Handle Missing
from sklearn.linear_model import LinearRegression #Handle Missing
from sklearn.experimental import enable_iterative_imputer #Handle Missing(Imported Because of API Change)
from sklearn.impute import IterativeImputer #Handle Missing

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

def handling_missing_data(x_train,y_train, x_val ,y_val, train_and_evaluate,weights):

    if x_train.isna().sum().sum() == 0 and x_val.isna().sum().sum() == 0:
        return x_train, y_train, x_val, y_val, "No Null Data", 100, {"No Null Data": 100}, [], None
    
    if x_train.isna().sum().sum() == 0 and x_val.isna().sum().sum() > 0:
        print("Train is clean, but Val has nulls. Forcing Iterative Imputer on Val...")
        imputer_instance = IterativeModelImputer()
        imputer_instance.fit(x_train, y_train) # Learns relationships from clean train
        x_val, y_val = imputer_instance.transform(x_val, y_val)

    acc_holder = {}
    best_algo = None
    best_accuracy = -1
    best_imputer = None
    x_train_backup=x_train.copy()
    x_val_backup=x_val.copy()
    for name, imputer in algorithm_functions_clean_data.items():
        try:
            imputer_instance = imputer.__class__()
            imputer_instance.fit(x_train.copy(), y_train.copy())
            X_transformed, Y_transformed = imputer_instance.transform(x_train.copy(), y_train.copy())
            accuracy = train_and_evaluate(X_transformed, Y_transformed, weights,expand=False)
            acc_holder[name] = accuracy
            print(f"{name}: {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_algo = name
                best_imputer = imputer_instance  
        except TimeoutError:
            acc_holder[name] = -1

    if best_algo:
        x_train, y_train = best_imputer.transform(x_train.copy(), y_train.copy())
        x_val, y_val = best_imputer.transform(x_val.copy(), y_val.copy())
        changes = track_changes(x_train_backup, x_train)
        changes.append(track_changes(x_val_backup,x_val))
    else:
        x_train, y_train = x_train.copy(), y_train.copy()
        x_train, y_train = x_train.copy(), y_train.copy()
        changes=[]
    return x_train, y_train, x_val, y_val, best_algo, best_accuracy, acc_holder, changes,best_imputer

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
        # Save the means per class and the overall mean from TRAIN
        self.means_dict = X.groupby(Y).mean().to_dict()
        self.overall_mean = X.mean()
        return self

    def transform(self, X, Y):
        X_copy = X.copy()
        if Y is None: # If we don't have Y (inference), use overall mean
            return X_copy.fillna(self.overall_mean), Y
            
        for col in X.columns:
            # Map the class-based means onto the nulls
            X_copy[col] = X_copy[col].fillna(pd.Series(Y).map(self.means_dict[col]))
            # If any still remain (e.g., a class didn't exist in train), use overall mean
            X_copy[col] = X_copy[col].fillna(self.overall_mean[col])
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
        return X.ffill().bfill(), Y

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
        self.fill_values = X.mean() # Absolute safety net from TRAIN data

        for col in X.columns:
            missing_mask = X[col].isna()
            # Even if 0 nulls in train, we want a model in case Val has nulls
            known_X = X[~missing_mask].drop(columns=[col])
            known_y = X.loc[~missing_mask, col]
            
            if not known_X.empty and known_y.nunique() > 1:
                model = LinearRegression()
                # Use training means to fill other columns so the model can fit
                model.fit(known_X.fillna(self.fill_values.drop(col)), known_y)
                self.models[col] = model
        return self

    def transform(self, X, Y=None):
        X_copy = X.copy()
        
        for col in X_copy.columns:
            mask = X_copy[col].isna()
            if mask.any():
                if col in self.models:
                    # Predict using the model
                    X_subset = X_copy.loc[mask].drop(columns=[col])
                    X_copy.loc[mask, col] = self.models[col].predict(X_subset.fillna(self.fill_values))
                else:
                    # Fallback to the training mean if no model was possible
                    X_copy.loc[mask, col] = self.fill_values[col]
        
        # Final insurance: fill any remaining (e.g. if prediction returned NaN)
        return X_copy.fillna(self.fill_values), Y

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
    # "drop_rows": DropRowsImputer(),
    # "impute_mean": MeanImputer(),
    # "impute_median": MedianImputer(),
    # "impute_class_mean": ClassMeanImputer(),
    # "impute_class_median": ClassMedianImputer(),
    # "ffill": FFillImputer(),
    # "bfill": BFillImputer(),
    # "interpolate": InterpolateImputer(),
    # "Iterative_model_Imputation": IterativeModelImputer(),
    # "KNN_Imputation": KNNImputerWrapper(),
    "Model_imputation": ModelImputer(),
}
