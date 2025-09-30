from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler # Nomalization
from test_models import train_and_evaluate
import pandas as pd

def MinMaxScalarNorm(X,Y):
  scaler = MinMaxScaler()
  scaler.fit(X)
  return pd.DataFrame(scaler.transform(X),columns=X.columns,index=X.index),Y

def RobustScalarNorm(X, Y):
    transformer = RobustScaler(with_centering=True, with_scaling=True)
    X_transformed = transformer.fit_transform(X)
    denom = X_transformed.max(axis=0) - X_transformed.min(axis=0)
    denom[denom == 0] = 1
    X_scaled = (X_transformed - X_transformed.min(axis=0)) / denom
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), Y

def ZScoreNormalizationNorm(X, Y):
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X)
    denom = X_transformed.max(axis=0) - X_transformed.min(axis=0)
    denom[denom == 0] = 1
    X_scaled = (X_transformed - X_transformed.min(axis=0)) / denom
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), Y


algorithm_functions_normalization = {
    "MinMaxScalarNorm": MinMaxScalarNorm,
    "RobustScalarNorm": RobustScalarNorm,
    "ZScoreNormalizationNorm": ZScoreNormalizationNorm,
}
def normalization(X,Y):
    acc_holder = {}
    x_copy = X.copy()
    y_copy = Y.copy()
    best_algo = None
    best_accuracy = 0


    for name, func in algorithm_functions_normalization.items():
        print(name)
        print()

        X_Adjusted,Y_Adjusted = func(x_copy.copy(), y_copy.copy())

        accuracy = train_and_evaluate(X_Adjusted, Y_Adjusted,expand=False)
        acc_holder[name] = accuracy


        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_algo = name
    if best_algo:
        x_copy,y_copy = algorithm_functions_normalization[best_algo](x_copy.copy(), y_copy.copy())

    return x_copy, y_copy, best_algo, best_accuracy,acc_holder