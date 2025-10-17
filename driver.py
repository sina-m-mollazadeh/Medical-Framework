import joblib
import cloudpickle
import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid
from loader import process_date_columns,sanitize_column_names
import warnings
import re

verbose=False
if(not verbose):
    warnings.filterwarnings('ignore', 
                           message='.*Your system has an old version of glibc.*')

    warnings.filterwarnings('ignore', 
                           message='.*The argument \'infer_datetime_format\' is deprecated.*')

base="healthcare/"
model = joblib.load(base+"model.pkl")
features = joblib.load(base+"features.pkl")
with open(base+"clean_data.pkl", "rb") as f:
    imputer = cloudpickle.load(f)

all_mappings = joblib.load(base+"all_mapping.pkl")
y_mappings = joblib.load(base+"y_mappings.pkl")


def get_confidence(model, x_input, num_classes=2):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_input)
        return np.max(probs, axis=1)
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(x_input)
        if num_classes == 2:
            return sigmoid(raw)
        else:
            return np.ones(len(x_input)) * -1
    else:
        return np.ones(len(x_input)) * -1

x_new = pd.read_csv(base+"healthcare_test.csv")


x_new=sanitize_column_names(x_new,y_column="")

x_new=process_date_columns(x_new)
for col, mapping in all_mappings.items():
    if col in x_new.columns:
        x_new[col] = x_new[col].map(mapping)

x_new_imputed= x_new
available_features = [col for col in features if col in x_new_imputed.columns]
x_new_selected = x_new_imputed[available_features]

y_pred = model.predict(x_new_selected)
confidences = get_confidence(model, x_new_selected)
reverse_mapping = {v: k for k, v in y_mappings['Test Results'].items()}

# Map the predictions
y_pred_mapped = [reverse_mapping[pred] for pred in y_pred]
# for i, (pred, conf) in enumerate(zip(y_pred, confidences)):
#     print(f"Sample {i+1} → Prediction: {pred}, Confidence: {conf:.2%}")

for i, (pred, conf) in enumerate(zip(y_pred_mapped, confidences)):
    print(f"Sample {i+1} → Prediction: {pred}, Confidence: {conf:.2%}")