import joblib
import cloudpickle
import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid

# === Load Saved Artifacts ===
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

with open("clean_data.pkl", "rb") as f:
    imputer = cloudpickle.load(f)

all_mappings = joblib.load("all_mapping.pkl")

# === Confidence Helper ===
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

# === Your New Input Sample(s) ===
# Example: One row input
# You can replace this with any df you receive
x_new = pd.DataFrame([{
    "FileNo": 1,
    "event": 2,
    "Sex": 2,
    "age_1": 10,
    "Glocuse": 30,
    "HDL": 2,
    "TRIGLYCERIDES": 58,
    "LdL": 10,
    "Cholesterol": 0,
    "BPS": 0,
    "BPD": 0,
    "HSCRP": 1900
}])

# === Apply Mappings (Label Encoding) ===
for col, mapping in all_mappings.items():
    if col in x_new.columns:
        x_new[col] = x_new[col].map(mapping)

# === Impute Missing Values ===
x_new_imputed = imputer.transform(x_new) if imputer != None else x_new

# === Select Only Selected Features ===
x_new_selected = x_new_imputed[features]

# === Predict & Confidence ===
y_pred = model.predict(x_new_selected)
confidences = get_confidence(model, x_new_selected)
# === Output ===
for i, (pred, conf) in enumerate(zip(y_pred, confidences)):
    print(f"Sample {i+1} → Prediction: {pred}, Confidence: {conf:.2%}")
