import joblib
import cloudpickle
import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
with open("clean_data.pkl", "rb") as f:
    imputer = cloudpickle.load(f)

all_mappings = joblib.load("all_mapping.pkl")


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

x_new = pd.DataFrame([{
    "FileNo": 1,
    "Age": 50,
    "Sex": 2,
    "Smoking_Status": 1,
    "BMI": 31.2025636700961,
    "WC": 87,
    "SBP": 120,
    "DBP": 80,
    "AnxietyScore": 13,
    "DepressionScore": 10,
    "LDL": 788887.4,
    "Glucose": 70,
    "Uric_Acid": 4.1,
    "Cholesterol": 136,
    "hs_CRP": 1.89,
    "HDL": 47,
    "TG": 58,
    "PAL": 2.03529915544114,
    "PAB":1.60919809860742,
    "GFR":107.308219178082,
    "Nightly_Sleep": 7,
    "CAD_Total":0,
    "DurationC1":129,
    "phase2":1

}])

for col, mapping in all_mappings.items():
    if col in x_new.columns:
        x_new[col] = x_new[col].map(mapping)

x_new_imputed= x_new
available_features = [col for col in features if col in x_new_imputed.columns]
x_new_selected = x_new_imputed[available_features]

y_pred = model.predict(x_new_selected)
confidences = get_confidence(model, x_new_selected)

for i, (pred, conf) in enumerate(zip(y_pred, confidences)):
    print(f"Sample {i+1} → Prediction: {pred}, Confidence: {conf:.2%}")
