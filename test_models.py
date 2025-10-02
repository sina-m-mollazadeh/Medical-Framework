from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
import numpy as np

def train_and_evaluate(X, y, expand=False):
    """Evaluate multiple weak models using StratifiedKFold CV and stricter metrics."""

    if len(y) <= 5 or len(X) <= 5:
        return 0

    models = [
        DummyClassifier(strategy="most_frequent"),   # Baseline
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=7),
        DecisionTreeClassifier(max_depth=1, random_state=42),  # Intentionally weak
        DecisionTreeClassifier(max_depth=3, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        DecisionTreeClassifier(max_depth=None, min_samples_split=10, random_state=42),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_scores = []

    for i, model in enumerate(models):
        # Cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=cv)

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y, y_pred, average='macro', zero_division=0)
        f1   = f1_score(y, y_pred, average='macro', zero_division=0)

        # collect all scores equally
        coefs_scores = [1,1,1,1]
        all_scores.extend([acc*coefs_scores[0], prec*coefs_scores[1],
                           rec*coefs_scores[2], f1*coefs_scores[3]])

        if expand:
            print(f"\nModel {i+1}: {model.__class__.__name__}")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall   : {rec:.4f}")
            print(f"  F1-score : {f1:.4f}")

    weighted_score = np.mean(all_scores)

    return weighted_score
