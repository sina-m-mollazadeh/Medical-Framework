from sklearn.model_selection import train_test_split #Weak Model Test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #Weak Model Test
from sklearn.tree import DecisionTreeClassifier #Weak Model Test
from sklearn.neighbors import KNeighborsClassifier #Weak Model Test

def train_and_evaluate(X, y, expand=False):
    """Trains multiple weak models (3x KNN and 3x Decision Tree) and evaluates them."""

    if len(y) <= 5 or len(X) <= 5:
        return 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = [
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=7),

        DecisionTreeClassifier(max_depth=3, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        DecisionTreeClassifier(max_depth=None, min_samples_split=10, random_state=42),

    ]

    all_scores = []

    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted')
        coefs_scores=[1,1,1,1]
        all_scores.extend([acc*coefs_scores[0], prec*coefs_scores[1], rec*coefs_scores[2], f1*coefs_scores[3]])

        if expand:
            print(f"\nModel {i+1}: {model.__class__.__name__}")
            print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

    weighted_score = sum(all_scores) / len(all_scores)

    return weighted_score
