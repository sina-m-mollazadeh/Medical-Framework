from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_and_evaluate(X, y, weights=None, expand=False):
    """
    Evaluate multiple models using ROC-AUC.
    0.50 is random chance, 1.0 is perfect separation.
    """
    if len(y) <= 5 or len(X) <= 5:
        return 0
    
    num_classes = len(np.unique(y))
    
    models = [
        DummyClassifier(strategy="stratified", random_state=42),  
        make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)),
        make_pipeline(StandardScaler(), SVC(class_weight='balanced', C=0.1, max_iter=1000, random_state=42, probability=True)),
        DecisionTreeClassifier(class_weight='balanced', max_depth=3, random_state=42),   
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        ExtraTreesClassifier(class_weight='balanced', n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    ]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_scores = []
    model_details = {}
    
    for model in models:
        try:
            if num_classes == 2:
                # Use ROC-AUC for binary clinical data
                y_prob = cross_val_predict(model, X, y, cv=cv, n_jobs=-1, method='predict_proba')[:, 1]
                score = roc_auc_score(y, y_prob)
            else:
                # Fallback to macro F1 for multi-class
                y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
                score = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            model_scores.append(score)
            
            if expand:
                name = str(model.steps[-1][1]).split('(')[0] if hasattr(model, 'steps') else str(model).split('(')[0]
                model_details[name] = {'Score': score}
                
        except Exception as e:
            continue
    
    if not model_scores:
        return 0
    
    final_score = np.max(model_scores)
    
    if expand:
        print("\n" + "="*40)
        print(f"PREPROCESSING EVAL (Max ROC-AUC: {final_score:.4f})")
        print("="*40)
        best_name = max(model_details, key=lambda k: model_details[k]['Score'])
        print(f"Best Model: {best_name} -> Score: {model_details[best_name]['Score']:.4f}")
        print("="*40)
    
    return final_score