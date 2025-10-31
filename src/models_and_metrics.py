from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# have this compatable with cross_val scoring names
METRIC_FUNCTIONS = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
    "roc_auc": lambda y_true, y_proba: roc_auc_score(y_true, y_proba),
}

MODELS = {
    # --- Linear Models ---
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    "SGDClassifier": SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
    
    # --- SVMs ---
    "LinearSVC": LinearSVC(random_state=42, max_iter=5000),
    #"SVC_rbf": SVC(kernel="rbf", probability=True, random_state=42),
    
    # --- Naive Bayes (for count/tfidf only) ---
    "MultinomialNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "BernoulliNB": BernoulliNB(),
    
    # --- Trees & Ensembles ---
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42, n_jobs=-1),
    "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
    
    # --- Neural ---
    "MLPClassifier": MLPClassifier(random_state=42, max_iter=300),
}

# Global incompatibility map
INCOMPATIBLE_MODELS = {
    "w2v": {"MultinomialNB", "ComplementNB"},
    "fasttext": {"MultinomialNB", "ComplementNB"},
    # Add more if needed later
}


# have this compatable with cross_val scoring names
SCORING_METRICS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
]

CV_STRATEGY = KFold(n_splits=5, shuffle=True, random_state=42)




def copy_models(models, embedding_type=None):
    """
    Returns a deep copy of the models dict.
    Removes incompatible models automatically based on embedding_type.
    """
    copied = {name: type(m)(**m.get_params()) for name, m in models.items()}
    
    if embedding_type and embedding_type in INCOMPATIBLE_MODELS:
        for bad_model in INCOMPATIBLE_MODELS[embedding_type]:
            copied.pop(bad_model, None)
    
    return copied
