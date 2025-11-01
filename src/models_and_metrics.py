from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    BaggingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, StratifiedKFold
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


BASE_LEARNERS = [
    # --- Linear Models (excellent for text data) ---
    ('logreg', Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])),

    ('sgd', Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', SGDClassifier(loss='log_loss', max_iter=2000, random_state=42))
    ])),

    # --- Nonlinear Kernel Model ---
    ('svc_rbf', Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42))
    ])),

    # --- Probabilistic Baseline ---
    ('bernoulli_nb', BernoulliNB()),

    # --- Ensemble Trees (robust, interpretability) ---
    ('rf', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)),

    # --- Neural Network (captures complex nonlinearities) ---
    ('mlp', Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))
    ]))
]


# === Models ===
MODELS = {
    # --- Linear / probabilistic ---
    "LogisticRegression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "SGDClassifier": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SGDClassifier(loss='log_loss', max_iter=2000, random_state=42))
    ]),
    "RidgeClassifier": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RidgeClassifier(random_state=42))
    ]),
    "PassiveAggressive": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', PassiveAggressiveClassifier(max_iter=2000, random_state=42))
    ]),
    
    # --- SVM family ---
    "LinearSVC": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(max_iter=5000, random_state=42))
    ]),
    "SVC_RBF": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', random_state=42))
    ]),

    # --- Naive Bayes ---
    "MultinomialNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "BernoulliNB": BernoulliNB(),

    # --- Tree-based / Ensemble methods ---
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "Bagging(KNN)": BaggingClassifier(
        estimator=KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        n_estimators=25, random_state=42, n_jobs=-1
    ),

    # --- Neural Network ---
    "MLP": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))
    ]),

    # --- Instance-based ---
    "KNN": Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=7, n_jobs=-1))
    ]),

    # --- Meta Models ---
    "StackingClassifier": StackingClassifier(
        estimators=BASE_LEARNERS,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_jobs=-1
    ),
    "VotingClassifier": VotingClassifier(
        estimators=BASE_LEARNERS,
        voting='soft',
        n_jobs=-1
    )
}

# Global incompatibility map
INCOMPATIBLE_MODELS = {
    # Dense embeddings (negative or continuous values)
    "w2v": {
        "MultinomialNB", "ComplementNB",  # NB fails on negative or continuous inputs
    },
    "fast_text": {
        "MultinomialNB", "ComplementNB",  # same as above
    },

    # Sparse high-dimensional embeddings (TF-IDF, Count)
    "tfidf": {
        "KNN", "DecisionTree", "RandomForest", "ExtraTrees",
        "GradientBoosting", "Bagging(LogReg)", "MLP",  # prone to overfit or fail
    },
    "count": {
        "KNN", "DecisionTree", "RandomForest", "ExtraTrees",
        "GradientBoosting", "Bagging(LogReg)", "MLP",
    },
}


# have this compatable with cross_val scoring names
SCORING_METRICS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
]

CV_STRATEGY = KFold(n_splits=5, shuffle=True, random_state=42)


def copy_models(models, embedding_type=None):
    """
    Deep-copies the given models dict and removes incompatible models 
    based on the embedding type.
    """
    # make an actual deep copy
    copied = deepcopy(models)

    # Drop incompatible models for this embedding type
    if embedding_type and embedding_type in INCOMPATIBLE_MODELS:
        bad_models = INCOMPATIBLE_MODELS[embedding_type]
        for bad_model in bad_models:
            if bad_model in copied:
                copied.pop(bad_model)

        print(f"[INFO] Dropped {len(bad_models)} incompatible models for '{embedding_type}': {', '.join(bad_models)}")

    return copied

