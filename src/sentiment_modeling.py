import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Map metric names to sklearn functions
_METRIC_FUNCTIONS = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
    "roc_auc": lambda y_true, y_proba: roc_auc_score(y_true, y_proba),
}


def evaluate_metrics(y_true, y_pred, y_proba, metrics):
    """Dynamically evaluate any combination of metrics."""
    results = {}
    for metric in metrics:
        metric = metric.lower()
        func = _METRIC_FUNCTIONS.get(metric)
        if not func:
            raise ValueError(f"Unsupported metric: '{metric}'")

        # Some metrics require probabilities
        if metric in ["roc_auc"]:
            if y_proba is None:
                results[metric.capitalize()] = np.nan
            else:
                results[metric.capitalize()] = func(y_true, y_proba)
        else:
            results[metric.capitalize()] = func(y_true, y_pred)
    return results


def train_and_eval_models(
    data,
    models,
    cv_strategy=None,
    scoring_metrics=None,
    n_jobs=-1
):
    """
    Unified training + evaluation function.
    Supports both train/test and cross-validation modes.
    Allows dynamic choice of metrics.

    Args:
        data: tuple
            - (X_train, X_test, y_train, y_test) if cv_strategy is None
            - (X, y) if cv_strategy is provided
        models: dict of model name -> model instance
        cv_strategy: cross-validation splitter or None
        scoring_metrics: list of metrics to evaluate (default common set)
        n_jobs: parallel jobs for CV

    Returns:
        models, pd.DataFrame of evaluation results
    """

    if scoring_metrics is None:
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    results = []

    for name, model in models.items():
        print(f"Evaluating {name}...")

        if cv_strategy:  # ===== Cross-validation mode =====
            X, y = data

            # Convert custom metrics into sklearn-compatible scoring names
            scoring = {m: m for m in scoring_metrics if m in _METRIC_FUNCTIONS.keys()}

            scores = cross_validate(
                estimator=model,
                X=X,
                y=y,
                cv=cv_strategy,
                scoring=scoring,
                return_train_score=True,
                n_jobs=n_jobs
            )

            model_data = {"Model": name}
            model_data["Fit_Time_sec"] = np.mean(scores['fit_time'])

            # Aggregate train/test results
            for data_type in ("Train", "Test"):
                for metric in scoring_metrics:
                    key = f"{data_type}_{metric.capitalize()}"
                    score_key = f"{data_type.lower()}_{metric}"
                    if score_key in scores:
                        model_data[key] = np.mean(scores[score_key])
                    else:
                        model_data[key] = np.nan

            results.append(model_data)

        else:  # ===== Train/Test mode =====
            X_train, X_test, y_train, y_test = data
            start_time = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start_time

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Probabilities or decision scores
            if hasattr(model, "predict_proba"):
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_train_proba = model.decision_function(X_train)
                y_test_proba = model.decision_function(X_test)
            else:
                y_train_proba = y_test_proba = None

            model_data = {"Model": name, "Fit_Time_sec": fit_time}

            # Compute metrics for both train and test
            for dataset, y_true, y_pred, y_proba in [
                ("Train", y_train, y_train_pred, y_train_proba),
                ("Test", y_test, y_test_pred, y_test_proba)
            ]:
                evals = evaluate_metrics(y_true, y_pred, y_proba, scoring_metrics)
                for metric_name, metric_val in evals.items():
                    model_data[f"{dataset}_{metric_name}"] = metric_val

            results.append(model_data)

    df = pd.DataFrame(results)
    if 'Test_F1-Score' in df.columns:
        df = df.sort_values(by='Test_F1-Score', ascending=False)
    print("\nEvaluation Complete.")

    return models, df
