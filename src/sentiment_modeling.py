import os
import time
import warnings

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_validate
from sklearn.preprocessing import label_binarize
from sklearn.base import clone
from sklearn.metrics import *

from .models_and_metrics import METRIC_FUNCTIONS

def evaluate_metrics(y_true, y_pred, y_proba, metrics):
    """Dynamically evaluate any combination of metrics."""
    results = {}
    for metric in metrics:
        metric = metric.lower()
        func = METRIC_FUNCTIONS.get(metric)
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


def get_predictions_with_proba(model, X):
    """Get predictions and probabilities from model."""
    y_pred = model.predict(X)
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X)
    else:
        y_proba = None
    
    return y_pred, y_proba
    
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
            scoring = {m: m for m in scoring_metrics if m in METRIC_FUNCTIONS.keys()}

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
            model_data["Fit_Time_sec"] = np.sum(scores['fit_time'])

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
            y_train_pred, y_train_proba = get_predictions_with_proba(model, X_train)
            y_test_pred, y_test_proba = get_predictions_with_proba(model, X_test)
            

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


def combine_results(dfs, names):
    """
    Combine multiple model result DataFrames and tag each with embedding name.
    """
    combined = []
    for df, name in zip(dfs, names):
        df = df.copy()
        df["Embedding"] = name
        df["Model"] = df["Model"].astype(str) + f"{name}"
        combined.append(df)
    return pd.concat(combined, ignore_index=True)

def combine_model_dicts(model_dicts, names=None, prefix=True):
    combined = {}

    for i, models in enumerate(model_dicts):
        embed_name = names[i] if names and i < len(names) else f"Set{i+1}"

        for model_name, model_obj in models.items():
            if prefix:
                new_name = f"{embed_name}{model_name}"
            else:
                new_name = f"{model_name}{embed_name}"

            combined[new_name] = clone(model_obj)

    return combined


def find_best_models(df, scoring_func=None, top_n=5):
    """
    Evaluate and rank models across embeddings using all available metrics.
    User can override the default scoring formula with scoring_func(df).
    
    Args:
        df: Combined DataFrame with metrics
        scoring_func: Optional custom scoring lambda or function
        top_n: Number of top models to return
    Returns:
        Ranked DataFrame of best models
    """
    df = df.copy()

    # Default comprehensive scoring formula using all metrics
    if scoring_func is None:
        # Normalize fit time (0-1 scale; lower = better)
        df["Fit_Time_norm"] = (df["Fit_Time_sec"] - df["Fit_Time_sec"].min()) / (
            df["Fit_Time_sec"].max() - df["Fit_Time_sec"].min()
        )
    
        # --- Derived metrics ---
        df["Overfit_Acc"] = abs(df["Train_Accuracy"] - df["Test_Accuracy"])
        df["Overfit_F1"] = abs(df["Train_F1"] - df["Test_F1"])
        df["Overfit_Precision"] = abs(df["Train_Precision"] - df["Test_Precision"])
        df["Overfit_Recall"] = abs(df["Train_Recall"] - df["Test_Recall"])
        df["Overfit_Roc"] = abs(df["Train_Roc_auc"] - df["Test_Roc_auc"])
        df["PR_Balance"] = abs(df["Test_Precision"] - df["Test_Recall"])
        
        df["Overall_Score"] = (
            # Performance
            0.15 * df["Test_Accuracy"]
            + 0.15 * df["Test_F1"]
            + 0.15 * df["Test_Roc_auc"]
            + 0.1 * df["Test_Precision"]
            + 0.1 * df["Test_Recall"]

            # Generalization (train-test consistency)
            + 0.05 * (1 - df["Overfit_Acc"])
            + 0.05 * (1 - df["Overfit_F1"])
            + 0.05 * (1 - df["Overfit_Precision"])
            + 0.05 * (1 - df["Overfit_Recall"])
            + 0.05 * (1 - df["Overfit_Roc"])

            # Stability and efficiency
            + 0.05 * (1 - df["Fit_Time_norm"])
            + 0.05 * (1 - df["PR_Balance"])
        )
    else:
        # Allow user to supply custom scoring
        df["Overall_Score"] = scoring_func(df)

    # Sort by score
    ranked = df.sort_values("Overall_Score", ascending=False).reset_index(drop=True)

    return ranked.head(top_n)


def plot_model_performance(df, metric="Overall_Score", fig_size=(14, 14), save_fig_path=""):
    df_plot = df.sort_values(metric, ascending=False)

    plt.figure(figsize=fig_size)
    ax = sns.barplot(
        data=df_plot,
        x="Model",
        y=metric,
        hue="Embedding",
    )
    ax.set_xlabel(metric)
    plt.xticks(rotation=-45)
    ax.set_ylabel("Model")

    ax.set_title(f"Model Performance by {metric}", fontsize=16, fontweight="bold", pad=15)

    plt.legend(
        title="Embedding",
        bbox_to_anchor=(1.02, 1),
    )
    
    for container in ax.containers:
         ax.bar_label(container, fmt="%.3f")
        
    if save_fig_path != "":
        plt.savefig(save_fig_path)
        
    plt.show()













