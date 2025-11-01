import ast
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from copy import deepcopy


def load_data_csv(path: str) -> pd.DataFrame:
    """Load CSV from interim data folder."""
    return pd.read_csv(path)

def save_data_csv(df: pd.DataFrame, path: str) -> None:
    """Save dataframe to interim data folder."""
    df.to_csv(path, index=False)

def save_pickle(obj, path: str) -> None:
    """Serialize Python object to pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    """Load serialized Python object."""
    with open(path, "rb") as f:
        return pickle.load(f)

def combine_results(dfs, names):
    """
    Combine multiple model result DataFrames and tag each with embedding name.
    """
    combined = []
    for df, name in zip(dfs, names):
        df = df.copy()
        df["Embedding"] = name
        df["Model"] = df["Model"].astype(str) + f"_{name}"
        combined.append(df)
    return pd.concat(combined, ignore_index=True)

def combine_model_dicts(model_dicts, names=None, prefix=False):
    combined = {}

    for i, models in enumerate(model_dicts):
        embed_name = names[i] if names and i < len(names) else f"Set{i+1}"

        for model_name, model_obj in models.items():
            if prefix:
                new_name = f"{embed_name}_{model_name}"
            else:
                new_name = f"{model_name}_{embed_name}"

            combined[new_name] = deepcopy(model_obj)

    return combined


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

def fix_config_types(d):
    """
    Recursively fix types in a dict loaded from YAML/JSON.
    Handles:
      - None ('None', 'none', 'NULL', etc.) -> None
      - Booleans ('True', 'FALSE', etc.) -> True/False
      - Numeric strings -> int/float
      - Tuple/list strings -> tuple/list
      - Nested dicts/lists
    """
    none_values = {'none', 'null', ''}
    true_values = {'true'}
    false_values = {'false'}

    for k, v in d.items():
        if isinstance(v, dict):
            fix_config_types(v)
        elif isinstance(v, str):
            v_lower = v.strip().lower()
            # None
            if v_lower in none_values:
                d[k] = None
            # Booleans
            elif v_lower in true_values:
                d[k] = True
            elif v_lower in false_values:
                d[k] = False
            else:
                # Try parsing numbers, tuples, lists, etc.
                try:
                    d[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    # keep as string if parsing fails
                    pass
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], dict):
                    fix_config_types(v[i])
                elif isinstance(v[i], str):
                    v_lower = v[i].strip().lower()
                    if v_lower in none_values:
                        v[i] = None
                    elif v_lower in true_values:
                        v[i] = True
                    elif v_lower in false_values:
                        v[i] = False
                    else:
                        try:
                            v[i] = ast.literal_eval(v[i])
                        except (ValueError, SyntaxError):
                            pass
