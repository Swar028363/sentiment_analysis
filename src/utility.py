import ast
import pickle

import numpy as np
import pandas as pd

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
