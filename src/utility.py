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
