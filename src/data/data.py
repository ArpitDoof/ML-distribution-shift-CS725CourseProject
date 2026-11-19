import numpy as np
import pandas as pd
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO


class DataLoader:
    """Download and cache the AI4I dataset locally."""

    URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]
        self.data_path = project_root / "data" / "ai4i2020.csv"
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.data_path.exists():
            with urlopen(self.URL) as response:
                with ZipFile(BytesIO(response.read())) as archive:
                    self.data_path.write_bytes(archive.read("ai4i2020.csv"))

def _get_product_shift(X_test, y_test, random_state):
    """Resample the test set to give L, M, and H products equal representation."""
    counts = X_test["Type"].value_counts()
    weights = X_test["Type"].map(1 / counts).to_numpy()
    weights /= weights.sum()
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X_test), size=len(X_test), replace=True, p=weights)
    return X_test.iloc[indices].reset_index(drop=True), y_test.iloc[indices].reset_index(drop=True)


def _get_joint_shift(X_test, y_test, random_state):
    """Resample the test set to favor higher torque-speed combinations."""
    load = X_test["Torque [Nm]"].to_numpy() * X_test["Rotational speed [rpm]"].to_numpy()
    z_load = (load - load.mean()) / load.std()
    weights = np.exp(np.clip(z_load, -5, 5))
    weights /= weights.sum()
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X_test), size=len(X_test), replace=True, p=weights)
    return X_test.iloc[indices].reset_index(drop=True), y_test.iloc[indices].reset_index(drop=True)


def get(type="iid", test_size=0.2, random_state=42, temp_delta=2.0):
    """Return train/test data, optionally with a warmer-temperature test shift."""
    data_path = DataLoader().data_path
    data = pd.read_csv(data_path)

    y = data["Machine failure"]
    X = data.drop(
        columns=["UID", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    )

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(data))
    split = int(len(data) * (1 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    if type == "temp":
        X_test = _get_temp_shift(X_test, temp_delta)
    elif type == "wear":
        X_test, y_test = _get_wear_shift(X_test, y_test, random_state)
    elif type == "product":
        X_test, y_test = _get_product_shift(X_test, y_test, random_state)
    elif type == "joint":
        X_test, y_test = _get_joint_shift(X_test, y_test, random_state)
    elif type not in ("", "iid"):
        raise ValueError(f"Unknown data type: {type}")

    return X_train, X_test, y_train, y_test

def _get_temp_shift(X_test, delta):
    """Raise both test temperature features by the same amount in kelvin."""
    X_test = X_test.copy()
    X_test["Air temperature [K]"] += delta
    X_test["Process temperature [K]"] += delta
    return X_test


def _get_wear_shift(X_test, y_test, random_state):
    """Resample the test set to favor rows with higher tool wear."""
    wear = X_test["Tool wear [min]"].to_numpy()
    z_wear = (wear - wear.mean()) / wear.std()
    weights = np.exp(np.clip(z_wear, -5, 5))
    weights /= weights.sum()

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X_test), size=len(X_test), replace=True, p=weights)
    return X_test.iloc[indices].reset_index(drop=True), y_test.iloc[indices].reset_index(drop=True)
