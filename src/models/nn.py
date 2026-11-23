import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.metrics import calculate_metrics


class SimpleNN(nn.Module):
    """A small one-hidden-layer neural network baseline."""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


class ShiftRobustNN(nn.Module):
    """A regularized MLP that trains with feature and hidden-unit dropout."""

    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


class Trainer:
    """Train one of this module's neural networks and save its metrics."""

    def __init__(
        self,
        model_class,
        model_kwargs=None,
        learning_rate=0.001,
        epochs=50,
        batch_size=64,
        weight_decay=0.0001,
        random_state=42,
        results_dir=None,
    ):
        if model_class not in (SimpleNN, ShiftRobustNN):
            raise ValueError("Trainer supports SimpleNN and ShiftRobustNN only")
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.results_dir = Path(results_dir) if results_dir else Path(__file__).resolve().parents[2] / "results"
        self.model = None

    def _encode(self, X, fit=False):
        encoded = pd.get_dummies(X).astype("float32")
        if fit:
            self.feature_columns = encoded.columns
            values = encoded.to_numpy()
            self.mean = values.mean(axis=0)
            self.scale = values.std(axis=0)
            self.scale[self.scale == 0] = 1.0
        else:
            encoded = encoded.reindex(columns=self.feature_columns, fill_value=0)
            values = encoded.to_numpy(dtype="float32")
        return (values - self.mean) / self.scale

    def fit(self, X_train, y_train):
        torch.manual_seed(self.random_state)
        x = torch.tensor(self._encode(X_train, fit=True), dtype=torch.float32)
        y = torch.tensor(np.asarray(y_train, dtype="float32").reshape(-1), dtype=torch.float32)

        self.model = self.model_class(input_dim=x.shape[1], **self.model_kwargs)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            indices = torch.randperm(len(x))
            self.model.train()
            for start in range(0, len(x), self.batch_size):
                batch = indices[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.model(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
        return self

    def evaluate(self, X_test, y_test, shift_name="iid"):
        if self.model is None:
            raise RuntimeError("Call fit() before evaluate().")

        x = torch.tensor(self._encode(X_test), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(x)).numpy()

        scores = calculate_metrics(y_test, probabilities)
        output_dir = self.results_dir / shift_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.model_class.__name__}.json"
        with output_path.open("w") as result_file:
            json.dump(scores, result_file, indent=2)
        return scores
