import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.metrics import calculate_metrics, plot_calibration_curve, plot_roc_curve


class DeepNN(nn.Module):
    """A deeper baseline with three hidden layers."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


class WideNN(nn.Module):
    """A one-hidden-layer baseline with more neurons."""

    def __init__(self, input_dim, hidden_dim=128):
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


class ShiftAwareNN(nn.Module):
    """Use product-specific heads over shared, shift-aware features."""

    def __init__(self, input_dim, type_indices, hidden_dim=64, dropout=0.2):
        super().__init__()
        if len(type_indices) != 3:
            raise ValueError("ShiftAwareNN needs one-hot columns for Type L, M, and H")
        self.type_indices = type_indices
        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.product_heads = nn.ModuleList(
            [nn.Linear(hidden_dim // 2, 1) for _ in range(3)]
        )

    def forward(self, x):
        shared_features = self.shared(x)
        outputs = torch.cat([head(shared_features) for head in self.product_heads], dim=1)
        product_ids = torch.argmax(x[:, self.type_indices], dim=1)
        product_weights = torch.nn.functional.one_hot(product_ids, num_classes=3).to(x.dtype)
        return (outputs * product_weights).sum(dim=1)


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
        device="cpu",
    ):
        if model_class not in (DeepNN, WideNN, ShiftRobustNN, ShiftAwareNN):
            raise ValueError("Trainer supports DeepNN, WideNN, ShiftRobustNN, and ShiftAwareNN only")
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.results_dir = Path(results_dir) if results_dir else Path(__file__).resolve().parents[2] / "results"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but is not available")
            if self.device.index is None or self.device.index >= torch.cuda.device_count():
                raise RuntimeError(f"Requested {self.device}, but only {torch.cuda.device_count()} CUDA device(s) are available")
        self.model = None

    def _encode(self, X, fit=False):
        X = X.copy()
        if self.model_class is ShiftAwareNN:
            air = X["Air temperature [K]"]
            process = X["Process temperature [K]"]
            torque = X["Torque [Nm]"]
            wear = X["Tool wear [min]"]
            speed = X["Rotational speed [rpm]"]
            X["Temperature difference [K]"] = process - air
            X["Torque x wear"] = torque * wear
            X["Torque x speed"] = torque * speed
            X = X.drop(columns=["Air temperature [K]", "Process temperature [K]"])

        encoded = pd.get_dummies(X).astype("float32")
        if fit:
            self.feature_columns = encoded.columns
            values = encoded.to_numpy()
            self.mean = values.mean(axis=0)
            self.scale = values.std(axis=0)
            self.scale[self.scale == 0] = 1.0
            if self.model_class is ShiftAwareNN:
                self.type_indices = [
                    index for index, name in enumerate(self.feature_columns)
                    if name.startswith("Type_")
                ]
        else:
            encoded = encoded.reindex(columns=self.feature_columns, fill_value=0)
            values = encoded.to_numpy(dtype="float32")
        return (values - self.mean) / self.scale

    def fit(self, X_train, y_train):
        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        x = torch.tensor(self._encode(X_train, fit=True), dtype=torch.float32, device=self.device)
        y = torch.tensor(
            np.asarray(y_train, dtype="float32").reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        self.train_failure_rate = float(y.mean())

        model_kwargs = dict(self.model_kwargs)
        if self.model_class is ShiftAwareNN:
            model_kwargs["type_indices"] = self.type_indices
        self.model = self.model_class(input_dim=x.shape[1], **model_kwargs).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            indices = torch.randperm(len(x), device=self.device)
            self.model.train()
            for start in range(0, len(x), self.batch_size):
                batch = indices[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.model(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
        return self

    def evaluate(
        self,
        X_test,
        y_test,
        shift_name="iid",
        target_failure_rate=None,
        metrics_dir=None,
        plots_dir=None,
        threshold=None,
        threshold_tag="th_change",
        save_default=True,
        save_plots=True,
    ):
        probabilities = self.predict_proba(X_test, target_failure_rate)
        scores = calculate_metrics(y_test, probabilities)
        model_name = self.model_class.__name__
        if metrics_dir is None:
            output_dir = self.results_dir / shift_name
            output_path = output_dir / f"{model_name}.json"
        else:
            output_dir = Path(metrics_dir) / model_name
            output_path = output_dir / f"{shift_name}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_default:
            with output_path.open("w") as result_file:
                json.dump(scores, result_file, indent=2)

        result_scores = scores
        if threshold is not None:
            threshold_scores = calculate_metrics(y_test, probabilities, threshold=threshold)
            threshold_scores["decision_threshold"] = float(threshold)
            threshold_path = output_dir / f"{shift_name}_{threshold_tag}_{threshold:.3f}.json"
            with threshold_path.open("w") as result_file:
                json.dump(threshold_scores, result_file, indent=2)
            result_scores = threshold_scores

        if plots_dir is None:
            plot_dir = self.results_dir / shift_name
            plot_prefix = model_name
        else:
            plot_dir = Path(plots_dir) / model_name
            plot_prefix = shift_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        if save_plots and scores["roc_auc"] is not None:
            plot_roc_curve(
                y_test,
                probabilities,
                plot_dir / f"{plot_prefix}_roc.svg",
                title=f"{model_name} ROC Curve ({shift_name})",
            )
        if save_plots:
            plot_calibration_curve(
                y_test,
                probabilities,
                plot_dir / f"{plot_prefix}_calibration.svg",
                title=f"{model_name} Calibration ({shift_name})",
            )
        return result_scores

    def predict_proba(self, X, target_failure_rate=None):
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")

        x = torch.tensor(self._encode(X), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            if target_failure_rate is not None:
                if not 0 < target_failure_rate < 1:
                    raise ValueError("target_failure_rate must be between 0 and 1")
                train_rate = self.train_failure_rate
                if not 0 < train_rate < 1:
                    raise ValueError("Training data must contain both failure labels for prior correction")
                prior_adjustment = np.log(target_failure_rate / (1 - target_failure_rate))
                prior_adjustment -= np.log(train_rate / (1 - train_rate))
                logits += prior_adjustment
            return torch.sigmoid(logits).cpu().numpy()
