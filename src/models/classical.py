import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.metrics import calculate_metrics, plot_calibration_curve, plot_roc_curve


def _sigmoid(logits):
    """Numerically stable logistic sigmoid for arbitrary finite logits."""
    logits = np.asarray(logits, dtype=float)
    positive = logits >= 0
    probabilities = np.empty_like(logits)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    return probabilities


def _add_shift_aware_features(X):
    """Engineer shift-invariant temperature difference and mechanical interactions."""
    X = X.copy()
    air = X["Air temperature [K]"]
    process = X["Process temperature [K]"]
    torque = X["Torque [Nm]"]
    wear = X["Tool wear [min]"]
    speed = X["Rotational speed [rpm]"]
    X["Temperature difference [K]"] = process - air
    X["Torque x wear"] = torque * wear
    X["Torque x speed"] = torque * speed
    return X.drop(columns=["Air temperature [K]", "Process temperature [K]"])


def fit_temperature_scaling(y_true, logits, bounds=(0.05, 100.0)):
    """Fit a scalar post-hoc temperature T > 0 minimizing validation NLL."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    z = np.asarray(logits, dtype=float).reshape(-1)
    if len(y) != len(z):
        raise ValueError("y_true and logits must have the same length")

    def nll(temperature):
        scaled = z / temperature
        return float(np.mean(np.logaddexp(0.0, scaled) - y * scaled))

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x)


def estimate_importance_weights(
    X_source,
    X_target,
    random_state=42,
    clip_bounds=(0.1, 20.0),
):
    """Estimate covariate-shift density ratios p_target(x) / p_source(x) using unlabeled features."""
    source_encoded = pd.get_dummies(X_source).astype("float64")
    target_encoded = (
        pd.get_dummies(X_target)
        .astype("float64")
        .reindex(columns=source_encoded.columns, fill_value=0.0)
    )

    X_domain = np.vstack([source_encoded.to_numpy(), target_encoded.to_numpy()])
    y_domain = np.concatenate([
        np.zeros(len(source_encoded), dtype=int),
        np.ones(len(target_encoded), dtype=int),
    ])

    base_scaler = StandardScaler()
    X_scaled = base_scaler.fit_transform(X_domain)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    poly_scaler = StandardScaler()
    X_poly_scaled = poly_scaler.fit_transform(X_poly)

    domain_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    domain_clf.fit(X_poly_scaled, y_domain)

    source_domain_prob = domain_clf.predict_proba(X_poly_scaled[: len(source_encoded)])[:, 1]
    source_domain_prob = np.clip(source_domain_prob, 1e-4, 1.0 - 1e-4)

    prior_ratio = len(source_encoded) / len(target_encoded)
    weights = (source_domain_prob / (1.0 - source_domain_prob)) * prior_ratio
    weights = np.clip(weights, clip_bounds[0], clip_bounds[1])
    weights = weights / np.mean(weights)
    return weights


class GaussianNBModel:
    """Generative baseline: Gaussian Naive Bayes over standardized numeric + one-hot features."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.model = GaussianNB(var_smoothing=var_smoothing)

    def _encode(self, X, fit=False):
        encoded = pd.get_dummies(X).astype("float64")
        if fit:
            self.feature_columns = encoded.columns
            values = encoded.to_numpy()
            self.mean = values.mean(axis=0)
            self.scale = values.std(axis=0)
            self.scale[self.scale == 0] = 1.0
        else:
            encoded = encoded.reindex(columns=self.feature_columns, fill_value=0.0)
            values = encoded.to_numpy(dtype="float64")
        return (values - self.mean) / self.scale

    def fit(self, X_train, y_train, sample_weight=None):
        x = self._encode(X_train, fit=True)
        y = np.asarray(y_train, dtype=int).reshape(-1)
        self.model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict_logits(self, X):
        x = self._encode(X, fit=False)
        log_probs = self.model.predict_log_proba(x)
        return log_probs[:, 1] - log_probs[:, 0]


class MixedNBModel:
    """Mixed Generative Naive Bayes: GaussianNB for numeric sensors + CategoricalNB for Type."""

    def __init__(self, var_smoothing=1e-9, alpha=1.0, categorical_col="Type"):
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.categorical_col = categorical_col
        self.gnb = GaussianNB(var_smoothing=var_smoothing)
        self.cnb = None

    def _split_and_encode(self, X, fit=False):
        num_df = X.drop(columns=[self.categorical_col]).astype("float64")
        cat_series = X[self.categorical_col].astype(str)

        if fit:
            self.numeric_columns = num_df.columns
            num_values = num_df.to_numpy()
            self.mean = num_values.mean(axis=0)
            self.scale = num_values.std(axis=0)
            self.scale[self.scale == 0] = 1.0
            self.categories = sorted(cat_series.unique().tolist())
            self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        else:
            num_df = num_df.reindex(columns=self.numeric_columns, fill_value=0.0)
            num_values = num_df.to_numpy(dtype="float64")

        x_num = (num_values - self.mean) / self.scale
        x_cat = cat_series.map(self.category_to_idx).fillna(0).to_numpy(dtype=int).reshape(-1, 1)
        return x_num, x_cat

    def fit(self, X_train, y_train, sample_weight=None):
        x_num, x_cat = self._split_and_encode(X_train, fit=True)
        y = np.asarray(y_train, dtype=int).reshape(-1)
        self.gnb.fit(x_num, y, sample_weight=sample_weight)
        self.cnb = CategoricalNB(alpha=self.alpha, min_categories=len(self.categories))
        self.cnb.fit(x_cat, y, sample_weight=sample_weight)
        return self

    def predict_logits(self, X):
        x_num, x_cat = self._split_and_encode(X, fit=False)
        log_p_num = self.gnb.predict_log_proba(x_num)
        log_p_cat = self.cnb.predict_log_proba(x_cat)
        prior = self.gnb.class_prior_
        prior_log_odds = np.log(prior[1] / prior[0])
        return (log_p_num[:, 1] - log_p_num[:, 0]) + (log_p_cat[:, 1] - log_p_cat[:, 0]) - prior_log_odds


class LogisticRegressionModel:
    """Discriminative linear baseline over standardized numeric + one-hot features."""

    def __init__(self, C=1.0, max_iter=1000, random_state=42, class_weight=None):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
        )

    def _transform_features(self, X):
        return X.copy()

    def _encode(self, X, fit=False):
        X_feat = self._transform_features(X)
        encoded = pd.get_dummies(X_feat).astype("float64")
        if fit:
            self.feature_columns = encoded.columns
            values = encoded.to_numpy()
            self.mean = values.mean(axis=0)
            self.scale = values.std(axis=0)
            self.scale[self.scale == 0] = 1.0
        else:
            encoded = encoded.reindex(columns=self.feature_columns, fill_value=0.0)
            values = encoded.to_numpy(dtype="float64")
        return (values - self.mean) / self.scale

    def fit(self, X_train, y_train, sample_weight=None):
        x = self._encode(X_train, fit=True)
        y = np.asarray(y_train, dtype=int).reshape(-1)
        self.model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict_logits(self, X):
        x = self._encode(X, fit=False)
        return self.model.decision_function(x)


class ShiftAwareLogisticRegression(LogisticRegressionModel):
    """Logistic Regression with shift-invariant temperature difference and torque interactions."""

    def _transform_features(self, X):
        return _add_shift_aware_features(X)


SUPPORTED_CLASSICAL_MODELS = (
    GaussianNBModel,
    MixedNBModel,
    LogisticRegressionModel,
    ShiftAwareLogisticRegression,
)


class ClassicalTrainer:
    """Train, calibrate, prior-adjust, and evaluate classical models with parity to NN Trainer."""

    def __init__(
        self,
        model_class,
        model_kwargs=None,
        random_state=42,
        results_dir=None,
    ):
        if model_class not in SUPPORTED_CLASSICAL_MODELS:
            supported = ", ".join(cls.__name__ for cls in SUPPORTED_CLASSICAL_MODELS)
            raise ValueError(f"ClassicalTrainer supports {supported} only")
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.random_state = random_state
        self.results_dir = (
            Path(results_dir) if results_dir else Path(__file__).resolve().parents[2] / "results"
        )
        self.model = None
        self.train_failure_rate = None
        self.temperature = 1.0

    def fit(self, X_train, y_train, sample_weight=None):
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float).reshape(-1)
            self.train_failure_rate = float(np.average(y, weights=weights))
        else:
            self.train_failure_rate = float(np.mean(y))

        kwargs = dict(self.model_kwargs)
        if issubclass(self.model_class, LogisticRegressionModel) and "random_state" not in kwargs:
            kwargs["random_state"] = self.random_state

        self.model = self.model_class(**kwargs)
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        self.temperature = 1.0
        return self

    def fit_temperature(self, X_val, y_val):
        """Fit temperature scaling parameter T on validation logits."""
        logits = self.predict_logits(X_val)
        self.temperature = fit_temperature_scaling(y_val, logits)
        return self.temperature

    def predict_logits(self, X):
        if self.model is None:
            raise RuntimeError("Call fit() before predict_logits().")
        return np.asarray(self.model.predict_logits(X), dtype=float).reshape(-1)

    def predict_proba(self, X, target_failure_rate=None, temperature=None):
        logits = self.predict_logits(X)
        temp = 1.0 if temperature is None else float(temperature)
        if temp <= 0:
            raise ValueError("temperature must be positive")
        logits = logits / temp

        if target_failure_rate is not None:
            if not 0 < target_failure_rate < 1:
                raise ValueError("target_failure_rate must be between 0 and 1")
            train_rate = self.train_failure_rate
            if not 0 < train_rate < 1:
                raise ValueError("Training data must contain both failure labels for prior correction")
            prior_adjustment = np.log(target_failure_rate / (1.0 - target_failure_rate))
            prior_adjustment -= np.log(train_rate / (1.0 - train_rate))
            logits = logits + prior_adjustment

        return _sigmoid(logits)

    def evaluate(
        self,
        X_test,
        y_test,
        shift_name="iid",
        target_failure_rate=None,
        temperature=None,
        metrics_dir=None,
        plots_dir=None,
        threshold=None,
        threshold_tag="th_change",
        save_default=True,
        save_plots=True,
        file_suffix="",
    ):
        probabilities = self.predict_proba(
            X_test,
            target_failure_rate=target_failure_rate,
            temperature=temperature,
        )
        scores = calculate_metrics(y_test, probabilities)
        if temperature is not None:
            scores["temperature"] = float(temperature)
        if target_failure_rate is not None:
            scores["target_failure_rate"] = float(target_failure_rate)

        model_name = self.model_class.__name__
        base_name = f"{shift_name}{file_suffix}"

        if metrics_dir is None:
            output_dir = self.results_dir / shift_name
            output_path = output_dir / f"{model_name}{file_suffix}.json"
        else:
            output_dir = Path(metrics_dir) / model_name
            output_path = output_dir / f"{base_name}.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_default:
            with output_path.open("w") as result_file:
                json.dump(scores, result_file, indent=2)

        result_scores = scores
        if threshold is not None:
            threshold_scores = calculate_metrics(y_test, probabilities, threshold=threshold)
            threshold_scores["decision_threshold"] = float(threshold)
            if temperature is not None:
                threshold_scores["temperature"] = float(temperature)
            if target_failure_rate is not None:
                threshold_scores["target_failure_rate"] = float(target_failure_rate)
            threshold_path = output_dir / f"{base_name}_{threshold_tag}_{threshold:.3f}.json"
            with threshold_path.open("w") as result_file:
                json.dump(threshold_scores, result_file, indent=2)
            result_scores = threshold_scores

        if plots_dir is None:
            plot_dir = self.results_dir / shift_name
            plot_prefix = f"{model_name}{file_suffix}"
        else:
            plot_dir = Path(plots_dir) / model_name
            plot_prefix = base_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        if save_plots and scores["roc_auc"] is not None:
            plot_roc_curve(
                y_test,
                probabilities,
                plot_dir / f"{plot_prefix}_roc.svg",
                title=f"{model_name} ROC Curve ({base_name})",
            )
        if save_plots:
            plot_calibration_curve(
                y_test,
                probabilities,
                plot_dir / f"{plot_prefix}_calibration.svg",
                title=f"{model_name} Calibration ({base_name})",
            )
        return result_scores
