from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data import get
from src.metrics import (
    adjust_threshold_for_prior,
    find_best_f1_threshold,
    plot_calibration_comparison,
    plot_roc_comparison,
)
from src.models.nn import DeepNN, ShiftAwareNN, ShiftRobustNN, Trainer, WideNN


ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "metrics" / "nn"
PLOTS_DIR = ROOT / "plots" / "nn"
RANDOM_STATE = 42

SHIFTS = [
    ("temp_plus_2K", {"type": "temp"}, None),
    ("high_wear", {"type": "wear"}, None),
    ("balanced_product_mix", {"type": "product"}, None),
    ("high_joint_load", {"type": "joint"}, None),
    ("failure_rate_20pct", {"type": "label", "failure_rate": 0.2}, 0.2),
    (
        "torque_corruption_20pct",
        {"type": "corruption", "feature": "Torque [Nm]", "corruption_pct": 20},
        None,
    ),
]

MODELS = [DeepNN, WideNN, ShiftRobustNN, ShiftAwareNN]


def split_for_shift_adaptation(X, y, random_state):
    """Keep duplicate resampled rows together when making calibration/test halves."""
    frame = X.reset_index(drop=True).copy()
    frame["__target__"] = y.to_numpy()
    group_ids = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    groups = pd.DataFrame({"group": group_ids, "label": y.to_numpy()}).drop_duplicates("group")
    calibration_groups, _ = train_test_split(
        groups,
        test_size=0.5,
        random_state=random_state,
        stratify=groups["label"],
    )
    is_calibration = pd.Series(group_ids).isin(calibration_groups["group"]).to_numpy()
    return (
        X.iloc[is_calibration].reset_index(drop=True),
        y.iloc[is_calibration].reset_index(drop=True),
        X.iloc[~is_calibration].reset_index(drop=True),
        y.iloc[~is_calibration].reset_index(drop=True),
    )


def main():
    X_train, _, y_train, _ = get(type="iid", random_state=RANDOM_STATE)
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    test_sets = {}
    for shift_name, shift_args, target_failure_rate in SHIFTS:
        _, X_test, _, y_test = get(random_state=RANDOM_STATE, **shift_args)
        X_calibration, y_calibration, X_adapt_test, y_adapt_test = split_for_shift_adaptation(
            X_test, y_test, RANDOM_STATE
        )
        test_sets[shift_name] = {
            "X_test": X_test,
            "y_test": y_test,
            "target_failure_rate": target_failure_rate,
            "X_calibration": X_calibration,
            "y_calibration": y_calibration,
            "X_adapt_test": X_adapt_test,
            "y_adapt_test": y_adapt_test,
        }

    for model_class in MODELS:
        trainer = Trainer(
            model_class,
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            random_state=RANDOM_STATE,
            device="cuda:0",
        )
        trainer.fit(X_fit, y_fit)
        validation_probabilities = trainer.predict_proba(X_validation)
        threshold = find_best_f1_threshold(y_validation, validation_probabilities)
        comparison_curves = {}

        for shift_name, data in test_sets.items():
            X_test, y_test = data["X_test"], data["y_test"]
            target_failure_rate = data["target_failure_rate"]
            if model_class is not ShiftAwareNN:
                target_failure_rate = None
            shift_threshold = threshold
            if model_class is ShiftAwareNN and target_failure_rate is not None:
                shift_threshold = adjust_threshold_for_prior(
                    threshold,
                    trainer.train_failure_rate,
                    target_failure_rate,
                )
            scores = trainer.evaluate(
                X_test,
                y_test,
                shift_name=shift_name,
                target_failure_rate=target_failure_rate,
                metrics_dir=METRICS_DIR,
                plots_dir=PLOTS_DIR,
            )
            probabilities = trainer.predict_proba(X_test, target_failure_rate)
            comparison_curves[shift_name] = (y_test, probabilities)

            calibration_probabilities = trainer.predict_proba(
                data["X_calibration"], target_failure_rate
            )
            adapted_threshold = find_best_f1_threshold(
                data["y_calibration"], calibration_probabilities
            )
            source_holdout_scores = trainer.evaluate(
                data["X_adapt_test"],
                data["y_adapt_test"],
                shift_name=shift_name,
                target_failure_rate=target_failure_rate,
                metrics_dir=METRICS_DIR,
                threshold=shift_threshold,
                threshold_tag="th_change",
                save_default=False,
                save_plots=False,
            )
            adapted_scores = trainer.evaluate(
                data["X_adapt_test"],
                data["y_adapt_test"],
                shift_name=shift_name,
                target_failure_rate=target_failure_rate,
                metrics_dir=METRICS_DIR,
                threshold=adapted_threshold,
                threshold_tag="shift_adapted_th",
                save_default=False,
                save_plots=False,
            )
            print(
                f"{model_class.__name__} | {shift_name} | "
                f"source threshold: {shift_threshold:.3f} | "
                f"source F1: {source_holdout_scores['f1_score']:.3f} | "
                f"shift threshold: {adapted_threshold:.3f} | "
                f"shift-adapted F1: {adapted_scores['f1_score']:.3f} | "
                f"ROC-AUC: {scores['roc_auc']}"
            )

        model_plots_dir = PLOTS_DIR / model_class.__name__
        plot_roc_comparison(
            comparison_curves,
            model_plots_dir / "all_shifts_roc.svg",
            title=f"{model_class.__name__} ROC Curves Across Shifts",
        )
        plot_calibration_comparison(
            comparison_curves,
            model_plots_dir / "all_shifts_calibration.svg",
            title=f"{model_class.__name__} Calibration Across Shifts",
        )


if __name__ == "__main__":
    main()
