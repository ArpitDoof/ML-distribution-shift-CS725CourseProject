import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data import get
from src.metrics import (
    adjust_threshold_for_prior,
    find_best_f1_threshold,
    plot_calibration_comparison,
    plot_roc_comparison,
)
from src.models.classical import (
    ClassicalTrainer,
    GaussianNBModel,
    LogisticRegressionModel,
    MixedNBModel,
    ShiftAwareLogisticRegression,
    estimate_importance_weights,
)


ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "metrics" / "classical"
PLOTS_DIR = ROOT / "plots" / "classical"
RANDOM_STATE = 42

SHIFTS = [
    ("iid", {"type": "iid"}, None),
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

COVARIATE_SHIFTS = {
    "temp_plus_2K",
    "high_wear",
    "balanced_product_mix",
    "high_joint_load",
}

MODELS = [
    GaussianNBModel,
    MixedNBModel,
    LogisticRegressionModel,
    ShiftAwareLogisticRegression,
]


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

    # Precompute unsupervised importance weights for covariate shifts (shared across models)
    importance_weights_by_shift = {}
    for shift_name in COVARIATE_SHIFTS:
        importance_weights_by_shift[shift_name] = estimate_importance_weights(
            X_fit, test_sets[shift_name]["X_test"], random_state=RANDOM_STATE
        )

    summary = {"models": {}, "mitigations": {}}
    per_shift_model_curves = {shift_name: {} for shift_name, _, _ in SHIFTS}

    print("=" * 96)
    print("NB vs Logistic Regression — Baseline Evaluation Across IID and 6 Distribution Shifts")
    print("=" * 96)

    for model_class in MODELS:
        model_name = model_class.__name__
        trainer = ClassicalTrainer(model_class, random_state=RANDOM_STATE)
        trainer.fit(X_fit, y_fit)

        # Source validation F1 threshold & post-hoc temperature scaling
        validation_probabilities = trainer.predict_proba(X_validation)
        source_threshold = find_best_f1_threshold(y_validation, validation_probabilities)
        temperature = trainer.fit_temperature(X_validation, y_validation)
        calibrated_val_probs = trainer.predict_proba(X_validation, temperature=temperature)
        calibrated_threshold = find_best_f1_threshold(y_validation, calibrated_val_probs)

        summary["models"][model_name] = {
            "source_f1_threshold": source_threshold,
            "fitted_temperature": temperature,
            "shifts": {},
        }
        summary["mitigations"][model_name] = {
            "temperature_scaling": {},
            "prior_reweighting": {},
            "importance_weighting": {},
        }

        comparison_curves = {}
        print(f"\n--- {model_name} (source th={source_threshold:.3f}, temp T={temperature:.3f}) ---")

        for shift_name, data in test_sets.items():
            X_test, y_test = data["X_test"], data["y_test"]
            target_failure_rate = data["target_failure_rate"]

            # 1. Unmitigated baseline evaluation (default 0.5 threshold on full shifted set)
            default_scores = trainer.evaluate(
                X_test,
                y_test,
                shift_name=shift_name,
                metrics_dir=METRICS_DIR,
                plots_dir=PLOTS_DIR,
            )
            probabilities = trainer.predict_proba(X_test)
            comparison_curves[shift_name] = (y_test, probabilities)
            per_shift_model_curves[shift_name][model_name] = (y_test, probabilities)

            # 2. Source-tuned threshold vs shift-adapted threshold on holdout half (matches run_nn.py)
            calibration_probabilities = trainer.predict_proba(data["X_calibration"])
            adapted_threshold = find_best_f1_threshold(
                data["y_calibration"], calibration_probabilities
            )
            source_holdout_scores = trainer.evaluate(
                data["X_adapt_test"],
                data["y_adapt_test"],
                shift_name=shift_name,
                metrics_dir=METRICS_DIR,
                threshold=source_threshold,
                threshold_tag="th_change",
                save_default=False,
                save_plots=False,
            )
            adapted_scores = trainer.evaluate(
                data["X_adapt_test"],
                data["y_adapt_test"],
                shift_name=shift_name,
                metrics_dir=METRICS_DIR,
                threshold=adapted_threshold,
                threshold_tag="shift_adapted_th",
                save_default=False,
                save_plots=False,
            )

            summary["models"][model_name]["shifts"][shift_name] = {
                "default_0.5": {
                    "accuracy": default_scores["accuracy"],
                    "f1_score": default_scores["f1_score"],
                    "roc_auc": default_scores["roc_auc"],
                    "ece": default_scores["expected_calibration_error"],
                },
                "source_tuned_holdout": {
                    "threshold": source_threshold,
                    "f1_score": source_holdout_scores["f1_score"],
                    "ece": source_holdout_scores["expected_calibration_error"],
                },
                "shift_adapted_holdout": {
                    "threshold": adapted_threshold,
                    "f1_score": adapted_scores["f1_score"],
                    "ece": adapted_scores["expected_calibration_error"],
                },
            }

            print(
                f"{model_name:28s} | {shift_name:23s} | "
                f"AUC: {default_scores['roc_auc']:.3f} | "
                f"ECE: {default_scores['expected_calibration_error']:.4f} | "
                f"F1@0.5: {default_scores['f1_score']:.3f} | "
                f"src-th F1: {source_holdout_scores['f1_score']:.3f} | "
                f"adapt-th F1: {adapted_scores['f1_score']:.3f}"
            )

            # 3. Mitigation A: Temperature Scaling (post-hoc calibration across all shifts)
            temp_scaled_scores = trainer.evaluate(
                X_test,
                y_test,
                shift_name=shift_name,
                temperature=temperature,
                metrics_dir=METRICS_DIR,
                plots_dir=PLOTS_DIR,
                threshold=calibrated_threshold,
                threshold_tag="temp_scaled_th",
                save_default=True,
                save_plots=False,
                file_suffix="_temp_scaled",
            )
            temp_default_probs = trainer.predict_proba(X_test, temperature=temperature)
            temp_default = trainer.evaluate(
                X_test,
                y_test,
                shift_name=shift_name,
                temperature=temperature,
                metrics_dir=METRICS_DIR,
                save_default=False,
                save_plots=False,
            )
            summary["mitigations"][model_name]["temperature_scaling"][shift_name] = {
                "temperature": temperature,
                "ece_before": default_scores["expected_calibration_error"],
                "ece_after": temp_default["expected_calibration_error"],
                "f1_0.5_before": default_scores["f1_score"],
                "f1_0.5_after": temp_default["f1_score"],
                "f1_tuned_after": temp_scaled_scores["f1_score"],
                "roc_auc": temp_default["roc_auc"],
            }
            _ = temp_default_probs

            # 4. Mitigation B: Prior Re-weighting on Label Shift
            if target_failure_rate is not None:
                prior_adj_threshold = adjust_threshold_for_prior(
                    source_threshold,
                    trainer.train_failure_rate,
                    target_failure_rate,
                )
                prior_default = trainer.evaluate(
                    X_test,
                    y_test,
                    shift_name=shift_name,
                    target_failure_rate=target_failure_rate,
                    metrics_dir=METRICS_DIR,
                    plots_dir=PLOTS_DIR,
                    save_default=True,
                    save_plots=True,
                    file_suffix="_prior_adjusted",
                )
                prior_cal = trainer.evaluate(
                    X_test,
                    y_test,
                    shift_name=shift_name,
                    target_failure_rate=target_failure_rate,
                    temperature=temperature,
                    metrics_dir=METRICS_DIR,
                    plots_dir=PLOTS_DIR,
                    save_default=True,
                    save_plots=False,
                    file_suffix="_temp_and_prior_adjusted",
                )
                summary["mitigations"][model_name]["prior_reweighting"][shift_name] = {
                    "prior_adjusted_threshold": prior_adj_threshold,
                    "ece_before": default_scores["expected_calibration_error"],
                    "ece_after_prior": prior_default["expected_calibration_error"],
                    "ece_after_temp_and_prior": prior_cal["expected_calibration_error"],
                    "f1_0.5_before": default_scores["f1_score"],
                    "f1_0.5_after_prior": prior_default["f1_score"],
                    "f1_0.5_after_temp_and_prior": prior_cal["f1_score"],
                    "roc_auc": prior_default["roc_auc"],
                }

            # 5. Mitigation C: Importance Weighting on Covariate Shifts
            if shift_name in COVARIATE_SHIFTS:
                iw_trainer = ClassicalTrainer(model_class, random_state=RANDOM_STATE)
                iw_trainer.fit(
                    X_fit,
                    y_fit,
                    sample_weight=importance_weights_by_shift[shift_name],
                )
                iw_val_probs = iw_trainer.predict_proba(X_validation)
                iw_val_th = find_best_f1_threshold(y_validation, iw_val_probs)
                iw_default = iw_trainer.evaluate(
                    X_test,
                    y_test,
                    shift_name=shift_name,
                    metrics_dir=METRICS_DIR,
                    plots_dir=PLOTS_DIR,
                    save_default=True,
                    save_plots=False,
                    file_suffix="_importance_weighted",
                )
                iw_holdout = iw_trainer.evaluate(
                    data["X_adapt_test"],
                    data["y_adapt_test"],
                    shift_name=shift_name,
                    metrics_dir=METRICS_DIR,
                    threshold=iw_val_th,
                    threshold_tag="iw_th_change",
                    save_default=False,
                    save_plots=False,
                    file_suffix="_importance_weighted",
                )
                summary["mitigations"][model_name]["importance_weighting"][shift_name] = {
                    "roc_auc_before": default_scores["roc_auc"],
                    "roc_auc_after": iw_default["roc_auc"],
                    "ece_before": default_scores["expected_calibration_error"],
                    "ece_after": iw_default["expected_calibration_error"],
                    "f1_0.5_before": default_scores["f1_score"],
                    "f1_0.5_after": iw_default["f1_score"],
                    "src_th_f1_before": source_holdout_scores["f1_score"],
                    "src_th_f1_after": iw_holdout["f1_score"],
                }

        # Combined plots across shifts per model (exact match to run_nn.py)
        model_plots_dir = PLOTS_DIR / model_name
        plot_roc_comparison(
            comparison_curves,
            model_plots_dir / "all_shifts_roc.svg",
            title=f"{model_name} ROC Curves Across Shifts",
        )
        plot_calibration_comparison(
            comparison_curves,
            model_plots_dir / "all_shifts_calibration.svg",
            title=f"{model_name} Calibration Across Shifts",
        )

    # Head-to-head NB vs LR overlay plots for each shift
    compare_plots_dir = PLOTS_DIR / "nb_vs_lr_comparison"
    for shift_name, curves in per_shift_model_curves.items():
        plot_roc_comparison(
            curves,
            compare_plots_dir / f"{shift_name}_roc.svg",
            title=f"NB vs Logistic Regression ROC ({shift_name})",
        )
        plot_calibration_comparison(
            curves,
            compare_plots_dir / f"{shift_name}_calibration.svg",
            title=f"NB vs Logistic Regression Calibration ({shift_name})",
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = METRICS_DIR / "nb_vs_lr_summary.json"
    with summary_path.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("\n" + "=" * 96)
    print("Mitigation Highlights")
    print("=" * 96)
    print("\n[1] Temperature Scaling (ECE before -> after on IID & Shifts):")
    for model_name, mit in summary["mitigations"].items():
        ts = mit["temperature_scaling"]
        iid_ts = ts["iid"]
        corr_ts = ts["torque_corruption_20pct"]
        temp_ts = ts["temp_plus_2K"]
        print(
            f"  {model_name:28s} (T={iid_ts['temperature']:5.2f}) | "
            f"IID ECE: {iid_ts['ece_before']:.4f}->{iid_ts['ece_after']:.4f} | "
            f"Temp+2K ECE: {temp_ts['ece_before']:.4f}->{temp_ts['ece_after']:.4f} | "
            f"Corrupt ECE: {corr_ts['ece_before']:.4f}->{corr_ts['ece_after']:.4f}"
        )

    print("\n[2] Prior Re-weighting on Label Shift (failure_rate_20pct):")
    for model_name, mit in summary["mitigations"].items():
        pr = mit["prior_reweighting"]["failure_rate_20pct"]
        print(
            f"  {model_name:28s} | "
            f"F1@0.5: {pr['f1_0.5_before']:.3f} -> {pr['f1_0.5_after_prior']:.3f} "
            f"(+temp: {pr['f1_0.5_after_temp_and_prior']:.3f}) | "
            f"ECE: {pr['ece_before']:.4f} -> {pr['ece_after_prior']:.4f} "
            f"(+temp: {pr['ece_after_temp_and_prior']:.4f})"
        )

    print("\n[3] Importance Weighting on Covariate Shifts (ROC-AUC & src-th F1 before -> after):")
    for model_name, mit in summary["mitigations"].items():
        for shift_name in ("temp_plus_2K", "high_wear", "balanced_product_mix", "high_joint_load"):
            iw = mit["importance_weighting"][shift_name]
            print(
                f"  {model_name:28s} | {shift_name:20s} | "
                f"AUC: {iw['roc_auc_before']:.3f}->{iw['roc_auc_after']:.3f} | "
                f"src-th F1: {iw['src_th_f1_before']:.3f}->{iw['src_th_f1_after']:.3f} | "
                f"ECE: {iw['ece_before']:.4f}->{iw['ece_after']:.4f}"
            )

    print(f"\nSaved metrics to {METRICS_DIR} and plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
