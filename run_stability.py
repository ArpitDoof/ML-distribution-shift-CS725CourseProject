import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.data import get
from src.metrics import find_best_f1_threshold
from src.models.classical import (
    ClassicalTrainer,
    GaussianNBModel,
    LogisticRegressionModel,
    MixedNBModel,
    ShiftAwareLogisticRegression,
)
from src.stability import (
    DEFAULT_SUBPOP_SIZES,
    compute_loss,
    estimate_worst_case_risk,
    plot_stability_comparison,
    plot_stability_curve,
    plot_subpopulation_profile,
    save_stability_json,
)


ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "metrics" / "stability"
PLOTS_DIR = ROOT / "plots" / "stability"
RANDOM_STATE = 42

MODELS = [
    GaussianNBModel,
    MixedNBModel,
    LogisticRegressionModel,
    ShiftAwareLogisticRegression,
]

AI4I_SENSORS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

SHIFT_SPECS = [
    (
        "marginal_all_features",
        ["Type"] + AI4I_SENSORS,
        [],
        0.0,
    ),
    (
        "cond_mechanical_given_type_y",
        ["Tool wear [min]", "Torque [Nm]", "Rotational speed [rpm]"],
        ["Type", "Machine failure"],
        0.0,
    ),
    (
        "cond_thermal_given_type_y",
        ["Air temperature [K]", "Process temperature [K]"],
        ["Type", "Machine failure"],
        0.0,
    ),
    (
        "cond_product_given_sensors",
        ["Type"],
        AI4I_SENSORS,
        1e-5,  # Algorithm 2 uniform smoothing for discrete W
    ),
]

OBSERVED_SHIFTS = {
    "high_wear": {"type": "wear"},
    "high_joint_load": {"type": "joint"},
    "balanced_product_mix": {"type": "product"},
    "temp_plus_2K": {"type": "temp"},
}

LOSSES = ["bce", "brier", "zero_one"]


def run_synthetic_validation():
    """Reproduce Subbaswamy et al. (2021) Fig. 2: marginal P(X1, X2) vs conditional P(X2 | X1)."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(0.0, 1.0, size=(3000, 2))
    y = (X[:, 0] > np.sin(2.0 * X[:, 1])).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    losses = compute_loss(y_te, probs, loss="bce")
    df_eval = pd.DataFrame({"X1": X_te[:, 0], "X2": X_te[:, 1], "Y": y_te})

    marginal_res, _ = estimate_worst_case_risk(
        df_eval,
        losses,
        mutable_cols=["X1", "X2"],
        immutable_cols=[],
        subpop_sizes=DEFAULT_SUBPOP_SIZES,
        discrete_noise=0.0,
        random_state=RANDOM_STATE,
    )
    cond_res, _ = estimate_worst_case_risk(
        df_eval,
        losses,
        mutable_cols=["X2"],
        immutable_cols=["X1"],
        subpop_sizes=DEFAULT_SUBPOP_SIZES,
        discrete_noise=0.0,
        random_state=RANDOM_STATE,
    )

    validation = {
        "marginal_P_X1_X2": marginal_res,
        "conditional_P_X2_given_X1": cond_res,
    }
    save_stability_json(validation, METRICS_DIR / "synthetic_validation.json")
    plot_stability_comparison(
        {
            "Marginal P(X1, X2)": marginal_res,
            "Conditional P(X2 | X1)": cond_res,
        },
        PLOTS_DIR / "synthetic_validation_stability.svg",
        title="Paper Fig. 2 Validation: Marginal vs Conditional Shift (BCE)",
        y_label="Worst-Case BCE Loss",
    )
    plot_subpopulation_profile(
        cond_res,
        PLOTS_DIR / "synthetic_validation_conditional_profile.svg",
        title="Synthetic Conditional Shift P(X2 | X1): X1 Preserved Across (1 - alpha)",
    )

    m10 = [p for p in marginal_res["points"] if np.isclose(p["subpop_size"], 0.1)][0]
    c10 = [p for p in cond_res["points"] if np.isclose(p["subpop_size"], 0.1)][0]
    print("=" * 92)
    print("Synthetic Validation (Subbaswamy et al., Fig. 2 — 2D Gaussian Sine, BCE Loss)")
    print("=" * 92)
    print(
        f"  Worst 10% Marginal    P(X1, X2): risk={m10['worst_case_risk']:.3f} "
        f"(CI [{m10['ci_low']:.3f}, {m10['ci_high']:.3f}]) | "
        f"X1 std_ratio={m10['profile']['mutable']['X1']['std_ratio']:.3f} (collapses under marginal shift)"
    )
    print(
        f"  Worst 10% Conditional P(X2 | X1): risk={c10['worst_case_risk']:.3f} "
        f"(CI [{c10['ci_low']:.3f}, {c10['ci_high']:.3f}]) | "
        f"X1 std_ratio={c10['profile']['immutable']['X1']['std_ratio']:.3f} (stays ~1.0 under conditional shift)"
    )
    return validation


def main():
    run_synthetic_validation()

    X_train, X_test, y_train, y_test = get(type="iid", random_state=RANDOM_STATE)
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # Preload explicit shifted test sets for empirical validation overlays
    observed_test_sets = {}
    for shift_name, shift_args in OBSERVED_SHIFTS.items():
        _, X_s, _, y_s = get(random_state=RANDOM_STATE, **shift_args)
        observed_test_sets[shift_name] = (X_s, y_s)

    df_eval = X_test.copy()
    df_eval["Machine failure"] = y_test.to_numpy()

    summary = {"models": {}}
    comparison_curves = {
        (spec_name, loss_name): {}
        for spec_name, _, _, _ in SHIFT_SPECS
        for loss_name in LOSSES
    }

    print("\n" + "=" * 92)
    print("AI4I Stability Analysis (Naive Bayes vs Logistic Regression — Subbaswamy Algorithm 1 & 2)")
    print("=" * 92)

    for model_class in MODELS:
        model_name = model_class.__name__
        trainer = ClassicalTrainer(model_class, random_state=RANDOM_STATE)
        trainer.fit(X_fit, y_fit)

        val_probs = trainer.predict_proba(X_validation)
        src_threshold = find_best_f1_threshold(y_validation, val_probs)
        test_probs = trainer.predict_proba(X_test)

        # Compute actual observed losses on our explicit shifts for overlay verification
        observed_losses = {loss_name: {} for loss_name in LOSSES}
        for shift_name, (X_s, y_s) in observed_test_sets.items():
            p_s = trainer.predict_proba(X_s)
            for loss_name in LOSSES:
                val = float(np.mean(compute_loss(y_s, p_s, loss=loss_name, threshold=src_threshold)))
                observed_losses[loss_name][shift_name] = (1.0 - val) if loss_name == "zero_one" else val

        summary["models"][model_name] = {
            "source_f1_threshold": src_threshold,
            "observed_shifts": observed_losses,
            "specs": {},
        }
        print(f"\n--- {model_name} (source F1 threshold={src_threshold:.3f}) ---")

        for spec_name, mutable_cols, immutable_cols, noise in SHIFT_SPECS:
            summary["models"][model_name]["specs"][spec_name] = {}

            for loss_name in LOSSES:
                losses = compute_loss(y_test, test_probs, loss=loss_name, threshold=src_threshold)
                result, _ = estimate_worst_case_risk(
                    df_eval,
                    losses,
                    mutable_cols=mutable_cols,
                    immutable_cols=immutable_cols,
                    subpop_sizes=DEFAULT_SUBPOP_SIZES,
                    discrete_noise=noise,
                    stratify_col="Machine failure",
                    random_state=RANDOM_STATE,
                )
                result["loss"] = loss_name
                result["threshold"] = src_threshold if loss_name == "zero_one" else None
                result["observed_shifts"] = observed_losses[loss_name]

                save_stability_json(
                    result,
                    METRICS_DIR / model_name / f"{spec_name}_{loss_name}.json",
                )

                as_acc = loss_name == "zero_one"
                y_label = "Worst-Case Accuracy" if as_acc else f"Worst-Case {loss_name.upper()} Loss"
                plot_stability_curve(
                    result,
                    PLOTS_DIR / model_name / f"{spec_name}_{loss_name}_stability.svg",
                    title=f"{model_name}: {spec_name} ({loss_name.upper()})",
                    y_label=y_label,
                    as_accuracy=as_acc,
                    observed_lines=observed_losses[loss_name] if spec_name == "marginal_all_features" else None,
                )

                if loss_name == "bce":
                    plot_subpopulation_profile(
                        result,
                        PLOTS_DIR / model_name / f"{spec_name}_profile.svg",
                        title=f"{model_name} Worst-Case Profile ({spec_name})",
                    )

                comparison_curves[(spec_name, loss_name)][model_name] = result

                p_full = result["points"][0]
                p_20 = [p for p in result["points"] if np.isclose(p["subpop_size"], 0.2)][0]
                p_10 = [p for p in result["points"] if np.isclose(p["subpop_size"], 0.1)][0]
                summary["models"][model_name]["specs"][spec_name][loss_name] = {
                    "iid_1.0": p_full["worst_case_risk"],
                    "worst_0.2": {
                        "risk": p_20["worst_case_risk"],
                        "ci": [p_20["ci_low"], p_20["ci_high"]],
                    },
                    "worst_0.1": {
                        "risk": p_10["worst_case_risk"],
                        "ci": [p_10["ci_low"], p_10["ci_high"]],
                    },
                }

                if loss_name == "bce":
                    print(
                        f"  {spec_name:30s} | BCE: "
                        f"1.0={p_full['worst_case_risk']:.4f} -> "
                        f"0.2={p_20['worst_case_risk']:.4f} [{p_20['ci_low']:.4f},{p_20['ci_high']:.4f}] -> "
                        f"0.1={p_10['worst_case_risk']:.4f} [{p_10['ci_low']:.4f},{p_10['ci_high']:.4f}]"
                    )

    # Save NB vs LR head-to-head stability comparisons
    for (spec_name, loss_name), curves in comparison_curves.items():
        as_acc = loss_name == "zero_one"
        y_label = "Worst-Case Accuracy" if as_acc else f"Worst-Case {loss_name.upper()} Loss"
        plot_stability_comparison(
            curves,
            PLOTS_DIR / "nb_vs_lr_comparison" / f"{spec_name}_{loss_name}_comparison.svg",
            title=f"NB vs LR Stability: {spec_name} ({loss_name.upper()})",
            y_label=y_label,
            as_accuracy=as_acc,
        )

    save_stability_json(summary, METRICS_DIR / "stability_summary.json")
    print(f"\nSaved stability metrics to {METRICS_DIR} and plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
