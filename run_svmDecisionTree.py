
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.data import get
from src.metrics import (
    calculate_metrics,
    find_best_f1_threshold,
    plot_roc_comparison,
    plot_calibration_comparison,
)
from src.models.svmdecisiontree import build_svm, build_decision_tree


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

METRICS_DIR = ROOT / "metrics" / "classical"
PLOTS_DIR = ROOT / "plots" / "classical"

RANDOM_STATE = 42

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Distribution shifts
# --------------------------------------------------

SHIFTS = [
    ("iid", {"type": "iid"}),
    ("temp_plus_2K", {"type": "temp"}),
    ("high_wear", {"type": "wear"}),
    ("balanced_product_mix", {"type": "product"}),
    ("high_joint_load", {"type": "joint"}),
    ("failure_rate_20pct", {"type": "label", "failure_rate": 0.2}),
    (
        "torque_corruption_20pct",
        {
            "type": "corruption",
            "feature": "Torque [Nm]",
            "corruption_pct": 20,
        },
    ),
]


# --------------------------------------------------
# 3. Model configurations
# --------------------------------------------------

MODELS = {
    "SVM_Linear": lambda: build_svm(kernel="linear"),
    "SVM_RBF": lambda: build_svm(kernel="rbf"),

    "DecisionTree_Depth3": lambda: build_decision_tree(max_depth=3),
    "DecisionTree_Depth5": lambda: build_decision_tree(max_depth=5),
    "DecisionTree_Depth10": lambda: build_decision_tree(max_depth=10),
}


# --------------------------------------------------
# 4. Main experiment
# --------------------------------------------------

def main():

    # Load original IID training data
    X_train, _, y_train, _ = get(
        type="iid",
        random_state=RANDOM_STATE,
    )

    # Split training data into fitting and validation sets
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # Load all test distributions
    test_sets = {}

    for shift_name, shift_args in SHIFTS:

        _, X_test, _, y_test = get(
            random_state=RANDOM_STATE,
            **shift_args,
        )

        test_sets[shift_name] = (X_test, y_test)

    # Store experiment results
    all_results = {}

    # --------------------------------------------------
    # 5. Train and evaluate each model
    # --------------------------------------------------

    for model_name, model_builder in MODELS.items():

        print("\n" + "=" * 60)
        print(f"Training: {model_name}")
        print("=" * 60)

        # Build model
        model = model_builder()

        # Train on original training distribution
        model.fit(X_fit, y_fit)

        # Get validation probabilities
        validation_probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        # Find optimal F1 threshold using validation data
        threshold = find_best_f1_threshold(
            y_validation,
            validation_probabilities,
        )

        print(f"Optimal validation threshold: {threshold:.3f}")

        model_results = {}
        comparison_curves = {}

        # --------------------------------------------------
        # 6. Evaluate on each distribution shift
        # --------------------------------------------------

        for shift_name, (X_test, y_test) in test_sets.items():

            # Predict failure probabilities
            probabilities = model.predict_proba(X_test)[:, 1]

            # Calculate metrics using validation threshold
            scores = calculate_metrics(
                y_test,
                probabilities,
                threshold=threshold,
            )

            # Store threshold and results
            scores["threshold"] = threshold

            model_results[shift_name] = scores

            # Save data for comparison plots
            comparison_curves[shift_name] = (
                y_test,
                probabilities,
            )

            print(
                f"{shift_name:28s} | "
                f"F1: {scores['f1_score']:.4f} | "
                f"Accuracy: {scores['accuracy']:.4f} | "
                f"Recall: {scores['recall']:.4f} | "
                f"Precision: {scores['precision']:.4f} | "
                f"ROC-AUC: {scores['roc_auc']}"
            )

        # --------------------------------------------------
        # 7. Save metrics as JSON
        # --------------------------------------------------

        all_results[model_name] = model_results

        metrics_file = METRICS_DIR / f"{model_name}.json"

        with open(metrics_file, "w") as file:
            json.dump(model_results, file, indent=4)

        print(f"\nMetrics saved to: {metrics_file}")

        # --------------------------------------------------
        # 8. Generate comparison plots
        # --------------------------------------------------

        model_plot_dir = PLOTS_DIR / model_name
        model_plot_dir.mkdir(parents=True, exist_ok=True)

        plot_roc_comparison(
            comparison_curves,
            model_plot_dir / "roc_comparison.svg",
            title=f"{model_name} - ROC Across Distribution Shifts",
        )

        plot_calibration_comparison(
            comparison_curves,
            model_plot_dir / "calibration_comparison.svg",
            title=f"{model_name} - Calibration Across Distribution Shifts",
        )

    # --------------------------------------------------
    # 9. Save combined results
    # --------------------------------------------------

    combined_file = METRICS_DIR / "SVMandDecisionTree.json"

    with open(combined_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print("\n" + "=" * 60)
    print("ALL CLASSICAL ML EXPERIMENTS COMPLETED")
    print(f"Combined results: {combined_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
