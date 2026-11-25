from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data import get
from src.metrics import (
    calculate_metrics,
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
)
from src.models.nn import DeepNN, ShiftAwareNN, ShiftRobustNN, Trainer, WideNN
from src.models.svmdecisiontree import build_decision_tree, build_svm
from src.stability import (
    DEFAULT_SUBPOP_SIZES,
    compute_loss,
    estimate_worst_case_risk,
    plot_stability_comparison,
    save_stability_json,
)


ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "metrics" / "cross_family"
PLOTS_DIR = ROOT / "plots" / "cross_family"
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

AI4I_SENSORS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

STABILITY_SPECS = [
    ("marginal_all_features", ["Type"] + AI4I_SENSORS, [], 0.0),
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
    ("cond_product_given_sensors", ["Type"], AI4I_SENSORS, 1e-5),
]

# 6 clean representatives (one per paradigm) for uncluttered multi-curve overlay SVGs
REPRESENTATIVE_MODELS = [
    "MixedNBModel",
    "LogisticRegressionModel",
    "SVM_RBF",
    "DecisionTree_Depth5",
    "DeepNN",
    "ShiftAwareNN",
]

MODEL_FAMILY = {
    "GaussianNBModel": "Generative (NB)",
    "MixedNBModel": "Generative (NB)",
    "LogisticRegressionModel": "Discriminative (LR)",
    "ShiftAwareLogisticRegression": "Discriminative (LR)",
    "SVM_Linear": "Parametric (SVM)",
    "SVM_RBF": "Parametric (SVM)",
    "DecisionTree_Depth3": "Non-Parametric (DT)",
    "DecisionTree_Depth5": "Non-Parametric (DT)",
    "DecisionTree_Depth10": "Non-Parametric (DT)",
    "DeepNN": "Neural Network",
    "WideNN": "Neural Network",
    "ShiftRobustNN": "Neural Network",
    "ShiftAwareNN": "Neural Network",
}


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


def restore_balanced_prior(probabilities, train_rate):
    """Undo exact 50/50 balanced class_weight prior shift back to true source failure rate."""
    p = np.asarray(probabilities, dtype=float)
    num = p * train_rate
    den = num + (1.0 - p) * (1.0 - train_rate)
    return np.where(den > 0, num / den, 0.0)


def apply_prior_adjustment(probabilities, source_rate, target_rate):
    """Apply exact Bayes log-odds shift from source_rate to target_rate."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p))
    delta = np.log(target_rate / (1.0 - target_rate)) - np.log(source_rate / (1.0 - source_rate))
    adj = logits + delta
    pos = adj >= 0
    out = np.empty_like(adj)
    out[pos] = 1.0 / (1.0 + np.exp(-adj[pos]))
    exp_x = np.exp(adj[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def fit_all_models(X_fit, y_fit):
    """Fit all 13 models across all 3 families read-only without modifying teammate code."""
    train_rate = float(np.mean(y_fit))
    fitted = {}

    # 1. Generative (NB) & Discriminative (LR)
    for cls in (GaussianNBModel, MixedNBModel, LogisticRegressionModel, ShiftAwareLogisticRegression):
        trainer = ClassicalTrainer(cls, random_state=RANDOM_STATE).fit(X_fit, y_fit)
        fitted[cls.__name__] = {
            "predict_proba": lambda X, t=trainer: t.predict_proba(X),
            "predict_raw": None,
        }

    # 2. Parametric (SVM) & Non-Parametric (Decision Trees)
    svm_builders = {
        "SVM_Linear": lambda: build_svm(kernel="linear"),
        "SVM_RBF": lambda: build_svm(kernel="rbf"),
    }
    for name, builder in svm_builders.items():
        pipe = builder().fit(X_fit, y_fit)
        fitted[name] = {
            "predict_proba": lambda X, m=pipe: m.predict_proba(X)[:, 1],
            "predict_raw": None,
        }

    dt_builders = {
        "DecisionTree_Depth3": lambda: build_decision_tree(max_depth=3),
        "DecisionTree_Depth5": lambda: build_decision_tree(max_depth=5),
        "DecisionTree_Depth10": lambda: build_decision_tree(max_depth=10),
    }
    for name, builder in dt_builders.items():
        pipe = builder().fit(X_fit, y_fit)
        # Restore true source prior from class_weight="balanced" for fair ECE/BCE parity; keep raw too
        fitted[name] = {
            "predict_proba": lambda X, m=pipe, r=train_rate: restore_balanced_prior(
                m.predict_proba(X)[:, 1], r
            ),
            "predict_raw": lambda X, m=pipe: m.predict_proba(X)[:, 1],
        }

    # 3. Neural Networks (same hyperparameters as run_nn.py, CPU-safe)
    for cls in (DeepNN, WideNN, ShiftRobustNN, ShiftAwareNN):
        trainer = Trainer(
            cls,
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            random_state=RANDOM_STATE,
            device="cpu",
        ).fit(X_fit, y_fit)
        fitted[cls.__name__] = {
            "predict_proba": lambda X, t=trainer: t.predict_proba(X),
            "predict_raw": None,
        }

    return fitted


def plot_metric_heatmap(matrix, model_names, shift_names, save_path, title, higher_is_better=True, fmt="{:.3f}"):
    """Save a crisp Models x Shifts heatmap as SVG."""
    cell_w, cell_h = 108, 32
    left, top = 235, 68
    width = left + len(shift_names) * cell_w + 24
    height = top + len(model_names) * cell_h + 70

    values = [matrix[m][s] for m in model_names for s in shift_names]
    vmin, vmax = min(values), max(values)
    span = max(vmax - vmin, 1e-6)

    def cell_color(val):
        norm = (val - vmin) / span
        score = norm if higher_is_better else (1.0 - norm)
        # Soft red (low) -> cream (mid) -> teal/green (high)
        if score < 0.5:
            t = score * 2.0
            r = int(229 + (252 - 229) * t)
            g = int(115 + (244 - 115) * t)
            b = int(115 + (214 - 115) * t)
        else:
            t = (score - 0.5) * 2.0
            r = int(252 + (92 - 252) * t)
            g = int(244 + (184 - 244) * t)
            b = int(214 + (138 - 214) * t)
        return f"rgb({r},{g},{b})"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">{escape(title)}</text>',
    ]

    short_labels = {
        "iid": "IID",
        "temp_plus_2K": "Temp +2K",
        "high_wear": "High Wear",
        "balanced_product_mix": "Prod Mix",
        "high_joint_load": "Joint Load",
        "failure_rate_20pct": "Label 20%",
        "torque_corruption_20pct": "Corrupt 20%",
    }
    for j, shift in enumerate(shift_names):
        cx = left + j * cell_w + cell_w / 2
        svg.append(
            f'<text x="{cx:.1f}" y="{top - 12}" text-anchor="middle" font-size="12" font-weight="bold">{escape(short_labels.get(shift, shift))}</text>'
        )

    for i, model in enumerate(model_names):
        y = top + i * cell_h
        svg.append(
            f'<text x="{left - 10}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end" font-size="12">{escape(model)}</text>'
        )
        for j, shift in enumerate(shift_names):
            x = left + j * cell_w
            val = matrix[model][shift]
            svg.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{cell_color(val)}" stroke="white" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{x + cell_w / 2 - 1:.1f}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="middle" font-size="12" fill="#111111">{fmt.format(val)}</text>'
            )

    svg.append(
        f'<text x="{left}" y="{height - 20}" font-size="11" fill="#555555">'
        'Green = stronger performance, Red = weaker performance. Decision Trees use Bayes prior-restored posteriors for fair ECE parity.'
        '</text>'
    )
    svg.append("</svg>")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def main():
    X_train, X_test_iid, y_train, y_test_iid = get(type="iid", random_state=RANDOM_STATE)
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    train_rate = float(np.mean(y_fit))

    test_sets = {}
    for shift_name, shift_args, target_rate in SHIFTS:
        _, X_test, _, y_test = get(random_state=RANDOM_STATE, **shift_args)
        X_cal, y_cal, X_hold, y_hold = split_for_shift_adaptation(X_test, y_test, RANDOM_STATE)
        test_sets[shift_name] = {
            "X_test": X_test,
            "y_test": y_test,
            "target_rate": target_rate,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_hold": X_hold,
            "y_hold": y_hold,
        }

    print("=" * 100)
    print("Training All 13 Models Across 3 Families (NB/LR, SVM/DT, Neural Networks)")
    print("=" * 100)
    models = fit_all_models(X_fit, y_fit)
    model_names = list(models.keys())
    shift_names = [s for s, _, _ in SHIFTS]

    benchmark = {"train_failure_rate": train_rate, "models": {}}
    auc_matrix = {m: {} for m in model_names}
    f1_matrix = {m: {} for m in model_names}
    ece_matrix = {m: {} for m in model_names}
    rep_curves_by_shift = {s: {} for s in shift_names}

    print("\n" + "=" * 100)
    print("Part 1: Cross-Family Benchmark Across IID + 6 Distribution Shifts")
    print("=" * 100)

    source_thresholds = {}
    for model_name, entry in models.items():
        predict_proba = entry["predict_proba"]
        val_probs = predict_proba(X_validation)
        src_th = find_best_f1_threshold(y_validation, val_probs)
        source_thresholds[model_name] = src_th

        benchmark["models"][model_name] = {
            "family": MODEL_FAMILY[model_name],
            "source_f1_threshold": src_th,
            "shifts": {},
            "label_shift_prior_mitigation": {},
        }

        iid_ref = None
        for shift_name, data in test_sets.items():
            X_test, y_test = data["X_test"], data["y_test"]
            probs = predict_proba(X_test)
            default_scores = calculate_metrics(y_test, probs, threshold=0.5)
            src_full_scores = calculate_metrics(y_test, probs, threshold=src_th)

            # Holdout split parity with run_nn.py & run_classical.py
            cal_probs = predict_proba(data["X_cal"])
            adapt_th = find_best_f1_threshold(data["y_cal"], cal_probs)
            hold_probs = predict_proba(data["X_hold"])
            src_hold = calculate_metrics(data["y_hold"], hold_probs, threshold=src_th)
            adapt_hold = calculate_metrics(data["y_hold"], hold_probs, threshold=adapt_th)

            shift_record = {
                "roc_auc": default_scores["roc_auc"],
                "ece": default_scores["expected_calibration_error"],
                "f1_default_0.5": default_scores["f1_score"],
                "accuracy_default_0.5": default_scores["accuracy"],
                "f1_src_th_full": src_full_scores["f1_score"],
                "accuracy_src_th_full": src_full_scores["accuracy"],
                "f1_src_th_holdout": src_hold["f1_score"],
                "f1_shift_adapted_holdout": adapt_hold["f1_score"],
                "shift_adapted_threshold": adapt_th,
            }
            if entry["predict_raw"] is not None:
                raw_scores = calculate_metrics(y_test, entry["predict_raw"](X_test), threshold=0.5)
                shift_record["raw_balanced_ece"] = raw_scores["expected_calibration_error"]

            if shift_name == "iid":
                iid_ref = shift_record
            shift_record["delta_auc_vs_iid"] = shift_record["roc_auc"] - iid_ref["roc_auc"]
            shift_record["delta_f1_vs_iid"] = shift_record["f1_src_th_full"] - iid_ref["f1_src_th_full"]
            shift_record["delta_ece_vs_iid"] = shift_record["ece"] - iid_ref["ece"]

            benchmark["models"][model_name]["shifts"][shift_name] = shift_record
            auc_matrix[model_name][shift_name] = shift_record["roc_auc"]
            f1_matrix[model_name][shift_name] = shift_record["f1_src_th_full"]
            ece_matrix[model_name][shift_name] = shift_record["ece"]

            if model_name in REPRESENTATIVE_MODELS:
                rep_curves_by_shift[shift_name][model_name] = (y_test, probs)

            # Label shift exact prior re-weighting across all 13 models
            if data["target_rate"] is not None:
                prior_probs = apply_prior_adjustment(probs, train_rate, data["target_rate"])
                prior_scores = calculate_metrics(y_test, prior_probs, threshold=0.5)
                benchmark["models"][model_name]["label_shift_prior_mitigation"][shift_name] = {
                    "f1_0.5_before": default_scores["f1_score"],
                    "f1_0.5_after_prior": prior_scores["f1_score"],
                    "ece_before": default_scores["expected_calibration_error"],
                    "ece_after_prior": prior_scores["expected_calibration_error"],
                    "roc_auc": prior_scores["roc_auc"],
                }

        s = benchmark["models"][model_name]["shifts"]
        print(
            f"{model_name:28s} | IID AUC={s['iid']['roc_auc']:.3f} F1={s['iid']['f1_src_th_full']:.3f} ECE={s['iid']['ece']:.4f} | "
            f"Temp+2K AUC={s['temp_plus_2K']['roc_auc']:.3f} F1={s['temp_plus_2K']['f1_src_th_full']:.3f} | "
            f"Joint F1={s['high_joint_load']['f1_src_th_full']:.3f} | "
            f"Corrupt AUC={s['torque_corruption_20pct']['roc_auc']:.3f}"
        )

    # Save heatmaps & per-shift representative ROC/Calibration overlays
    plot_metric_heatmap(
        auc_matrix, model_names, shift_names,
        PLOTS_DIR / "heatmap_roc_auc.svg",
        "Cross-Family ROC-AUC Across Distribution Shifts (13 Models)",
        higher_is_better=True, fmt="{:.3f}",
    )
    plot_metric_heatmap(
        f1_matrix, model_names, shift_names,
        PLOTS_DIR / "heatmap_f1_src_th.svg",
        "Cross-Family Source-Tuned F1 Across Distribution Shifts (13 Models)",
        higher_is_better=True, fmt="{:.3f}",
    )
    plot_metric_heatmap(
        ece_matrix, model_names, shift_names,
        PLOTS_DIR / "heatmap_ece.svg",
        "Cross-Family Expected Calibration Error (ECE) Across Shifts (Lower is Better)",
        higher_is_better=False, fmt="{:.4f}",
    )

    for shift_name, curves in rep_curves_by_shift.items():
        plot_roc_comparison(
            curves,
            PLOTS_DIR / "per_shift_comparison" / f"{shift_name}_roc.svg",
            title=f"Cross-Family ROC Comparison ({shift_name})",
        )
        plot_calibration_comparison(
            curves,
            PLOTS_DIR / "per_shift_comparison" / f"{shift_name}_calibration.svg",
            title=f"Cross-Family Calibration Comparison ({shift_name})",
        )

    # ------------------------------------------------------------------
    # Part 2: Cross-Family Subbaswamy Worst-Case Stability Analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Part 2: Cross-Family Subbaswamy Stability Analysis (All 13 Models on Held-Out IID Test Set)")
    print("=" * 100)

    df_eval = X_test_iid.copy()
    df_eval["Machine failure"] = y_test_iid.to_numpy()
    stability_summary = {"models": {}}
    rep_stability_curves = {
        (spec_name, loss_name): {}
        for spec_name, _, _, _ in STABILITY_SPECS
        for loss_name in ("bce", "zero_one")
    }

    for model_name, entry in models.items():
        probs = entry["predict_proba"](X_test_iid)
        src_th = source_thresholds[model_name]
        stability_summary["models"][model_name] = {
            "family": MODEL_FAMILY[model_name],
            "source_f1_threshold": src_th,
            "specs": {},
        }
        row_bits = []
        for spec_name, mutable_cols, immutable_cols, noise in STABILITY_SPECS:
            stability_summary["models"][model_name]["specs"][spec_name] = {}
            for loss_name in ("bce", "zero_one"):
                losses = compute_loss(y_test_iid, probs, loss=loss_name, threshold=src_th)
                res, _ = estimate_worst_case_risk(
                    df_eval,
                    losses,
                    mutable_cols=mutable_cols,
                    immutable_cols=immutable_cols,
                    subpop_sizes=DEFAULT_SUBPOP_SIZES,
                    discrete_noise=noise,
                    stratify_col="Machine failure",
                    random_state=RANDOM_STATE,
                )
                p_full = res["points"][0]
                p_20 = [p for p in res["points"] if np.isclose(p["subpop_size"], 0.2)][0]
                p_10 = [p for p in res["points"] if np.isclose(p["subpop_size"], 0.1)][0]
                stability_summary["models"][model_name]["specs"][spec_name][loss_name] = {
                    "iid_1.0": p_full["worst_case_risk"],
                    "worst_0.2": {"risk": p_20["worst_case_risk"], "ci": [p_20["ci_low"], p_20["ci_high"]]},
                    "worst_0.1": {"risk": p_10["worst_case_risk"], "ci": [p_10["ci_low"], p_10["ci_high"]]},
                }
                if model_name in REPRESENTATIVE_MODELS:
                    rep_stability_curves[(spec_name, loss_name)][model_name] = res
                if loss_name == "bce" and spec_name in ("marginal_all_features", "cond_mechanical_given_type_y", "cond_thermal_given_type_y"):
                    tag = spec_name.split("_")[1]
                    row_bits.append(f"{tag}@0.1={p_10['worst_case_risk']:.3f}")
        print(f"{model_name:28s} | BCE IID={stability_summary['models'][model_name]['specs']['marginal_all_features']['bce']['iid_1.0']:.3f} | " + " | ".join(row_bits))

    for (spec_name, loss_name), curves in rep_stability_curves.items():
        as_acc = loss_name == "zero_one"
        y_label = "Worst-Case Accuracy" if as_acc else "Worst-Case BCE Loss"
        plot_stability_comparison(
            curves,
            PLOTS_DIR / "stability" / f"{spec_name}_{loss_name}_cross_family.svg",
            title=f"Cross-Family Stability: {spec_name} ({loss_name.upper()})",
            y_label=y_label,
            as_accuracy=as_acc,
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "cross_family_benchmark.json").open("w") as f:
        json.dump(benchmark, f, indent=2)
    save_stability_json(stability_summary, METRICS_DIR / "cross_family_stability.json")

    print("\n" + "=" * 100)
    print("Label Shift Prior Re-weighting Across All 13 Models (failure_rate_20pct)")
    print("=" * 100)
    for model_name in model_names:
        pr = benchmark["models"][model_name]["label_shift_prior_mitigation"]["failure_rate_20pct"]
        print(
            f"  {model_name:28s} | F1@0.5: {pr['f1_0.5_before']:.3f} -> {pr['f1_0.5_after_prior']:.3f} | "
            f"ECE: {pr['ece_before']:.4f} -> {pr['ece_after_prior']:.4f}"
        )

    print(f"\nSaved cross-family metrics to {METRICS_DIR} and plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
