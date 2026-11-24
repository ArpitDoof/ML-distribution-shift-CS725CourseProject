from html import escape
from pathlib import Path

import numpy as np


def _roc_auc(y_true, scores):
    positives = y_true == 1
    n_positive = positives.sum()
    n_negative = len(y_true) - n_positive
    if n_positive == 0 or n_negative == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end

    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def calculate_metrics(y_true, probabilities, n_bins=10, threshold=0.5):
    """Calculate binary classification metrics from positive-class probabilities."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (probabilities >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    accuracy = (tn + tp) / len(y_true) if len(y_true) else 0.0

    per_class = {}
    for class_id, class_tp, class_fp, class_fn, support in (
        (0, tn, fn, fp, tn + fp),
        (1, tp, fp, fn, fn + tp),
    ):
        class_precision = class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0
        class_recall = class_tp / support if support else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall else 0.0
        )
        per_class[str(class_id)] = {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1_score": float(class_f1),
            "support": int(support),
        }

    bin_ids = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if np.any(in_bin):
            ece += np.mean(in_bin) * abs(np.mean(y_true[in_bin]) - np.mean(probabilities[in_bin]))

    return {
        "accuracy": float(accuracy),
        "f1_score": per_class["1"]["f1_score"],
        "recall": per_class["1"]["recall"],
        "precision": per_class["1"]["precision"],
        "roc_auc": _roc_auc(y_true, probabilities),
        "expected_calibration_error": float(ece),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "per_class": per_class,
    }


def find_best_f1_threshold(y_true, probabilities):
    """Choose a threshold that maximizes F1 on validation data."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    best_threshold, best_f1 = 0.5, -1.0

    for threshold in np.arange(0.01, 1.0, 0.01):
        predictions = probabilities >= threshold
        true_positive = np.sum(predictions & (y_true == 1))
        false_positive = np.sum(predictions & (y_true == 0))
        false_negative = np.sum((~predictions) & (y_true == 1))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1

    return float(best_threshold)


def adjust_threshold_for_prior(threshold, source_rate, target_rate):
    """Adjust a threshold after shifting probabilities to a new class prior."""
    if not 0 < threshold < 1 or not 0 < source_rate < 1 or not 0 < target_rate < 1:
        raise ValueError("threshold and failure rates must be between 0 and 1")
    source_logit = np.log(threshold / (1 - threshold))
    prior_change = np.log(target_rate / (1 - target_rate))
    prior_change -= np.log(source_rate / (1 - source_rate))
    adjusted_logit = source_logit + prior_change
    return float(1 / (1 + np.exp(-adjusted_logit)))


def plot_roc_curve(y_true, probabilities, save_path, title="ROC Curve"):
    """Save a simple ROC curve as an SVG file."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC curve requires both positive and negative examples")

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = y_true[order]
    sorted_scores = probabilities[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    last_in_tie = np.r_[np.where(np.diff(sorted_scores))[0], len(sorted_scores) - 1]
    tpr = np.r_[0, true_positives[last_in_tie] / positives]
    fpr = np.r_[0, false_positives[last_in_tie] / negatives]
    auc = _roc_auc(y_true, probabilities)

    width, height = 640, 520
    left, top, plot_width, plot_height = 80, 55, 500, 390
    x = lambda value: left + value * plot_width
    y = lambda value: top + (1 - value) * plot_height
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = tick / 5
        svg.extend([
            f'<line x1="{x(value)}" y1="{top}" x2="{x(value)}" y2="{top + plot_height}" stroke="#dddddd"/>',
            f'<line x1="{left}" y1="{y(value)}" x2="{left + plot_width}" y2="{y(value)}" stroke="#dddddd"/>',
            f'<text x="{x(value)}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12">{value:.1f}</text>',
            f'<text x="{left - 12}" y="{y(value) + 4}" text-anchor="end" font-size="12">{value:.1f}</text>',
        ])

    points = " ".join(f"{x(fpr_value):.1f},{y(tpr_value):.1f}" for fpr_value, tpr_value in zip(fpr, tpr))
    svg.extend([
        f'<line x1="{left}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#777777" stroke-dasharray="6,5"/>',
        f'<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="3"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 28}" text-anchor="middle" font-size="14">False Positive Rate</text>',
        f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 22 {top + plot_height / 2})">True Positive Rate</text>',
        f'<text x="{left + 12}" y="{top + 24}" font-size="13">ROC-AUC = {auc:.3f}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_calibration_curve(y_true, probabilities, save_path, n_bins=10, title="Calibration Curve"):
    """Save a binary reliability diagram as an SVG file."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")

    bin_ids = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    predicted, observed = [], []
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if np.any(in_bin):
            predicted.append(float(np.mean(probabilities[in_bin])))
            observed.append(float(np.mean(y_true[in_bin])))

    width, height = 640, 520
    left, top, plot_width, plot_height = 80, 55, 500, 390
    x = lambda value: left + value * plot_width
    y = lambda value: top + (1 - value) * plot_height
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = tick / 5
        svg.extend([
            f'<line x1="{x(value)}" y1="{top}" x2="{x(value)}" y2="{top + plot_height}" stroke="#dddddd"/>',
            f'<line x1="{left}" y1="{y(value)}" x2="{left + plot_width}" y2="{y(value)}" stroke="#dddddd"/>',
            f'<text x="{x(value)}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12">{value:.1f}</text>',
            f'<text x="{left - 12}" y="{y(value) + 4}" text-anchor="end" font-size="12">{value:.1f}</text>',
        ])

    points = " ".join(
        f"{x(predicted_value):.1f},{y(observed_value):.1f}"
        for predicted_value, observed_value in zip(predicted, observed)
    )
    svg.extend([
        f'<line x1="{left}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#777777" stroke-dasharray="6,5"/>',
        f'<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="2"/>',
    ])
    svg.extend(
        f'<circle cx="{x(predicted_value):.1f}" cy="{y(observed_value):.1f}" r="5" fill="#1769aa"/>'
        for predicted_value, observed_value in zip(predicted, observed)
    )
    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 28}" text-anchor="middle" font-size="14">Mean predicted failure probability</text>',
        f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 22 {top + plot_height / 2})">Observed failure rate</text>',
        '</svg>',
    ])
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def _comparison_svg(title, x_label, y_label):
    width, height = 800, 520
    left, top, plot_width, plot_height = 70, 55, 500, 390
    x = lambda value: left + value * plot_width
    y = lambda value: top + (1 - value) * plot_height
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = tick / 5
        svg.extend([
            f'<line x1="{x(value)}" y1="{top}" x2="{x(value)}" y2="{top + plot_height}" stroke="#dddddd"/>',
            f'<line x1="{left}" y1="{y(value)}" x2="{left + plot_width}" y2="{y(value)}" stroke="#dddddd"/>',
            f'<text x="{x(value)}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12">{value:.1f}</text>',
            f'<text x="{left - 10}" y="{y(value) + 4}" text-anchor="end" font-size="12">{value:.1f}</text>',
        ])
    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 24}" text-anchor="middle" font-size="14">{escape(x_label)}</text>',
        f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {top + plot_height / 2})">{escape(y_label)}</text>',
    ])
    return svg, x, y


def _roc_points(y_true, probabilities):
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    labels = y_true[order]
    scores = probabilities[order]
    true_positives = np.cumsum(labels == 1)
    false_positives = np.cumsum(labels == 0)
    last_in_tie = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    return (
        np.r_[0, false_positives[last_in_tie] / negatives],
        np.r_[0, true_positives[last_in_tie] / positives],
    )


def plot_roc_comparison(curves, save_path, title="ROC Curves Across Shifts"):
    """Overlay ROC curves; curves maps shift names to (labels, probabilities)."""
    colors = ["#1769aa", "#d1495b", "#2a9d8f", "#f4a261", "#7b2cbf", "#6c757d"]
    svg, x, y = _comparison_svg(title, "False Positive Rate", "True Positive Rate")
    svg.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#777777" stroke-dasharray="6,5"/>')

    for index, (name, (labels, probabilities)) in enumerate(curves.items()):
        labels = np.asarray(labels, dtype=int).reshape(-1)
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        points = _roc_points(labels, probabilities)
        if points is None:
            continue
        fpr, tpr = points
        color = colors[index % len(colors)]
        coordinates = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(fpr, tpr))
        auc = _roc_auc(labels, probabilities)
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = 82 + index * 23
        svg.append(f'<line x1="600" y1="{legend_y}" x2="620" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="628" y="{legend_y + 4}" font-size="12">{escape(name)} (AUC {auc:.3f})</text>')

    svg.append("</svg>")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_calibration_comparison(curves, save_path, n_bins=10, title="Calibration Across Shifts"):
    """Overlay reliability curves; curves maps shift names to (labels, probabilities)."""
    colors = ["#1769aa", "#d1495b", "#2a9d8f", "#f4a261", "#7b2cbf", "#6c757d"]
    svg, x, y = _comparison_svg(
        title,
        "Mean predicted failure probability",
        "Observed failure rate",
    )
    svg.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#777777" stroke-dasharray="6,5"/>')

    for index, (name, (labels, probabilities)) in enumerate(curves.items()):
        labels = np.asarray(labels, dtype=int).reshape(-1)
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        bin_ids = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
        predicted, observed = [], []
        for bin_id in range(n_bins):
            in_bin = bin_ids == bin_id
            if np.any(in_bin):
                predicted.append(float(np.mean(probabilities[in_bin])))
                observed.append(float(np.mean(labels[in_bin])))
        color = colors[index % len(colors)]
        coordinates = " ".join(
            f"{x(a):.1f},{y(b):.1f}" for a, b in zip(predicted, observed)
        )
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        svg.extend(
            f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4" fill="{color}"/>'
            for a, b in zip(predicted, observed)
        )
        legend_y = 82 + index * 23
        svg.append(f'<line x1="600" y1="{legend_y}" x2="620" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="628" y="{legend_y + 4}" font-size="12">{escape(name)}</text>')

    svg.append("</svg>")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return path
