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


def calculate_metrics(y_true, probabilities, n_bins=10):
    """Calculate binary classification metrics from positive-class probabilities."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")

    predictions = (probabilities >= 0.5).astype(int)
    true_positive = np.sum((predictions == 1) & (y_true == 1))
    false_positive = np.sum((predictions == 1) & (y_true == 0))
    false_negative = np.sum((predictions == 0) & (y_true == 1))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = np.mean(predictions == y_true) if len(y_true) else 0.0

    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = (predictions == y_true).astype(float)
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if np.any(in_bin):
            ece += np.mean(in_bin) * abs(np.mean(correct[in_bin]) - np.mean(confidence[in_bin]))

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "roc_auc": _roc_auc(y_true, probabilities),
        "expected_calibration_error": float(ece),
    }
