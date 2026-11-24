from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler


COLORS = ["#1769aa", "#d1495b", "#2a9d8f", "#f4a261", "#7b2cbf", "#6c757d"]
DEFAULT_SUBPOP_SIZES = (1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1)


def compute_loss(y_true, probabilities, loss="bce", threshold=0.5):
    """Return pointwise loss values for each example."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("y_true and probabilities must have the same length")

    if loss == "bce":
        p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        return -(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
    if loss == "brier":
        return (y - p) ** 2
    if loss == "zero_one":
        return ((p >= threshold).astype(float) != y).astype(float)
    raise ValueError(f"Unknown loss: {loss}")


def _encode(df, cols, reference_columns=None):
    encoded = pd.get_dummies(df[list(cols)], drop_first=False).astype(float)
    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0.0)
    return encoded.to_numpy(), encoded.columns


def _is_discrete_df(df, cols):
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            return False
    return True


def _fit_krr(X_tr, y_tr, X_va):
    """Fit standardized RBF Kernel Ridge Regression (Subbaswamy et al., Appendix F.3)."""
    if len(X_tr) < 5:
        return np.full(len(X_va), float(np.mean(y_tr)), dtype=float)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    gamma = 1.0 / max(X_tr_s.shape[1], 1)
    reg = KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma)
    reg.fit(X_tr_s, y_tr)
    return reg.predict(X_va_s)


def _fit_mu_fold(df_tr, y_tr, df_va, wz_cols, split_col=None):
    """Estimate mu_k(W, Z) on I_k^c (returning train-side and held-out I_k predictions)."""
    if split_col is not None and split_col in wz_cols:
        other_cols = [c for c in wz_cols if c != split_col]
        mu_tr = np.empty(len(df_tr), dtype=float)
        mu_va = np.empty(len(df_va), dtype=float)
        for val in df_tr[split_col].unique():
            tr_mask = (df_tr[split_col] == val).to_numpy()
            va_mask = (df_va[split_col] == val).to_numpy()
            if not other_cols:
                mean_val = float(np.mean(y_tr[tr_mask]))
                mu_tr[tr_mask] = mean_val
                mu_va[va_mask] = mean_val
                continue
            X_tr_g, ref = _encode(df_tr.iloc[tr_mask], other_cols)
            X_va_g, _ = _encode(df_va.iloc[va_mask], other_cols, reference_columns=ref)
            mu_tr[tr_mask] = _fit_krr(X_tr_g, y_tr[tr_mask], X_tr_g)
            if np.any(va_mask):
                mu_va[va_mask] = _fit_krr(X_tr_g, y_tr[tr_mask], X_va_g)
        return mu_tr, mu_va

    X_tr, ref = _encode(df_tr, wz_cols)
    X_va, _ = _encode(df_va, wz_cols, reference_columns=ref)
    return _fit_krr(X_tr, y_tr, X_tr), _fit_krr(X_tr, y_tr, X_va)


def _fit_eta_fold(df_tr, target_tr, df_va, immutable_cols, alpha, fallback_col=None):
    """Estimate conditional alpha-quantile eta_k(Z) (Appendix F.3: spline QR or discrete strata)."""
    if not immutable_cols:
        return np.full(len(df_va), float(np.quantile(target_tr, alpha)), dtype=float)

    if _is_discrete_df(df_tr, immutable_cols):
        tr_keys = df_tr[immutable_cols].astype(str).agg("|".join, axis=1)
        va_keys = df_va[immutable_cols].astype(str).agg("|".join, axis=1)
        global_q = float(np.quantile(target_tr, alpha))

        fallback_map = {}
        if fallback_col is not None and fallback_col in immutable_cols:
            for val, idx in df_tr.groupby(fallback_col).groups.items():
                fallback_map[str(val)] = float(np.quantile(target_tr[idx], alpha))

        group_q = {}
        for key, idx in tr_keys.groupby(tr_keys).groups.items():
            if len(idx) >= 12:
                group_q[key] = float(np.quantile(target_tr[idx], alpha))

        preds = np.empty(len(df_va), dtype=float)
        for i, key in enumerate(va_keys):
            if key in group_q:
                preds[i] = group_q[key]
            elif fallback_col is not None and fallback_col in immutable_cols:
                preds[i] = fallback_map.get(str(df_va[fallback_col].iloc[i]), global_q)
            else:
                preds[i] = global_q
        return preds

    # Continuous Z: B-spline basis expansion + L1-regularized QuantileRegressor (Appendix F.3)
    X_z_tr, z_ref = _encode(df_tr, immutable_cols)
    X_z_va, _ = _encode(df_va, immutable_cols, reference_columns=z_ref)
    scaler = StandardScaler()
    X_z_tr_s = scaler.fit_transform(X_z_tr)
    X_z_va_s = scaler.transform(X_z_va)
    spline = SplineTransformer(n_knots=4, degree=2, include_bias=False)
    B_tr = spline.fit_transform(X_z_tr_s)
    B_va = spline.transform(X_z_va_s)

    qr = QuantileRegressor(quantile=float(alpha), alpha=1e-3, solver="highs")
    qr.fit(B_tr, target_tr)
    return qr.predict(B_va)


def _subpopulation_profile(df, mask, mutable_cols, immutable_cols):
    """Summarize mutable sensor shifts and immutable-variable preservation on the selected subsample."""
    selected = df.iloc[mask]
    profile = {"selected_count": int(mask.sum()), "mutable": {}, "immutable": {}}

    for group_name, cols in (("mutable", mutable_cols), ("immutable", immutable_cols)):
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                full_mean = float(df[col].mean())
                full_std = float(df[col].std())
                sel_mean = float(selected[col].mean()) if len(selected) else full_mean
                sel_std = float(selected[col].std()) if len(selected) > 1 else 0.0
                z_shift = (sel_mean - full_mean) / full_std if full_std > 0 else 0.0
                std_ratio = sel_std / full_std if full_std > 0 else 1.0
                profile[group_name][col] = {
                    "full_mean": full_mean,
                    "selected_mean": sel_mean,
                    "z_shift": float(z_shift),
                    "full_std": full_std,
                    "selected_std": sel_std,
                    "std_ratio": float(std_ratio),
                }
            else:
                full_dist = df[col].value_counts(normalize=True).to_dict()
                sel_dist = selected[col].value_counts(normalize=True).to_dict() if len(selected) else {}
                profile[group_name][col] = {
                    "full_proportions": {str(k): float(v) for k, v in full_dist.items()},
                    "selected_proportions": {str(k): float(sel_dist.get(k, 0.0)) for k in full_dist},
                }
    return profile


def estimate_worst_case_risk(
    df,
    losses,
    mutable_cols,
    immutable_cols=(),
    subpop_sizes=DEFAULT_SUBPOP_SIZES,
    n_splits=5,
    discrete_noise=1e-5,
    stratify_col=None,
    random_state=42,
):
    """Estimate worst-case (1 - alpha) risk via Subbaswamy et al. (2021) Algorithm 1 / Algorithm 2."""
    df = df.reset_index(drop=True)
    losses = np.asarray(losses, dtype=float).reshape(-1)
    n = len(df)
    if len(losses) != n:
        raise ValueError("df and losses must have the same length")

    mutable_cols = list(mutable_cols)
    immutable_cols = list(immutable_cols)
    wz_cols = mutable_cols + immutable_cols

    if stratify_col is not None and stratify_col in df.columns:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(splitter.split(df, df[stratify_col]))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(splitter.split(df))

    rng = np.random.default_rng(random_state)
    noise = rng.uniform(0.0, discrete_noise, size=n) if discrete_noise > 0 else np.zeros(n)

    # Fit conditional expected loss mu_k(W, Z) once per fold (independent of alpha)
    mu_oof = np.empty(n, dtype=float)
    fold_cache = []
    for train_idx, val_idx in folds:
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_va = df.iloc[val_idx].reset_index(drop=True)
        mu_tr, mu_va = _fit_mu_fold(df_tr, losses[train_idx], df_va, wz_cols, split_col=stratify_col)
        mu_oof[val_idx] = mu_va
        fold_cache.append((train_idx, val_idx, df_tr, df_va, mu_tr))

    points = []
    masks = {}
    for prop in subpop_sizes:
        prop = float(prop)
        alpha = 1.0 - prop

        if np.isclose(prop, 1.0):
            risk = float(np.mean(losses))
            se = float(np.sqrt(np.mean((losses - risk) ** 2) / n))
            mask = np.ones(n, dtype=bool)
        else:
            eta_oof = np.empty(n, dtype=float)
            for train_idx, val_idx, df_tr, df_va, mu_tr in fold_cache:
                target = mu_tr + noise[train_idx]
                eta_oof[val_idx] = _fit_eta_fold(
                    df_tr, target, df_va, immutable_cols, alpha, fallback_col=stratify_col
                )

            mask = (mu_oof + noise) > eta_oof
            h = mask.astype(float)
            phi = (np.maximum(mu_oof + noise - eta_oof, 0.0) + h * (losses - mu_oof)) / prop + eta_oof
            risk = float(np.mean(phi))
            se = float(np.sqrt(np.mean((phi - risk) ** 2) / n))

        masks[prop] = mask
        points.append({
            "subpop_size": prop,
            "alpha": float(alpha),
            "worst_case_risk": risk,
            "std_error": se,
            "ci_low": float(risk - 1.96 * se),
            "ci_high": float(risk + 1.96 * se),
            "selected_fraction": float(np.mean(mask)),
            "profile": _subpopulation_profile(df, mask, mutable_cols, immutable_cols),
        })

    return {
        "mutable_cols": mutable_cols,
        "immutable_cols": immutable_cols,
        "n_samples": n,
        "n_splits": n_splits,
        "discrete_noise": float(discrete_noise),
        "points": points,
    }, masks


def save_stability_json(result, save_path):
    """Write stability analysis results to a JSON file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    return path


def _extract_series(result, as_accuracy=False):
    pts = sorted(result["points"], key=lambda item: item["subpop_size"])
    xs = [p["subpop_size"] for p in pts]
    if as_accuracy:
        ys = [1.0 - p["worst_case_risk"] for p in pts]
        lows = [1.0 - p["ci_high"] for p in pts]
        highs = [1.0 - p["ci_low"] for p in pts]
    else:
        ys = [p["worst_case_risk"] for p in pts]
        lows = [p["ci_low"] for p in pts]
        highs = [p["ci_high"] for p in pts]
    return xs, ys, lows, highs


def _stability_canvas(width, height, left, top, plot_width, plot_height, title, x_label, y_label, y_min, y_max):
    x_map = lambda v: left + v * plot_width
    y_span = max(y_max - y_min, 1e-6)
    y_map = lambda v: top + (1.0 - (v - y_min) / y_span) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{escape(title)}</text>',
    ]
    for tick in range(6):
        x_val = tick / 5
        y_val = y_min + (tick / 5) * y_span
        svg.extend([
            f'<line x1="{x_map(x_val):.1f}" y1="{top}" x2="{x_map(x_val):.1f}" y2="{top + plot_height}" stroke="#dddddd"/>',
            f'<line x1="{left}" y1="{y_map(y_val):.1f}" x2="{left + plot_width}" y2="{y_map(y_val):.1f}" stroke="#dddddd"/>',
            f'<text x="{x_map(x_val):.1f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12">{x_val:.1f}</text>',
            f'<text x="{left - 10}" y="{y_map(y_val) + 4:.1f}" text-anchor="end" font-size="12">{y_val:.3f}</text>',
        ])
    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 24}" text-anchor="middle" font-size="14">{escape(x_label)}</text>',
        f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {top + plot_height / 2})">{escape(y_label)}</text>',
    ])
    return svg, x_map, y_map


def plot_stability_curve(
    result,
    save_path,
    title="Worst-Case Stability Curve",
    y_label="Worst-Case Loss",
    as_accuracy=False,
    observed_lines=None,
):
    """Save a single-model stability curve with 95% CI band as SVG."""
    xs, ys, lows, highs = _extract_series(result, as_accuracy=as_accuracy)
    all_y = lows + highs + (list(observed_lines.values()) if observed_lines else [])
    pad = 0.08 * max(max(all_y) - min(all_y), 1e-3)
    y_min = max(0.0, min(all_y) - pad)
    y_max = min(1.0, max(all_y) + pad) if as_accuracy else max(all_y) + pad

    svg, x_map, y_map = _stability_canvas(
        760, 520, 82, 55, 480, 390, title, "Subpopulation Size (1 - alpha)", y_label, y_min, y_max
    )

    band_pts = [f"{x_map(a):.1f},{y_map(b):.1f}" for a, b in zip(xs, highs)]
    band_pts += [f"{x_map(a):.1f},{y_map(b):.1f}" for a, b in zip(reversed(xs), reversed(lows))]
    line_pts = " ".join(f"{x_map(a):.1f},{y_map(b):.1f}" for a, b in zip(xs, ys))
    svg.append(f'<polygon points="{" ".join(band_pts)}" fill="#1769aa" fill-opacity="0.18" stroke="none"/>')
    svg.append(f'<polyline points="{line_pts}" fill="none" stroke="#1769aa" stroke-width="2.5"/>')
    for a, b in zip(xs, ys):
        svg.append(f'<circle cx="{x_map(a):.1f}" cy="{y_map(b):.1f}" r="4" fill="#1769aa"/>')

    svg.append('<line x1="585" y1="82" x2="605" y2="82" stroke="#1769aa" stroke-width="3"/>')
    svg.append('<text x="613" y="86" font-size="12">Worst-case (95% CI)</text>')

    if observed_lines:
        for idx, (label, val) in enumerate(observed_lines.items(), start=1):
            color = COLORS[idx % len(COLORS)]
            yv = y_map(val)
            svg.append(
                f'<line x1="82" y1="{yv:.1f}" x2="562" y2="{yv:.1f}" stroke="{color}" stroke-width="1.8" stroke-dasharray="6,4"/>'
            )
            ly = 82 + idx * 23
            svg.append(f'<line x1="585" y1="{ly}" x2="605" y2="{ly}" stroke="{color}" stroke-width="2.5" stroke-dasharray="5,3"/>')
            svg.append(f'<text x="613" y="{ly + 4}" font-size="11">{escape(label)} ({val:.3f})</text>')

    svg.append("</svg>")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_stability_comparison(
    results_by_model,
    save_path,
    title="Model Stability Comparison",
    y_label="Worst-Case Loss",
    as_accuracy=False,
):
    """Overlay worst-case stability curves with 95% CI bands across models."""
    series = {name: _extract_series(res, as_accuracy=as_accuracy) for name, res in results_by_model.items()}
    all_y = [v for _, _, lows, highs in series.values() for v in (lows + highs)]
    pad = 0.08 * max(max(all_y) - min(all_y), 1e-3)
    y_min = max(0.0, min(all_y) - pad)
    y_max = min(1.0, max(all_y) + pad) if as_accuracy else max(all_y) + pad

    svg, x_map, y_map = _stability_canvas(
        820, 520, 82, 55, 490, 390, title, "Subpopulation Size (1 - alpha)", y_label, y_min, y_max
    )

    for idx, (name, (xs, ys, lows, highs)) in enumerate(series.items()):
        color = COLORS[idx % len(COLORS)]
        band_pts = [f"{x_map(a):.1f},{y_map(b):.1f}" for a, b in zip(xs, highs)]
        band_pts += [f"{x_map(a):.1f},{y_map(b):.1f}" for a, b in zip(reversed(xs), reversed(lows))]
        line_pts = " ".join(f"{x_map(a):.1f},{y_map(b):.1f}" for a, b in zip(xs, ys))
        svg.append(f'<polygon points="{" ".join(band_pts)}" fill="{color}" fill-opacity="0.14" stroke="none"/>')
        svg.append(f'<polyline points="{line_pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for a, b in zip(xs, ys):
            svg.append(f'<circle cx="{x_map(a):.1f}" cy="{y_map(b):.1f}" r="3.5" fill="{color}"/>')

        ly = 82 + idx * 24
        svg.append(f'<line x1="595" y1="{ly}" x2="615" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="623" y="{ly + 4}" font-size="11">{escape(name)}</text>')

    svg.append("</svg>")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_subpopulation_profile(result, save_path, title="Worst-Case Subpopulation Profile (Z-Shift)"):
    """Plot standardized mean shifts of numeric mutable & immutable variables across (1 - alpha)."""
    pts = sorted(result["points"], key=lambda item: item["subpop_size"])
    xs = [p["subpop_size"] for p in pts]

    tracked = []
    for group in ("mutable", "immutable"):
        for col, stats in pts[0]["profile"][group].items():
            if "z_shift" in stats:
                tracked.append((group, col))
    if not tracked:
        return None

    series = {}
    for group, col in tracked:
        tag = f"[W] {col}" if group == "mutable" else f"[Z] {col}"
        series[tag] = (group, [p["profile"][group][col]["z_shift"] for p in pts])

    all_y = [v for _, ys in series.values() for v in ys] + [-0.2, 0.2]
    pad = 0.1 * max(max(all_y) - min(all_y), 0.5)
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    svg, x_map, y_map = _stability_canvas(
        860, 520, 82, 55, 490, 390, title, "Subpopulation Size (1 - alpha)", "Mean Shift (Std. Devs)", y_min, y_max
    )
    svg.append(f'<line x1="82" y1="{y_map(0.0):.1f}" x2="572" y2="{y_map(0.0):.1f}" stroke="#777777" stroke-dasharray="6,5"/>')

    for idx, (label, (group, ys)) in enumerate(series.items()):
        color = COLORS[idx % len(COLORS)]
        dash = ' stroke-dasharray="5,4"' if group == "immutable" else ""
        coords = " ".join(f"{x_map(a):.1f},{y_map(b):.1f}" for a, b in zip(xs, ys))
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.3"{dash}/>')
        ly = 82 + idx * 23
        svg.append(f'<line x1="595" y1="{ly}" x2="615" y2="{ly}" stroke="{color}" stroke-width="2.5"{dash}/>')
        svg.append(f'<text x="623" y="{ly + 4}" font-size="11">{escape(label)}</text>')

    svg.append("</svg>")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return path
