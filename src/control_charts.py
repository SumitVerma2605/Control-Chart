"""
control_charts.py
-----------------
Statistical Process Control (SPC) charts module for DataFlow Analytics.
Implements: X-bar, R, S, Individuals (I-MR), P, and C charts with:
  - Auto-calculated control limits (UCL, CL, LCL)
  - Out-of-control point highlighting
  - Western Electric rules (runs, trends, cycles)
  - Interactive Plotly charts
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── SPC Constants (d2, D3, D4, A2, A3, c4, B3, B4) ──────────────────────────
# Indexed by subgroup size n (index 0 = n=1, index 1 = n=2, ...)
SPC_CONSTANTS = {
    "d2": [1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078],
    "D3": [0,     0,     0,     0,     0,     0.076, 0.136, 0.184, 0.223],
    "D4": [3.267, 2.574, 2.282, 2.114, 2.004, 1.924, 1.864, 1.816, 1.777],
    "A2": [1.880, 1.023, 0.729, 0.577, 0.483, 0.419, 0.373, 0.337, 0.308],
    "A3": [2.659, 1.954, 1.628, 1.427, 1.287, 1.182, 1.099, 1.032, 0.975],
    "c4": [0.798, 0.886, 0.921, 0.940, 0.952, 0.959, 0.965, 0.969, 0.973],
    "B3": [0,     0,     0,     0,     0.030, 0.118, 0.185, 0.239, 0.284],
    "B4": [3.267, 2.568, 2.266, 2.089, 1.970, 1.882, 1.815, 1.761, 1.716],
}


def _get_const(name: str, n: int) -> float:
    """Retrieve SPC constant for subgroup size n (clamps to table range 2–10)."""
    n = max(2, min(n, 10))
    return SPC_CONSTANTS[name][n - 2]


def _detect_violations(values: np.ndarray, ucl: float, lcl: float, cl: float) -> Dict:
    """
    Apply Western Electric Rules to detect out-of-control signals.
    Returns dict of {rule_name: [point_indices]}.
    """
    violations = {}
    n = len(values)

    # Rule 1: Single point beyond 3σ
    r1 = [i for i in range(n) if values[i] > ucl or values[i] < lcl]
    if r1:
        violations["Rule 1: Beyond control limits"] = r1

    # Rule 2: 8+ consecutive points same side of center line
    run_same_side = []
    count, side = 0, None
    for i, v in enumerate(values):
        cur_side = "above" if v > cl else "below"
        if cur_side == side:
            count += 1
            if count >= 8:
                run_same_side.append(i)
        else:
            count = 1
            side = cur_side
    if run_same_side:
        violations["Rule 2: 8 pts same side of CL"] = run_same_side

    # Rule 3: 6+ consecutive strictly ascending or descending
    trend_pts = []
    for i in range(5, n):
        seg = values[i - 5:i + 1]
        if all(seg[j] < seg[j + 1] for j in range(5)) or all(seg[j] > seg[j + 1] for j in range(5)):
            trend_pts.append(i)
    if trend_pts:
        violations["Rule 3: 6-pt trend"] = trend_pts

    # Rule 4: 2 of 3 beyond 2σ same side
    sigma = (ucl - cl) / 3 if ucl != cl else 1e-9
    two_sigma_pts = []
    for i in range(2, n):
        seg = values[i - 2:i + 1]
        above = sum(1 for v in seg if v > cl + 2 * sigma)
        below = sum(1 for v in seg if v < cl - 2 * sigma)
        if above >= 2 or below >= 2:
            two_sigma_pts.append(i)
    if two_sigma_pts:
        violations["Rule 4: 2 of 3 beyond 2σ"] = two_sigma_pts

    return violations


def _base_chart(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        hovermode="x unified",
        height=420,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _add_control_lines(fig, ucl, cl, lcl, row=None, col=None):
    """Add UCL, CL, LCL horizontal lines to a figure."""
    kw = dict(row=row, col=col) if row else {}
    for val, label, color, dash in [
        (ucl, f"UCL={ucl:.4f}", "#ef4444", "dash"),
        (cl,  f"CL={cl:.4f}",  "#22c55e", "solid"),
        (lcl, f"LCL={lcl:.4f}", "#3b82f6", "dash"),
    ]:
        fig.add_hline(y=val, line_dash=dash, line_color=color,
                      annotation_text=label,
                      annotation_position="right", **kw)


# ─────────────────────────────────────────────
# X-bar & R Chart
# ─────────────────────────────────────────────

def xbar_r_chart(
    df: pd.DataFrame,
    value_col: str,
    subgroup_col: str,
    sigma: float = 3.0,
) -> Tuple[go.Figure, Dict]:
    """
    X-bar and R chart for subgrouped data.
    
    Args:
        df: DataFrame with the measurement and subgroup columns.
        value_col: Column name for the measured values.
        subgroup_col: Column name defining subgroups (sample ID or datetime).
        sigma: Control limit width in sigma units (default 3).
    
    Returns:
        (fig, stats_dict)
    """
    grouped = df.groupby(subgroup_col)[value_col]
    means = grouped.mean()
    ranges = grouped.apply(lambda x: x.max() - x.min())
    sizes = grouped.size()
    n = int(sizes.median())

    R_bar = ranges.mean()
    X_dbar = means.mean()
    d2 = _get_const("d2", n)
    A2 = _get_const("A2", n)
    D3 = _get_const("D3", n)
    D4 = _get_const("D4", n)

    ucl_x  = X_dbar + A2 * R_bar
    lcl_x  = X_dbar - A2 * R_bar
    ucl_r  = D4 * R_bar
    lcl_r  = D3 * R_bar

    x_vals = means.values
    r_vals = ranges.values
    labels = means.index.tolist()

    viol_x = _detect_violations(x_vals, ucl_x, X_dbar, X_dbar)
    viol_r = _detect_violations(r_vals, ucl_r, R_bar, R_bar)

    fig = make_subplots(rows=2, cols=1, subplot_titles=["X-bar Chart", "R Chart"], vertical_spacing=0.12)

    # X-bar
    for name, val, color, dash in [(f"UCL={ucl_x:.4f}", ucl_x, "#ef4444", "dash"),
                                    (f"CL={X_dbar:.4f}", X_dbar, "#22c55e", "solid"),
                                    (f"LCL={lcl_x:.4f}", lcl_x, "#3b82f6", "dash")]:
        fig.add_hline(y=val, line_dash=dash, line_color=color, annotation_text=name,
                      annotation_position="right", row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(labels))), y=x_vals, mode="lines+markers",
                             name="X-bar", marker=dict(color="#a78bfa", size=6),
                             customdata=labels, hovertemplate="%{customdata}<br>X-bar=%{y:.4f}"), row=1, col=1)

    viol_idx_x = list(set(sum(viol_x.values(), [])))
    if viol_idx_x:
        fig.add_trace(go.Scatter(x=viol_idx_x, y=x_vals[viol_idx_x], mode="markers",
                                 marker=dict(color="#ef4444", size=10, symbol="circle-open", line=dict(width=2)),
                                 name="X-bar OOC"), row=1, col=1)

    # R chart
    for name, val, color, dash in [(f"UCL={ucl_r:.4f}", ucl_r, "#ef4444", "dash"),
                                    (f"CL={R_bar:.4f}", R_bar, "#22c55e", "solid"),
                                    (f"LCL={lcl_r:.4f}", lcl_r, "#3b82f6", "dash")]:
        fig.add_hline(y=val, line_dash=dash, line_color=color, annotation_text=name,
                      annotation_position="right", row=2, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(labels))), y=r_vals, mode="lines+markers",
                             name="Range", marker=dict(color="#34d399", size=6)), row=2, col=1)

    viol_idx_r = list(set(sum(viol_r.values(), [])))
    if viol_idx_r:
        fig.add_trace(go.Scatter(x=viol_idx_r, y=r_vals[viol_idx_r], mode="markers",
                                 marker=dict(color="#ef4444", size=10, symbol="circle-open", line=dict(width=2)),
                                 name="Range OOC"), row=2, col=1)

    fig.update_layout(title="X-bar & R Chart", template="plotly_dark", height=700, hovermode="x unified")

    stats = {
        "X_double_bar": X_dbar, "R_bar": R_bar,
        "UCL_X": ucl_x, "LCL_X": lcl_x,
        "UCL_R": ucl_r, "LCL_R": lcl_r,
        "n_subgroups": len(means),
        "violations_xbar": {k: len(v) for k, v in viol_x.items()},
        "violations_r": {k: len(v) for k, v in viol_r.items()},
        "Cp_estimate": (ucl_x - lcl_x) / (6 * R_bar / d2) if R_bar > 0 else None,
    }
    return fig, stats


# ─────────────────────────────────────────────
# X-bar & S Chart
# ─────────────────────────────────────────────

def xbar_s_chart(
    df: pd.DataFrame,
    value_col: str,
    subgroup_col: str,
) -> Tuple[go.Figure, Dict]:
    """X-bar and S (standard deviation) chart."""
    grouped = df.groupby(subgroup_col)[value_col]
    means = grouped.mean()
    stds = grouped.std(ddof=1).fillna(0)
    sizes = grouped.size()
    n = int(sizes.median())

    S_bar = stds.mean()
    X_dbar = means.mean()
    c4 = _get_const("c4", n)
    A3 = _get_const("A3", n)
    B3 = _get_const("B3", n)
    B4 = _get_const("B4", n)

    ucl_x = X_dbar + A3 * S_bar
    lcl_x = X_dbar - A3 * S_bar
    ucl_s = B4 * S_bar
    lcl_s = B3 * S_bar

    x_vals = means.values
    s_vals = stds.values
    labels = means.index.tolist()

    viol_x = _detect_violations(x_vals, ucl_x, X_dbar, X_dbar)
    viol_s = _detect_violations(s_vals, ucl_s, S_bar, S_bar)

    fig = make_subplots(rows=2, cols=1, subplot_titles=["X-bar Chart", "S Chart"], vertical_spacing=0.12)

    for val, name, color, dash in [(ucl_x, f"UCL={ucl_x:.4f}", "#ef4444", "dash"),
                                    (X_dbar, f"CL={X_dbar:.4f}", "#22c55e", "solid"),
                                    (lcl_x, f"LCL={lcl_x:.4f}", "#3b82f6", "dash")]:
        fig.add_hline(y=val, line_dash=dash, line_color=color, annotation_text=name,
                      annotation_position="right", row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(labels))), y=x_vals, mode="lines+markers",
                             name="X-bar", marker=dict(color="#a78bfa", size=6)), row=1, col=1)

    viol_idx_x = list(set(sum(viol_x.values(), [])))
    if viol_idx_x:
        fig.add_trace(go.Scatter(x=viol_idx_x, y=x_vals[viol_idx_x], mode="markers",
                                 marker=dict(color="#ef4444", size=10, symbol="circle-open", line=dict(width=2)),
                                 name="X-bar OOC"), row=1, col=1)

    for val, name, color, dash in [(ucl_s, f"UCL={ucl_s:.4f}", "#ef4444", "dash"),
                                    (S_bar, f"CL={S_bar:.4f}", "#22c55e", "solid"),
                                    (lcl_s, f"LCL={lcl_s:.4f}", "#3b82f6", "dash")]:
        fig.add_hline(y=val, line_dash=dash, line_color=color, annotation_text=name,
                      annotation_position="right", row=2, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(labels))), y=s_vals, mode="lines+markers",
                             name="Std Dev", marker=dict(color="#34d399", size=6)), row=2, col=1)

    viol_idx_s = list(set(sum(viol_s.values(), [])))
    if viol_idx_s:
        fig.add_trace(go.Scatter(x=viol_idx_s, y=s_vals[viol_idx_s], mode="markers",
                                 marker=dict(color="#ef4444", size=10, symbol="circle-open", line=dict(width=2)),
                                 name="S OOC"), row=2, col=1)

    fig.update_layout(title="X-bar & S Chart", template="plotly_dark", height=700, hovermode="x unified")

    stats = {
        "X_double_bar": X_dbar, "S_bar": S_bar,
        "UCL_X": ucl_x, "LCL_X": lcl_x,
        "UCL_S": ucl_s, "LCL_S": lcl_s,
        "n_subgroups": len(means),
        "violations_xbar": {k: len(v) for k, v in viol_x.items()},
        "violations_s": {k: len(v) for k, v in viol_s.items()},
    }
    return fig, stats


# ─────────────────────────────────────────────
# Individuals (I-MR) Chart
# ─────────────────────────────────────────────

def imr_chart(
    df: pd.DataFrame,
    value_col: str,
    order_col: Optional[str] = None,
) -> Tuple[go.Figure, Dict]:
    """
    Individuals and Moving Range (I-MR) chart for individual measurements.
    """
    data = df.copy()
    if order_col and order_col in data.columns:
        data = data.sort_values(order_col)

    X = data[value_col].dropna().reset_index(drop=True)
    MR = X.diff().abs()

    X_bar = X.mean()
    MR_bar = MR.mean()
    d2 = 1.128  # n=2

    sigma_hat = MR_bar / d2
    ucl_i  = X_bar + 3 * sigma_hat
    lcl_i  = X_bar - 3 * sigma_hat
    ucl_mr = 3.267 * MR_bar
    lcl_mr = 0.0

    x_arr = X.values
    mr_arr = MR.values

    viol_i  = _detect_violations(x_arr, ucl_i, X_bar, X_bar)
    viol_mr = _detect_violations(mr_arr[1:], ucl_mr, MR_bar, MR_bar)

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Individuals (I) Chart", "Moving Range (MR) Chart"],
                        vertical_spacing=0.12)

    for val, name, color, dash in [(ucl_i, f"UCL={ucl_i:.4f}", "#ef4444", "dash"),
                                    (X_bar, f"CL={X_bar:.4f}", "#22c55e", "solid"),
                                    (lcl_i, f"LCL={lcl_i:.4f}", "#3b82f6", "dash")]:
        fig.add_hline(y=val, line_dash=dash, line_color=color, annotation_text=name,
                      annotation_position="right", row=1, col=1)

    fig.add_trace(go.Scatter(y=x_arr, mode="lines+markers", name="Individual",
                             marker=dict(color="#a78bfa", size=5)), row=1, col=1)

    viol_idx_i = list(set(sum(viol_i.values(), [])))
    if viol_idx_i:
        fig.add_trace(go.Scatter(x=viol_idx_i, y=x_arr[viol_idx_i], mode="markers",
                                 marker=dict(color="#ef4444", size=10, symbol="circle-open", line=dict(width=2)),
                                 name="I OOC"), row=1, col=1)

    for val, name, color, dash in [(ucl_mr, f"UCL={ucl_mr:.4f}", "#ef4444", "dash"),
                                    (MR_bar, f"CL={MR_bar:.4f}", "#22c55e", "solid"),
                                    (0, "LCL=0", "#3b82f6", "dash")]:
        fig.add_hline(y=val, line_dash=dash, line_color=color, annotation_text=name,
                      annotation_position="right", row=2, col=1)

    fig.add_trace(go.Bar(y=mr_arr, name="Moving Range", marker_color="#34d399"), row=2, col=1)

    fig.update_layout(title="Individuals & Moving Range (I-MR) Chart",
                      template="plotly_dark", height=700, hovermode="x unified")

    stats = {
        "X_bar": X_bar, "MR_bar": MR_bar, "sigma_hat": sigma_hat,
        "UCL_I": ucl_i, "LCL_I": lcl_i,
        "UCL_MR": ucl_mr, "LCL_MR": lcl_mr,
        "n_points": len(X),
        "violations_I": {k: len(v) for k, v in viol_i.items()},
    }
    return fig, stats


# ─────────────────────────────────────────────
# P Chart (proportion defective)
# ─────────────────────────────────────────────

def p_chart(
    df: pd.DataFrame,
    defective_col: str,
    sample_size_col: str,
    subgroup_col: Optional[str] = None,
) -> Tuple[go.Figure, Dict]:
    """
    P chart for proportion of defective units.
    Handles variable subgroup sizes.
    """
    data = df.copy()
    if subgroup_col and subgroup_col in data.columns:
        grouped = data.groupby(subgroup_col).agg(
            n_defective=(defective_col, "sum"),
            n_total=(sample_size_col, "sum"),
        ).reset_index()
    else:
        grouped = data[[defective_col, sample_size_col]].copy()
        grouped.columns = ["n_defective", "n_total"]
        grouped = grouped.reset_index(drop=True)

    grouped["p"] = grouped["n_defective"] / grouped["n_total"]
    p_bar = grouped["n_defective"].sum() / grouped["n_total"].sum()

    grouped["ucl"] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / grouped["n_total"])
    grouped["lcl"] = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / grouped["n_total"]))

    p_vals = grouped["p"].values
    ucl_vals = grouped["ucl"].values
    lcl_vals = grouped["lcl"].values

    ooc_idx = [i for i in range(len(p_vals)) if p_vals[i] > ucl_vals[i] or p_vals[i] < lcl_vals[i]]

    fig = go.Figure()

    fig.add_hline(y=p_bar, line_color="#22c55e", line_dash="solid",
                  annotation_text=f"p̄={p_bar:.4f}", annotation_position="right")

    fig.add_trace(go.Scatter(y=ucl_vals, mode="lines", name="UCL",
                             line=dict(color="#ef4444", dash="dash")))
    fig.add_trace(go.Scatter(y=lcl_vals, mode="lines", name="LCL",
                             line=dict(color="#3b82f6", dash="dash")))
    fig.add_trace(go.Scatter(y=p_vals, mode="lines+markers", name="Proportion",
                             marker=dict(color="#a78bfa", size=6)))

    if ooc_idx:
        fig.add_trace(go.Scatter(x=ooc_idx, y=p_vals[ooc_idx], mode="markers",
                                 marker=dict(color="#ef4444", size=12, symbol="x"),
                                 name="OOC"))

    fig.update_layout(title="P Chart – Proportion Defective", template="plotly_dark",
                      yaxis_title="Proportion Defective", height=450, hovermode="x unified")

    stats = {
        "p_bar": p_bar, "n_subgroups": len(grouped),
        "n_ooc_points": len(ooc_idx), "ooc_indices": ooc_idx,
    }
    return fig, stats


# ─────────────────────────────────────────────
# C Chart (count of defects)
# ─────────────────────────────────────────────

def c_chart(
    df: pd.DataFrame,
    defect_count_col: str,
    subgroup_col: Optional[str] = None,
) -> Tuple[go.Figure, Dict]:
    """
    C chart for count of defects per inspection unit.
    Assumes constant area of opportunity.
    """
    data = df.copy()
    if subgroup_col and subgroup_col in data.columns:
        counts = data.groupby(subgroup_col)[defect_count_col].sum().reset_index()
        c_vals = counts[defect_count_col].values
    else:
        c_vals = data[defect_count_col].dropna().values

    c_bar = c_vals.mean()
    ucl = c_bar + 3 * np.sqrt(c_bar)
    lcl = max(0, c_bar - 3 * np.sqrt(c_bar))

    viol = _detect_violations(c_vals, ucl, c_bar, c_bar)
    ooc_idx = list(set(sum(viol.values(), [])))

    fig = go.Figure()
    fig.add_hline(y=ucl, line_color="#ef4444", line_dash="dash",
                  annotation_text=f"UCL={ucl:.4f}", annotation_position="right")
    fig.add_hline(y=c_bar, line_color="#22c55e", line_dash="solid",
                  annotation_text=f"c̄={c_bar:.4f}", annotation_position="right")
    fig.add_hline(y=lcl, line_color="#3b82f6", line_dash="dash",
                  annotation_text=f"LCL={lcl:.4f}", annotation_position="right")

    fig.add_trace(go.Scatter(y=c_vals, mode="lines+markers", name="Defect Count",
                             marker=dict(color="#a78bfa", size=6)))
    if ooc_idx:
        fig.add_trace(go.Scatter(x=ooc_idx, y=c_vals[ooc_idx], mode="markers",
                                 marker=dict(color="#ef4444", size=12, symbol="x"),
                                 name="OOC"))

    fig.update_layout(title="C Chart – Defect Count", template="plotly_dark",
                      yaxis_title="Defect Count", height=420, hovermode="x unified")

    stats = {
        "c_bar": c_bar, "UCL": ucl, "LCL": lcl,
        "n_points": len(c_vals), "violations": {k: len(v) for k, v in viol.items()},
    }
    return fig, stats


# ─────────────────────────────────────────────
# Chart Dispatcher
# ─────────────────────────────────────────────

def generate_control_chart(
    df: pd.DataFrame,
    chart_type: str,
    value_col: str,
    subgroup_col: Optional[str] = None,
    defective_col: Optional[str] = None,
    sample_size_col: Optional[str] = None,
    order_col: Optional[str] = None,
) -> Tuple[go.Figure, Dict]:
    """
    Unified dispatcher for all control chart types.
    chart_type: 'X-bar & R' | 'X-bar & S' | 'Individuals (I-MR)' | 'P Chart' | 'C Chart'
    """
    ct = chart_type.lower()

    if "x-bar" in ct and " r" in ct:
        if not subgroup_col:
            raise ValueError("X-bar & R chart requires a subgroup column.")
        return xbar_r_chart(df, value_col, subgroup_col)

    elif "x-bar" in ct and " s" in ct:
        if not subgroup_col:
            raise ValueError("X-bar & S chart requires a subgroup column.")
        return xbar_s_chart(df, value_col, subgroup_col)

    elif "imr" in ct or "individual" in ct:
        return imr_chart(df, value_col, order_col)

    elif "p chart" in ct or "p-chart" in ct:
        if not defective_col or not sample_size_col:
            raise ValueError("P chart requires defective_col and sample_size_col.")
        return p_chart(df, defective_col, sample_size_col, subgroup_col)

    elif "c chart" in ct or "c-chart" in ct:
        defect_col = defective_col or value_col
        return c_chart(df, defect_col, subgroup_col)

    else:
        raise ValueError(f"Unknown chart type: '{chart_type}'")
