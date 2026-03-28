"""
visualization.py
----------------
Shared visualization utilities for DataFlow Analytics.
Provides consistent chart styling, export helpers, and composite layouts.
"""

import io
import logging
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
pio.templates.default = "plotly_dark"

PALETTE = {
    "primary":   "#a78bfa",
    "secondary": "#34d399",
    "danger":    "#ef4444",
    "warning":   "#f59e0b",
    "info":      "#38bdf8",
    "neutral":   "#94a3b8",
}


# ─────────────────────────────────────────────
# Style Helpers
# ─────────────────────────────────────────────

def apply_dark_style():
    """Apply dark theme to matplotlib figures."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor":  "#1e293b",
        "figure.facecolor": "#0f172a",
        "grid.color":      "#334155",
        "text.color":      "#e2e8f0",
        "axes.labelcolor": "#e2e8f0",
        "xtick.color":     "#94a3b8",
        "ytick.color":     "#94a3b8",
    })


# ─────────────────────────────────────────────
# Matplotlib Figures (for download)
# ─────────────────────────────────────────────

def plot_seaborn_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Seaborn correlation heatmap as a matplotlib figure (for PNG export)."""
    apply_dark_style()
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.7)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation Matrix", fontsize=14, pad=12)
    fig.tight_layout()
    return fig


def plot_seaborn_pairplot(df: pd.DataFrame, hue: Optional[str] = None, max_cols: int = 5) -> plt.Figure:
    """Seaborn pair plot for quick EDA."""
    apply_dark_style()
    num_cols = df.select_dtypes(include=np.number).columns[:max_cols].tolist()
    data = df[num_cols + ([hue] if hue and hue in df.columns else [])].dropna()
    g = sns.pairplot(data, hue=hue, diag_kind="kde", plot_kws={"alpha": 0.5})
    g.fig.patch.set_facecolor("#0f172a")
    return g.fig


def plot_distribution_grid(df: pd.DataFrame, cols: Optional[List[str]] = None, bins: int = 30) -> plt.Figure:
    """Multi-panel histogram grid using matplotlib."""
    apply_dark_style()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if cols:
        num_cols = [c for c in cols if c in num_cols]
    num_cols = num_cols[:16]
    n = len(num_cols)
    if n == 0:
        return plt.figure()

    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=bins, color=PALETTE["primary"], alpha=0.75, edgecolor="none")
        try:
            import scipy.stats as spst
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde = spst.gaussian_kde(data)
            ax2 = ax.twinx()
            ax2.plot(kde_x, kde(kde_x), color=PALETTE["secondary"], lw=2)
            ax2.set_yticks([])
        except Exception:
            pass
        ax.set_title(col, fontsize=10, color="#e2e8f0")
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, color="#e2e8f0", y=1.02)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Plotly Helpers
# ─────────────────────────────────────────────

def metrics_table(metrics: Dict) -> go.Figure:
    """Render model metrics as a Plotly table."""
    keys = list(metrics.keys())
    vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()]

    fig = go.Figure(go.Table(
        header=dict(
            values=["Metric", "Value"],
            fill_color="#1e293b",
            font=dict(color="#e2e8f0", size=13),
            align="left",
        ),
        cells=dict(
            values=[keys, vals],
            fill_color=[["#0f172a", "#1e293b"] * (len(keys) // 2 + 1)],
            font=dict(color=["#94a3b8", "#a78bfa"], size=12),
            align="left",
        ),
    ))
    fig.update_layout(
        title="Model Performance Metrics",
        template="plotly_dark",
        height=200 + len(keys) * 30,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def cv_scores_chart(cv_scores: np.ndarray, model_name: str) -> go.Figure:
    """Bar chart of cross-validation fold scores."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(cv_scores))],
        y=np.abs(cv_scores),
        marker_color=PALETTE["primary"],
        text=[f"{abs(s):.4f}" for s in cv_scores],
        textposition="outside",
    ))
    fig.add_hline(y=np.abs(cv_scores).mean(), line_dash="dash",
                  line_color=PALETTE["secondary"],
                  annotation_text=f"Mean={np.abs(cv_scores).mean():.4f}")
    fig.update_layout(
        title=f"Cross-Validation Scores – {model_name}",
        yaxis_title="Score",
        template="plotly_dark",
        height=380,
    )
    return fig


def trend_decomposition_plot(df: pd.DataFrame, datetime_col: str, value_col: str) -> go.Figure:
    """Simple rolling trend/seasonality decomposition chart."""
    try:
        data = df[[datetime_col, value_col]].dropna().copy()
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        data = data.sort_values(datetime_col).set_index(datetime_col)
        series = data[value_col]
        window = max(7, len(series) // 20)

        trend = series.rolling(window, center=True).mean()
        residual = series - trend

        fig = make_subplots(rows=3, cols=1, subplot_titles=["Original", "Trend", "Residual"],
                            vertical_spacing=0.1)
        for i, (s, name, color) in enumerate([
            (series, "Original", PALETTE["primary"]),
            (trend,  "Trend",    PALETTE["secondary"]),
            (residual, "Residual", PALETTE["warning"]),
        ]):
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                     name=name, line=dict(color=color)), row=i + 1, col=1)

        fig.update_layout(title=f"Trend Decomposition: {value_col}",
                          template="plotly_dark", height=700, showlegend=False)
        return fig

    except Exception as e:
        logger.warning(f"Trend decomposition failed: {e}")
        return go.Figure()


# ─────────────────────────────────────────────
# Export Helpers
# ─────────────────────────────────────────────

def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 150) -> bytes:
    """Convert matplotlib figure to PNG/PDF bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def plotly_to_html(fig: go.Figure) -> str:
    """Convert Plotly figure to standalone HTML string."""
    return pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
