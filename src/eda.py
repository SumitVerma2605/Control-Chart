"""
eda.py
------
Exploratory Data Analysis module for DataFlow Analytics.
Generates summary statistics, correlation matrices, distribution plots,
boxplots, time series trends, and missing value heatmaps.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
sns.set_theme(style="darkgrid")


# ─────────────────────────────────────────────
# Summary Statistics
# ─────────────────────────────────────────────

def summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive summary statistics for all column types.
    Returns dict with numeric, categorical, and datetime summaries.
    """
    result = {}

    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        desc = numeric_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        desc["skewness"] = numeric_df.skew()
        desc["kurtosis"] = numeric_df.kurtosis()
        desc["cv"] = (numeric_df.std() / numeric_df.mean().replace(0, np.nan)) * 100
        result["numeric"] = desc

    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    if not cat_df.empty:
        cat_stats = {}
        for col in cat_df.columns:
            vc = cat_df[col].value_counts()
            cat_stats[col] = {
                "n_unique": cat_df[col].nunique(),
                "top_value": vc.index[0] if len(vc) > 0 else None,
                "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "missing": int(cat_df[col].isna().sum()),
            }
        result["categorical"] = pd.DataFrame(cat_stats).T

    dt_cols = df.select_dtypes(include=["datetime64"]).columns
    if len(dt_cols) > 0:
        dt_stats = {}
        for col in dt_cols:
            s = df[col].dropna()
            dt_stats[col] = {
                "min": str(s.min()),
                "max": str(s.max()),
                "range_days": (s.max() - s.min()).days if len(s) > 0 else None,
                "missing": int(df[col].isna().sum()),
            }
        result["datetime"] = pd.DataFrame(dt_stats).T

    logger.info("Summary statistics computed.")
    return result


# ─────────────────────────────────────────────
# Correlation Matrix
# ─────────────────────────────────────────────

def correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    min_cols: int = 2,
) -> Optional[go.Figure]:
    """Generate an interactive Plotly correlation heatmap."""
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < min_cols:
        logger.warning("Not enough numeric columns for correlation matrix.")
        return None

    corr = num_df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.where(~mask)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Correlation Matrix ({method.capitalize()})",
        aspect="auto",
        width=800,
        height=600,
    )
    fig.update_layout(
        margin=dict(l=60, r=60, t=80, b=60),
        font=dict(size=11),
    )
    return fig


# ─────────────────────────────────────────────
# Distribution Plots
# ─────────────────────────────────────────────

def distribution_plots(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    bins: int = 30,
    max_cols: int = 4,
) -> go.Figure:
    """
    Create interactive histogram + KDE for numeric columns.
    Returns a Plotly figure with subplots.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if columns:
        num_cols = [c for c in columns if c in num_cols]
    num_cols = num_cols[:12]  # max 12 plots

    if not num_cols:
        return go.Figure()

    n = len(num_cols)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=num_cols)

    for i, col in enumerate(num_cols):
        r, c = divmod(i, cols)
        data = df[col].dropna()

        fig.add_trace(
            go.Histogram(x=data, nbinsx=bins, name=col,
                         opacity=0.7, marker_color="#636EFA",
                         showlegend=False),
            row=r + 1, col=c + 1,
        )

    fig.update_layout(
        title="Feature Distributions",
        height=300 * rows,
        width=400 * cols,
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def kde_plot(df: pd.DataFrame, column: str, group_by: Optional[str] = None) -> go.Figure:
    """Interactive KDE plot, optionally grouped by a categorical column."""
    if column not in df.columns:
        return go.Figure()

    if group_by and group_by in df.columns:
        fig = px.histogram(
            df, x=column, color=group_by,
            marginal="violin", opacity=0.7,
            histnorm="density", barmode="overlay",
            title=f"KDE: {column} by {group_by}",
            template="plotly_dark",
        )
    else:
        fig = px.histogram(
            df, x=column,
            marginal="violin", opacity=0.75,
            histnorm="density",
            title=f"Distribution: {column}",
            template="plotly_dark",
        )
    return fig


# ─────────────────────────────────────────────
# Boxplots
# ─────────────────────────────────────────────

def boxplots_by_category(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
) -> go.Figure:
    """Interactive boxplot of a numeric column grouped by a categorical column."""
    if value_col not in df.columns or category_col not in df.columns:
        return go.Figure()

    fig = px.box(
        df, x=category_col, y=value_col,
        color=category_col,
        notched=True,
        title=f"{value_col} by {category_col}",
        template="plotly_dark",
        points="outliers",
    )
    fig.update_layout(showlegend=False)
    return fig


def multi_boxplot(df: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
    """Boxplots for all (or selected) numeric columns on one chart."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if columns:
        num_cols = [c for c in columns if c in num_cols]
    num_cols = num_cols[:20]

    if not num_cols:
        return go.Figure()

    fig = go.Figure()
    for col in num_cols:
        fig.add_trace(go.Box(y=df[col].dropna(), name=col, boxmean=True))

    fig.update_layout(
        title="Feature Boxplots",
        template="plotly_dark",
        height=500,
    )
    return fig


# ─────────────────────────────────────────────
# Time Series Trends
# ─────────────────────────────────────────────

def time_series_plot(
    df: pd.DataFrame,
    datetime_col: str,
    value_cols: List[str],
    group_by: Optional[str] = None,
) -> go.Figure:
    """Interactive time series trend chart with optional grouping."""
    if datetime_col not in df.columns:
        return go.Figure()

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    if group_by and group_by in df.columns:
        fig = px.line(
            df, x=datetime_col, y=value_cols[0] if len(value_cols) == 1 else value_cols,
            color=group_by,
            title=f"Time Series: {', '.join(value_cols)} by {group_by}",
            template="plotly_dark",
        )
    else:
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, col in enumerate(value_cols):
            fig.add_trace(go.Scatter(
                x=df[datetime_col], y=df[col],
                mode="lines+markers",
                name=col,
                line=dict(color=colors[i % len(colors)]),
            ))
        fig.update_layout(
            title=f"Time Series: {', '.join(value_cols)}",
            xaxis_title=datetime_col,
            template="plotly_dark",
            height=450,
        )

    fig.update_layout(hovermode="x unified")
    return fig


# ─────────────────────────────────────────────
# Missing Value Heatmap
# ─────────────────────────────────────────────

def missing_value_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Interactive heatmap of missing values across rows and columns.
    White = present, dark = missing.
    """
    missing = df.isna().astype(int)
    pct_missing = missing.mean() * 100
    pct_missing = pct_missing[pct_missing > 0].sort_values(ascending=False)

    if pct_missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="✅ No missing values found!", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(size=18))
        fig.update_layout(template="plotly_dark")
        return fig

    # Bar chart of missing %
    fig = go.Figure(go.Bar(
        x=pct_missing.index,
        y=pct_missing.values,
        marker_color=[
            "#ef4444" if v > 50 else "#f97316" if v > 20 else "#eab308"
            for v in pct_missing.values
        ],
        text=[f"{v:.1f}%" for v in pct_missing.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Missing Values by Column (%)",
        xaxis_title="Column",
        yaxis_title="% Missing",
        template="plotly_dark",
        height=400,
    )
    return fig


def missing_heatmap_matrix(df: pd.DataFrame, max_rows: int = 300) -> go.Figure:
    """Matrix heatmap: rows vs columns colored by presence/absence."""
    sample = df.head(max_rows)
    missing_matrix = sample.isna().astype(int)

    fig = px.imshow(
        missing_matrix.T,
        color_continuous_scale=["#1e293b", "#ef4444"],
        title=f"Missing Value Matrix (first {max_rows} rows)",
        labels=dict(x="Row Index", y="Column", color="Missing"),
        aspect="auto",
        template="plotly_dark",
    )
    fig.update_coloraxes(showscale=False)
    return fig


# ─────────────────────────────────────────────
# Pair Plot
# ─────────────────────────────────────────────

def pair_plot(df: pd.DataFrame, columns: Optional[List[str]] = None, color_col: Optional[str] = None) -> go.Figure:
    """Interactive scatter matrix (pair plot)."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if columns:
        num_cols = [c for c in columns if c in num_cols]
    num_cols = num_cols[:6]

    if not num_cols:
        return go.Figure()

    dims = [dict(label=c, values=df[c]) for c in num_cols]
    color_vals = df[color_col] if color_col and color_col in df.columns else None

    fig = go.Figure(go.Splom(
        dimensions=dims,
        showupperhalf=False,
        diagonal_visible=True,
        marker=dict(
            color=color_vals,
            size=5,
            showscale=False,
            colorscale="Viridis",
        ),
    ))
    fig.update_layout(
        title="Scatter Matrix (Pair Plot)",
        template="plotly_dark",
        height=700,
        width=900,
    )
    return fig
