"""
app.py
------
DataFlow Analytics — Main Streamlit Application
A production-ready interactive data analysis dashboard supporting:
  - CSV / Excel / GitHub URL data ingestion
  - Automated preprocessing
  - EDA visualizations
  - Statistical Process Control (SPC) charts
  - ML model training and evaluation
"""

import io
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml

warnings.filterwarnings("ignore")

# ── Project path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import (
    detect_schema,
    handle_missing_values,
    remove_duplicates,
    detect_outliers,
    treat_outliers,
    encode_categoricals,
    normalize_features,
    parse_datetime_columns,
    run_preprocessing_pipeline,
)
from eda import (
    summary_statistics,
    correlation_matrix,
    distribution_plots,
    kde_plot,
    boxplots_by_category,
    multi_boxplot,
    time_series_plot,
    missing_value_heatmap,
    missing_heatmap_matrix,
    pair_plot,
)
from control_charts import generate_control_chart
from modeling import (
    detect_problem_type,
    prepare_features,
    train_model,
    plot_feature_importance,
    plot_residuals,
    plot_actual_vs_predicted,
    plot_roc_curve,
    plot_confusion_matrix,
)
from visualization import (
    metrics_table,
    cv_scores_chart,
    trend_decomposition_plot,
    fig_to_bytes,
    plotly_to_html,
)


# ── Config ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_config():
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}

CFG = load_config()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataFlow Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f172a; }
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1e293b; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #a78bfa; }

    /* Headers */
    h1 { color: #a78bfa !important; }
    h2 { color: #7c3aed !important; }
    h3 { color: #38bdf8 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #a78bfa; font-size: 1.8rem; }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] { color: #94a3b8; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #1e293b; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #a78bfa !important; border-bottom: 2px solid #a78bfa; }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; transition: all 0.2s;
    }
    .stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(124,58,237,0.4); }

    /* Info / success boxes */
    .stInfo { background: #1e3a5f; border-left: 4px solid #38bdf8; }
    .stSuccess { background: #14532d; border-left: 4px solid #22c55e; }
    .stWarning { background: #451a03; border-left: 4px solid #f59e0b; }
    .stError   { background: #450a0a; border-left: 4px solid #ef4444; }

    /* DataFrames */
    .stDataFrame { background: #1e293b; border-radius: 8px; }

    /* Section dividers */
    hr { border-color: #334155; }

    /* Expanders */
    .streamlit-expanderHeader { background: #1e293b !important; color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        "raw_df": None,
        "processed_df": None,
        "schema": None,
        "preprocessing_report": None,
        "model_results": None,
        "data_source": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def load_from_url(url: str) -> pd.DataFrame:
    """Fetch CSV/Excel from a raw URL or GitHub API."""
    # Convert GitHub blob URL to raw
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    headers = {"Accept": "application/vnd.github.v3.raw"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    if url.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(resp.content))
    return pd.read_csv(io.StringIO(resp.text))


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 📊 DataFlow Analytics")
    st.markdown("*Production-Grade Data Analysis*")
    st.divider()

    st.markdown("### 📥 Data Source")
    source_type = st.radio(
        "Choose input method",
        ["Upload File", "GitHub / URL"],
        horizontal=True,
    )

    if source_type == "Upload File":
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help="Supports CSV and Excel files up to 200 MB",
        )
        if uploaded is not None and st.session_state.raw_df is None:
            with st.spinner("Loading data…"):
                try:
                    fbytes = uploaded.read()
                    if uploaded.name.endswith(".csv"):
                        df = load_csv(fbytes, uploaded.name)
                    else:
                        df = load_excel(fbytes, uploaded.name)
                    st.session_state.raw_df = df
                    st.session_state.processed_df = df.copy()
                    st.session_state.schema = detect_schema(df)
                    st.session_state.data_source = uploaded.name
                    st.success(f"✅ Loaded: {uploaded.name}")
                    logger.info(f"Loaded file: {uploaded.name} ({df.shape})")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    else:
        url_input = st.text_input(
            "GitHub raw URL or direct CSV link",
            placeholder="https://raw.githubusercontent.com/...",
        )
        if st.button("🔗 Fetch Data") and url_input.strip():
            with st.spinner("Fetching from URL…"):
                try:
                    df = load_from_url(url_input.strip())
                    st.session_state.raw_df = df
                    st.session_state.processed_df = df.copy()
                    st.session_state.schema = detect_schema(df)
                    st.session_state.data_source = url_input
                    st.success(f"✅ Loaded {len(df):,} rows")
                    logger.info(f"Loaded URL: {url_input} ({df.shape})")
                except Exception as e:
                    st.error(f"Error fetching URL: {e}")

    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.divider()
        st.markdown("### 📋 Dataset Info")
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", f"{df.shape[1]}")
        missing_pct = df.isna().mean().mean() * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")

        st.divider()
        if st.button("🗑️ Clear Data"):
            for k in ["raw_df", "processed_df", "schema", "preprocessing_report", "model_results"]:
                st.session_state[k] = None
            st.rerun()

    # Sample datasets
    st.divider()
    st.markdown("### 🗂️ Sample Datasets")
    sample_datasets = {
        "Iris (Classification)":      "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "Titanic (Classification)":   "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "Car MPG (Regression)":        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
        "Stock Prices (Time Series)":  "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv",
    }
    chosen = st.selectbox("Load a sample", ["-- choose --"] + list(sample_datasets.keys()))
    if chosen != "-- choose --" and st.button("Load Sample"):
        with st.spinner(f"Loading {chosen}…"):
            try:
                df = load_from_url(sample_datasets[chosen])
                st.session_state.raw_df = df
                st.session_state.processed_df = df.copy()
                st.session_state.schema = detect_schema(df)
                st.session_state.data_source = chosen
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")


# ─────────────────────────────────────────────
# Main Panel
# ─────────────────────────────────────────────

if st.session_state.raw_df is None:
    # Landing page
    st.markdown("# 📊 DataFlow Analytics")
    st.markdown("### Production-Ready Data Analysis & SPC Platform")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### 📥 Ingest")
        st.markdown("CSV · Excel · GitHub URLs · Direct links")
    with col2:
        st.markdown("#### 🧹 Preprocess")
        st.markdown("Auto-impute · Outlier handling · Encoding · Scaling")
    with col3:
        st.markdown("#### 📈 SPC Charts")
        st.markdown("X-bar/R · I-MR · P · C charts with control limits")
    with col4:
        st.markdown("#### 🤖 ML Models")
        st.markdown("Regression · Classification · CV · Feature importance")

    st.divider()
    st.info("👈 **Upload a file or load a sample dataset from the sidebar to get started.**")

    st.markdown("""
    #### Quick Start
    1. Upload a CSV/Excel file **or** paste a GitHub raw URL in the sidebar
    2. Use sample datasets to explore immediately
    3. Navigate the tabs: **Preview → Preprocess → EDA → Control Charts → ML Model**
    """)
    st.stop()


# ─── Active Dashboard ─────────────────────────────────────────────────────────
df_raw = st.session_state.raw_df
df_proc = st.session_state.processed_df
schema = st.session_state.schema

st.markdown(f"# 📊 DataFlow Analytics")
st.caption(f"Source: `{st.session_state.data_source}` &nbsp;|&nbsp; {len(df_raw):,} rows × {df_raw.shape[1]} columns")

tabs = st.tabs(["👁️ Preview", "🧹 Preprocess", "📊 EDA", "📈 Control Charts", "🤖 ML Model"])


# ═══════════════════════════════════════════════════════════
# TAB 1: DATA PREVIEW
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 👁️ Data Preview")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Rows",    f"{len(df_raw):,}")
    with col2: st.metric("Columns", f"{df_raw.shape[1]}")
    with col3: st.metric("Missing", f"{df_raw.isna().sum().sum():,}")
    with col4: st.metric("Duplicates", f"{df_raw.duplicated().sum():,}")

    st.divider()
    n_preview = st.slider("Rows to preview", 5, 100, 20)
    st.dataframe(df_raw.head(n_preview), use_container_width=True)

    st.divider()
    st.markdown("### 🔍 Detected Schema")
    if schema:
        schema_df = pd.DataFrame(schema).T
        st.dataframe(schema_df, use_container_width=True)

    st.divider()
    st.markdown("### 📥 Download Raw Data")
    csv_raw = df_raw.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv_raw, "raw_data.csv", "text/csv")


# ═══════════════════════════════════════════════════════════
# TAB 2: PREPROCESSING
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🧹 Data Preprocessing")

    with st.form("preprocess_form"):
        st.markdown("### ⚙️ Configure Pipeline")
        col1, col2 = st.columns(2)

        with col1:
            remove_dups   = st.checkbox("Remove duplicates", value=True)
            mv_strategy   = st.selectbox("Missing value strategy",
                                         ["mean", "median", "mode", "drop", "forward_fill"])
            outlier_method = st.selectbox("Outlier detection method", ["iqr", "zscore"])
            outlier_thresh = st.slider("Outlier threshold (IQR mult / Z-score)",
                                       0.5, 5.0, 1.5, 0.25)
            outlier_treat  = st.selectbox("Outlier treatment", ["clip", "remove", "winsorize"])

        with col2:
            encode_method = st.selectbox("Categorical encoding", ["onehot", "label"])
            norm_method   = st.selectbox("Numeric normalization", ["standard", "minmax", "robust", "none"])
            target_col    = st.selectbox("Target column (skip normalization/encoding)",
                                         ["-- none --"] + list(df_raw.columns))
            parse_dt      = st.checkbox("Auto-parse datetime columns", value=True)

        run_btn = st.form_submit_button("🚀 Run Preprocessing Pipeline", use_container_width=True)

    if run_btn:
        target = None if target_col == "-- none --" else target_col
        with st.spinner("Running preprocessing pipeline…"):
            try:
                proc_df, report = run_preprocessing_pipeline(
                    df_raw.copy(),
                    missing_strategy=mv_strategy,
                    outlier_method=outlier_method,
                    outlier_treatment=outlier_treat,
                    encode_method=encode_method,
                    normalize_method=norm_method if norm_method != "none" else "standard",
                    target_col=target,
                    remove_dups=remove_dups,
                )
                if norm_method == "none":
                    # Re-run without normalization
                    from preprocessing import run_preprocessing_pipeline as rpp
                    proc_df2, _ = rpp(df_raw.copy(), missing_strategy=mv_strategy,
                                      outlier_method=outlier_method, outlier_treatment=outlier_treat,
                                      encode_method=encode_method, normalize_method="standard",
                                      target_col=target, remove_dups=remove_dups)
                    # Use raw numerics
                    num_cols_raw = df_raw.select_dtypes(include=np.number).columns
                    for c in num_cols_raw:
                        if c in proc_df2.columns and c in df_raw.columns:
                            proc_df2[c] = df_raw[c].values[:len(proc_df2)]
                    proc_df = proc_df2

                st.session_state.processed_df = proc_df
                st.session_state.preprocessing_report = report
                st.success(f"✅ Pipeline complete! {proc_df.shape[0]:,} rows × {proc_df.shape[1]} columns")
            except Exception as e:
                st.error(f"Preprocessing error: {e}")
                logger.exception("Preprocessing failed")

    if st.session_state.preprocessing_report:
        report = st.session_state.preprocessing_report
        proc_df = st.session_state.processed_df

        st.divider()
        st.markdown("### 📋 Preprocessing Report")

        col1, col2, col3 = st.columns(3)
        with col1:
            dups = report.get("duplicates_removed", 0)
            st.metric("Duplicates Removed", dups)
        with col2:
            mv_cols = len(report.get("missing_values", {}))
            st.metric("Columns Imputed", mv_cols)
        with col3:
            enc_cols = len(report.get("encoding", {}))
            st.metric("Columns Encoded", enc_cols)

        with st.expander("🔎 Missing Value Imputation Details"):
            mv = report.get("missing_values", {})
            if mv:
                st.dataframe(pd.DataFrame(mv).T, use_container_width=True)
            else:
                st.info("No missing values found.")

        with st.expander("🔎 Outlier Detection Report"):
            out = report.get("outliers", {})
            if out:
                out_df = pd.DataFrame(out).T
                out_df = out_df[out_df["n_outliers"] > 0]
                st.dataframe(out_df, use_container_width=True)
            else:
                st.info("No outliers detected.")

        with st.expander("🔎 Encoding Details"):
            enc = report.get("encoding", {})
            if enc:
                for col, info in enc.items():
                    st.write(f"**{col}**: {info['method']}")
            else:
                st.info("No categorical encoding applied.")

        st.divider()
        st.markdown("### 👁️ Processed Data Preview")
        st.dataframe(proc_df.head(20), use_container_width=True)

        csv_proc = proc_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Processed CSV", csv_proc,
                           "processed_data.csv", "text/csv")


# ═══════════════════════════════════════════════════════════
# TAB 3: EDA
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 📊 Exploratory Data Analysis")
    df_eda = st.session_state.processed_df

    eda_tabs = st.tabs([
        "📈 Summary", "🔥 Correlation", "📉 Distributions",
        "📦 Boxplots", "⏱️ Time Series", "❓ Missing Values", "🔵 Pair Plot"
    ])

    # Summary
    with eda_tabs[0]:
        st.markdown("### Summary Statistics")
        stats = summary_statistics(df_eda)
        if "numeric" in stats:
            st.markdown("#### Numeric Features")
            st.dataframe(stats["numeric"].round(4), use_container_width=True)
        if "categorical" in stats:
            st.markdown("#### Categorical Features")
            st.dataframe(stats["categorical"], use_container_width=True)
        if "datetime" in stats:
            st.markdown("#### Datetime Columns")
            st.dataframe(stats["datetime"], use_container_width=True)

    # Correlation
    with eda_tabs[1]:
        st.markdown("### Correlation Matrix")
        method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
        fig_corr = correlation_matrix(df_eda, method=method)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
            html_corr = plotly_to_html(fig_corr)
            st.download_button("⬇️ Download HTML", html_corr, "correlation.html", "text/html")
        else:
            st.warning("Not enough numeric columns for correlation matrix.")

    # Distributions
    with eda_tabs[2]:
        st.markdown("### Feature Distributions")
        num_cols = df_eda.select_dtypes(include=np.number).columns.tolist()
        sel_cols = st.multiselect("Select columns (max 12)", num_cols, default=num_cols[:min(6, len(num_cols))])
        bins = st.slider("Bins", 10, 100, 30)
        if sel_cols:
            fig_dist = distribution_plots(df_eda, columns=sel_cols, bins=bins)
            st.plotly_chart(fig_dist, use_container_width=True)

            st.divider()
            st.markdown("#### KDE Plot")
            kde_col = st.selectbox("Feature", sel_cols, key="kde_col")
            cat_cols = df_eda.select_dtypes(include=["object", "category"]).columns.tolist()
            grp_col  = st.selectbox("Group by (optional)", ["-- none --"] + cat_cols, key="kde_grp")
            if kde_col:
                grp = None if grp_col == "-- none --" else grp_col
                st.plotly_chart(kde_plot(df_eda, kde_col, grp), use_container_width=True)

    # Boxplots
    with eda_tabs[3]:
        st.markdown("### Boxplots")
        num_cols2  = df_eda.select_dtypes(include=np.number).columns.tolist()
        cat_cols2  = df_eda.select_dtypes(include=["object", "category"]).columns.tolist()

        box_type = st.radio("Chart type", ["By Category", "All Features"], horizontal=True)
        if box_type == "By Category" and num_cols2 and cat_cols2:
            val  = st.selectbox("Value column", num_cols2)
            cat  = st.selectbox("Category column", cat_cols2)
            st.plotly_chart(boxplots_by_category(df_eda, val, cat), use_container_width=True)
        else:
            sel_box = st.multiselect("Select columns", num_cols2,
                                     default=num_cols2[:min(8, len(num_cols2))])
            if sel_box:
                st.plotly_chart(multi_boxplot(df_eda, sel_box), use_container_width=True)

    # Time Series
    with eda_tabs[4]:
        st.markdown("### Time Series Trends")
        dt_cols = [c for c, info in (schema or {}).items() if info.get("inferred_type") == "datetime" and c in df_eda.columns]
        num_ts  = df_eda.select_dtypes(include=np.number).columns.tolist()

        if not dt_cols:
            st.info("No datetime columns detected. Please ensure dates are parsed in the Preprocessing tab.")
        else:
            dt_sel  = st.selectbox("Datetime column", dt_cols)
            val_sel = st.multiselect("Value columns", num_ts,
                                     default=[num_ts[0]] if num_ts else [])
            if dt_sel and val_sel:
                st.plotly_chart(time_series_plot(df_eda, dt_sel, val_sel), use_container_width=True)

                st.markdown("#### Trend Decomposition")
                if len(val_sel) == 1:
                    st.plotly_chart(trend_decomposition_plot(df_eda, dt_sel, val_sel[0]),
                                    use_container_width=True)

    # Missing Values
    with eda_tabs[5]:
        st.markdown("### Missing Value Analysis")
        st.plotly_chart(missing_value_heatmap(df_raw), use_container_width=True)
        st.plotly_chart(missing_heatmap_matrix(df_raw), use_container_width=True)

    # Pair Plot
    with eda_tabs[6]:
        st.markdown("### Scatter Matrix (Pair Plot)")
        num_pair = df_eda.select_dtypes(include=np.number).columns.tolist()
        cat_pair = df_eda.select_dtypes(include=["object", "category"]).columns.tolist()
        sel_pair = st.multiselect("Select features (max 6)", num_pair,
                                  default=num_pair[:min(4, len(num_pair))])
        color_p  = st.selectbox("Color by", ["-- none --"] + cat_pair, key="pair_color")
        if sel_pair:
            cp = None if color_p == "-- none --" else color_p
            st.plotly_chart(pair_plot(df_eda, sel_pair, cp), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 4: CONTROL CHARTS
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 📈 Statistical Process Control Charts")
    df_spc = st.session_state.processed_df

    CHART_TYPES = ["X-bar & R", "X-bar & S", "Individuals (I-MR)", "P Chart", "C Chart"]
    num_spc = df_spc.select_dtypes(include=np.number).columns.tolist()
    all_spc = df_spc.columns.tolist()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Chart Configuration")
        chart_type = st.selectbox("Chart Type", CHART_TYPES)

        value_col = st.selectbox("Value / Measurement Column", num_spc) if num_spc else None

        subgroup_col, defective_col, sample_size_col, order_col = None, None, None, None

        if chart_type in ["X-bar & R", "X-bar & S"]:
            subgroup_col = st.selectbox("Subgroup / Sample ID Column", all_spc)
            st.info("Groups data by this column to compute subgroup means and ranges.")

        elif chart_type == "Individuals (I-MR)":
            order_col = st.selectbox("Ordering Column (optional)", ["-- none --"] + all_spc)
            if order_col == "-- none --":
                order_col = None

        elif chart_type == "P Chart":
            defective_col  = st.selectbox("# Defective Column", num_spc)
            sample_size_col = st.selectbox("Sample Size Column", num_spc)
            subgroup_col   = st.selectbox("Subgroup Column (optional)", ["-- none --"] + all_spc)
            if subgroup_col == "-- none --":
                subgroup_col = None

        elif chart_type == "C Chart":
            defective_col = st.selectbox("Defect Count Column", num_spc)
            subgroup_col  = st.selectbox("Subgroup Column (optional)", ["-- none --"] + all_spc)
            if subgroup_col == "-- none --":
                subgroup_col = None

        generate_btn = st.button("📈 Generate Control Chart", use_container_width=True)

    with col2:
        if generate_btn:
            with st.spinner("Generating control chart…"):
                try:
                    fig_spc, stats_spc = generate_control_chart(
                        df=df_spc,
                        chart_type=chart_type,
                        value_col=value_col or (defective_col or num_spc[0] if num_spc else ""),
                        subgroup_col=subgroup_col,
                        defective_col=defective_col,
                        sample_size_col=sample_size_col,
                        order_col=order_col,
                    )
                    st.plotly_chart(fig_spc, use_container_width=True)

                    st.markdown("#### 📊 Chart Statistics")
                    stat_items = {k: v for k, v in stats_spc.items()
                                  if not isinstance(v, list) and v is not None}
                    stat_df = pd.DataFrame(list(stat_items.items()), columns=["Statistic", "Value"])
                    st.dataframe(stat_df, use_container_width=True)

                    # Violations summary
                    viol_keys = [k for k in stats_spc if k.startswith("violations")]
                    for vk in viol_keys:
                        viol = stats_spc[vk]
                        if any(v > 0 for v in viol.values()):
                            st.warning(f"⚠️ **{vk.replace('_', ' ').title()}**: " +
                                       "; ".join(f"{r}: {n} pts" for r, n in viol.items() if n > 0))
                        else:
                            st.success(f"✅ **{vk.replace('_', ' ').title()}**: No violations detected.")

                    html_spc = plotly_to_html(fig_spc)
                    st.download_button("⬇️ Download Chart HTML", html_spc,
                                       f"spc_{chart_type.replace(' ', '_')}.html", "text/html")

                except Exception as e:
                    st.error(f"Chart generation failed: {e}")
                    logger.exception("Control chart error")

        else:
            st.markdown("#### Control Chart Guide")
            guide = {
                "X-bar & R":       "Variable data with subgroups; monitors process mean and range.",
                "X-bar & S":       "Like X-bar & R but uses std dev instead of range; better for n > 10.",
                "Individuals (I-MR)": "One measurement per time point; most common in practice.",
                "P Chart":         "Proportion defective; handles variable sample sizes.",
                "C Chart":         "Count of defects per unit; assumes constant area of opportunity.",
            }
            for ct, desc in guide.items():
                emoji = "✅" if ct == chart_type else "  "
                st.markdown(f"{emoji} **{ct}**: {desc}")


# ═══════════════════════════════════════════════════════════
# TAB 5: ML MODEL
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🤖 Machine Learning Model Training")
    df_ml = st.session_state.processed_df

    num_ml = df_ml.select_dtypes(include=np.number).columns.tolist()
    all_ml = df_ml.columns.tolist()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Model Configuration")
        target_ml = st.selectbox("Target Column", all_ml, key="ml_target")
        feature_ml = st.multiselect("Feature Columns (leave empty = all numeric)",
                                     [c for c in num_ml if c != target_ml])

        # Detect problem type
        if target_ml and target_ml in df_ml.columns:
            prob_type = detect_problem_type(df_ml[target_ml])
            st.info(f"🔍 Detected: **{prob_type.title()}**")
        else:
            prob_type = "regression"

        MODELS = {
            "regression":     ["Linear Regression", "Random Forest Regressor"],
            "classification": ["Logistic Regression", "Random Forest Classifier"],
        }
        try:
            from xgboost import XGBClassifier
            MODELS["classification"].append("XGBoost")
        except ImportError:
            pass
        try:
            from xgboost import XGBRegressor
            MODELS["regression"].append("XGBoost Regressor")
        except ImportError:
            pass

        model_name = st.selectbox("Model", MODELS.get(prob_type, MODELS["regression"]))
        test_size  = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        cv_folds   = st.slider("CV Folds", 2, 10, 5)

        train_btn  = st.button("🚀 Train Model", use_container_width=True)

    with col2:
        if train_btn and target_ml:
            with st.spinner(f"Training {model_name}…"):
                try:
                    feats = feature_ml if feature_ml else None
                    X, y = prepare_features(df_ml, target_ml, feats)

                    results = train_model(
                        X, y,
                        model_name=model_name,
                        problem_type=prob_type,
                        test_size=test_size,
                        cv_folds=cv_folds,
                    )
                    st.session_state.model_results = results
                    st.success(f"✅ Training complete!")

                except Exception as e:
                    st.error(f"Training failed: {e}")
                    logger.exception("Model training error")

        results = st.session_state.model_results
        if results:
            st.markdown("### 📊 Model Results")
            st.plotly_chart(metrics_table(results["metrics"]), use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("CV Mean Score",  f"{abs(results['cv_mean']):.4f}")
            with col_b:
                st.metric("CV Std Dev",     f"{results['cv_std']:.4f}")

            st.plotly_chart(cv_scores_chart(results["cv_scores"], results["model_name"]),
                            use_container_width=True)

            model_tabs = st.tabs(["📊 Feature Importance", "📉 Diagnostics", "📈 Performance"])

            with model_tabs[0]:
                top_n = st.slider("Top N features", 5, 30, 15)
                fig_fi = plot_feature_importance(results, top_n=top_n)
                if fig_fi.data:
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model.")

            with model_tabs[1]:
                if results["problem_type"] == "regression":
                    st.plotly_chart(plot_residuals(results), use_container_width=True)
                else:
                    st.plotly_chart(plot_confusion_matrix(results), use_container_width=True)

            with model_tabs[2]:
                if results["problem_type"] == "regression":
                    st.plotly_chart(plot_actual_vs_predicted(results), use_container_width=True)
                else:
                    fig_roc = plot_roc_curve(results)
                    if fig_roc.data:
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info("ROC curve available for binary classification only.")

                    if "classification_report" in results:
                        with st.expander("Classification Report"):
                            cr_df = pd.DataFrame(results["classification_report"]).T
                            st.dataframe(cr_df.round(4), use_container_width=True)

            # Download
            st.divider()
            pred_df = results["X_test"].copy()
            pred_df["y_actual"]    = results["y_test"].values
            pred_df["y_predicted"] = results["y_pred"]
            csv_pred = pred_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", csv_pred,
                               "predictions.csv", "text/csv")

        elif not train_btn:
            st.info("👈 Configure and click **Train Model** to begin.")

            st.markdown("""
            #### Available Models
            **Regression:** Linear Regression, Random Forest Regressor, XGBoost Regressor  
            **Classification:** Logistic Regression, Random Forest Classifier, XGBoost  

            #### Metrics
            | Problem | Metrics |
            |---------|---------|
            | Regression | RMSE, MAE, R² |
            | Classification | Accuracy, F1, ROC-AUC |
            """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "DataFlow Analytics v1.0.0 · Built with Streamlit · "
    "Powered by scikit-learn, Plotly & Pandas"
    "</div>",
    unsafe_allow_html=True,
)
