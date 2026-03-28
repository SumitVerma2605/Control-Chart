"""
preprocessing.py
----------------
Robust data preprocessing pipeline for the DataFlow Analytics system.
Handles: schema detection, missing values, duplicates, outliers,
encoding, datetime parsing, normalization, and grouping.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Schema Detection
# ─────────────────────────────────────────────

def detect_schema(df: pd.DataFrame) -> Dict:
    """
    Automatically infer column types: numeric, categorical, datetime, boolean.
    Returns a schema dict with type per column and basic stats.
    """
    schema = {}
    for col in df.columns:
        series = df[col].dropna()
        dtype = str(df[col].dtype)
        n_unique = series.nunique()
        total = len(series)

        if pd.api.types.is_bool_dtype(df[col]):
            col_type = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
        else:
            # Try datetime parse on object columns
            parsed = _try_parse_datetime(series)
            if parsed is not None:
                col_type = "datetime"
            elif n_unique / max(total, 1) < 0.05 or n_unique <= 20:
                col_type = "categorical"
            else:
                col_type = "text"

        schema[col] = {
            "dtype": dtype,
            "inferred_type": col_type,
            "n_unique": int(n_unique),
            "n_missing": int(df[col].isna().sum()),
            "pct_missing": round(df[col].isna().mean() * 100, 2),
        }
        logger.debug(f"Column '{col}': dtype={dtype}, inferred={col_type}")

    logger.info(f"Schema detected for {len(schema)} columns.")
    return schema


def _try_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
    """Attempt datetime parse on a string series. Returns parsed series or None."""
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
               "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%Y%m%d"]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            return parsed
        except Exception:
            continue
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        if parsed.notna().sum() / max(len(series), 1) > 0.8:
            return parsed
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# Datetime Parsing
# ─────────────────────────────────────────────

def parse_datetime_columns(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    """Convert detected datetime columns to proper datetime64 dtype."""
    df = df.copy()
    for col, info in schema.items():
        if info["inferred_type"] == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.info(f"Parsed datetime column: '{col}'")
            except Exception as e:
                logger.warning(f"Failed to parse '{col}' as datetime: {e}")
    return df


# ─────────────────────────────────────────────
# Duplicate Removal
# ─────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows and return (cleaned_df, n_removed)."""
    original_len = len(df)
    df = df.drop_duplicates()
    n_removed = original_len - len(df)
    logger.info(f"Removed {n_removed} duplicate rows.")
    return df.reset_index(drop=True), n_removed


# ─────────────────────────────────────────────
# Missing Value Handling
# ─────────────────────────────────────────────

def handle_missing_values(
    df: pd.DataFrame,
    schema: Dict,
    strategy: str = "mean",
    custom_fill: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing values per column type.
    
    Strategies: 'mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill'
    custom_fill: {col_name: fill_value} to override per column.
    Returns (cleaned_df, report).
    """
    df = df.copy()
    report = {}

    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        col_type = schema.get(col, {}).get("inferred_type", "text")

        # Custom fill override
        if custom_fill and col in custom_fill:
            fill_val = custom_fill[col]
            df[col].fillna(fill_val, inplace=True)
            report[col] = {"n_missing": n_missing, "method": f"custom ({fill_val})"}
            continue

        if strategy == "drop":
            df.dropna(subset=[col], inplace=True)
            report[col] = {"n_missing": n_missing, "method": "drop_rows"}

        elif col_type == "numeric":
            if strategy == "mean":
                val = df[col].mean()
                method = f"mean ({val:.4f})"
            elif strategy == "median":
                val = df[col].median()
                method = f"median ({val:.4f})"
            elif strategy == "mode":
                val = df[col].mode()[0]
                method = f"mode ({val})"
            elif strategy in ("forward_fill", "ffill"):
                df[col].ffill(inplace=True)
                df[col].bfill(inplace=True)   # handle leading NaNs
                report[col] = {"n_missing": n_missing, "method": "forward_fill"}
                continue
            elif strategy in ("backward_fill", "bfill"):
                df[col].bfill(inplace=True)
                report[col] = {"n_missing": n_missing, "method": "backward_fill"}
                continue
            else:
                val = df[col].median()
                method = f"median fallback ({val:.4f})"
            df[col].fillna(val, inplace=True)
            report[col] = {"n_missing": n_missing, "method": method}

        elif col_type in ("categorical", "text", "boolean"):
            val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(val, inplace=True)
            report[col] = {"n_missing": n_missing, "method": f"mode ({val})"}

        elif col_type == "datetime":
            df[col].ffill(inplace=True)
            df[col].bfill(inplace=True)
            report[col] = {"n_missing": n_missing, "method": "forward_fill (datetime)"}

    logger.info(f"Missing values handled for {len(report)} columns using strategy='{strategy}'.")
    return df.reset_index(drop=True), report


# ─────────────────────────────────────────────
# Outlier Detection & Treatment
# ─────────────────────────────────────────────

def detect_outliers(
    df: pd.DataFrame,
    schema: Dict,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect outliers in numeric columns using IQR or Z-score.
    Returns (flag_df with boolean mask, outlier_report).
    """
    numeric_cols = [c for c, i in schema.items() if i["inferred_type"] == "numeric" and c in df.columns]
    outlier_flags = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    report = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if method == "iqr":
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
            mask = (df[col] < lower) | (df[col] > upper)
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            z_full = pd.Series(np.nan, index=df.index)
            z_full.loc[series.index] = z_scores
            mask = z_full > threshold
        else:
            mask = pd.Series(False, index=df.index)

        outlier_flags[col] = mask
        n_out = mask.sum()
        report[col] = {"n_outliers": int(n_out), "pct": round(n_out / len(df) * 100, 2)}
        logger.debug(f"'{col}': {n_out} outliers detected via {method}")

    return outlier_flags, report


def treat_outliers(
    df: pd.DataFrame,
    outlier_flags: pd.DataFrame,
    method: str = "clip",   # clip | remove | winsorize
) -> pd.DataFrame:
    """Treat detected outliers by clipping, removing, or winsorizing."""
    df = df.copy()
    for col in outlier_flags.columns:
        if col not in df.columns:
            continue
        mask = outlier_flags[col]
        if mask.sum() == 0:
            continue

        if method == "clip":
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        elif method == "remove":
            df = df[~mask]
        elif method == "winsorize":
            df[col] = stats.mstats.winsorize(df[col], limits=[0.05, 0.05])

    logger.info(f"Outlier treatment '{method}' applied.")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    schema: Dict,
    method: str = "onehot",
    max_cardinality: int = 20,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical columns.
    - onehot: get_dummies (low cardinality)
    - label: LabelEncoder (high cardinality or explicit choice)
    Returns (encoded_df, encoder_map).
    """
    df = df.copy()
    encoder_map = {}
    cat_cols = [
        c for c, i in schema.items()
        if i["inferred_type"] in ("categorical", "boolean") and c in df.columns
    ]

    for col in cat_cols:
        n_unique = df[col].nunique()
        if method == "onehot" and n_unique <= max_cardinality:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoder_map[col] = {"method": "onehot", "categories": list(dummies.columns)}
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoder_map[col] = {"method": "label", "classes": list(le.classes_)}

    logger.info(f"Encoded {len(encoder_map)} categorical columns.")
    return df, encoder_map


# ─────────────────────────────────────────────
# Normalization / Scaling
# ─────────────────────────────────────────────

def normalize_features(
    df: pd.DataFrame,
    schema: Dict,
    method: str = "standard",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Normalize numeric features.
    Methods: 'standard' (z-score), 'minmax', 'robust'.
    Returns (scaled_df, fitted_scaler).
    """
    df = df.copy()
    exclude = set(exclude_cols or [])
    num_cols = [
        c for c, i in schema.items()
        if i["inferred_type"] == "numeric" and c in df.columns and c not in exclude
    ]

    if not num_cols:
        return df, None

    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}
    scaler = scalers.get(method, StandardScaler())
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info(f"Scaled {len(num_cols)} numeric columns using '{method}' normalization.")
    return df, scaler


# ─────────────────────────────────────────────
# Time-Based Grouping / Aggregation
# ─────────────────────────────────────────────

def group_by_time(
    df: pd.DataFrame,
    datetime_col: str,
    freq: str = "D",           # H=hour, D=day, W=week, M=month
    agg_funcs: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Resample and aggregate a DataFrame by time frequency.
    agg_funcs: dict of {col: agg_func} e.g. {'value': 'mean'}
    """
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found.")
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)

    if agg_funcs is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        agg_funcs = {c: "mean" for c in numeric_cols}

    grouped = df.resample(freq).agg(agg_funcs).reset_index()
    logger.info(f"Time-grouped by '{freq}' on column '{datetime_col}'.")
    return grouped


# ─────────────────────────────────────────────
# Full Pipeline Runner
# ─────────────────────────────────────────────

def run_preprocessing_pipeline(
    df: pd.DataFrame,
    missing_strategy: str = "mean",
    outlier_method: str = "iqr",
    outlier_treatment: str = "clip",
    encode_method: str = "onehot",
    normalize_method: str = "standard",
    target_col: Optional[str] = None,
    remove_dups: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the complete preprocessing pipeline in order.
    Returns (processed_df, full_report).
    """
    report = {}

    # Step 1: Schema
    schema = detect_schema(df)
    report["schema"] = schema

    # Step 2: Datetime parsing
    df = parse_datetime_columns(df, schema)

    # Step 3: Duplicates
    if remove_dups:
        df, n_dup = remove_duplicates(df)
        report["duplicates_removed"] = n_dup

    # Step 4: Missing values
    df, mv_report = handle_missing_values(df, schema, strategy=missing_strategy)
    report["missing_values"] = mv_report

    # Step 5: Outlier detection + treatment
    outlier_flags, outlier_report = detect_outliers(df, schema, method=outlier_method)
    report["outliers"] = outlier_report
    df = treat_outliers(df, outlier_flags, method=outlier_treatment)

    # Step 6: Encoding (skip target column)
    exclude_encode = [target_col] if target_col else []
    schema_for_encode = {k: v for k, v in schema.items() if k not in exclude_encode}
    df, encoder_map = encode_categoricals(df, schema_for_encode, method=encode_method)
    report["encoding"] = encoder_map

    # Step 7: Normalize numeric (skip target)
    exclude_norm = [target_col] if target_col else []
    df, scaler = normalize_features(df, schema, method=normalize_method, exclude_cols=exclude_norm)
    report["normalization"] = {"method": normalize_method, "scaler": str(type(scaler).__name__)}

    logger.info("Full preprocessing pipeline complete.")
    return df, report
