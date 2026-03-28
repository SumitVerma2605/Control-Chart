"""
modeling.py
-----------
Machine learning model training, evaluation, and prediction for DataFlow Analytics.
Supports regression and classification with cross-validation, feature importance,
and comprehensive metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Problem Type Detection
# ─────────────────────────────────────────────

def detect_problem_type(y: pd.Series, threshold: int = 10) -> str:
    """Infer whether the target is for regression or classification."""
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > threshold:
        return "regression"
    return "classification"


# ─────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into X (features) and y (target).
    Optionally select specific feature columns.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if feature_cols:
        X = df[feature_cols].copy()
    else:
        X = df.drop(columns=[target_col]).copy()

    y = df[target_col].copy()

    # Keep only numeric features (encoding must happen upstream)
    X = X.select_dtypes(include=np.number)

    if drop_na:
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

    logger.info(f"Features prepared: {X.shape[1]} columns, {len(X)} samples.")
    return X, y


# ─────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────

def get_model(model_name: str, problem_type: str, random_state: int = 42):
    """Return a fitted-ready sklearn estimator by name."""
    models = {
        "regression": {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=random_state),
        },
        "classification": {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=random_state),
        },
    }
    if XGBOOST_AVAILABLE:
        models["regression"]["XGBoost Regressor"] = XGBRegressor(random_state=random_state, verbosity=0)
        models["classification"]["XGBoost"] = XGBClassifier(random_state=random_state, verbosity=0, eval_metric="logloss")

    if problem_type not in models:
        raise ValueError(f"Unknown problem type: {problem_type}")
    if model_name not in models[problem_type]:
        raise ValueError(f"Model '{model_name}' not available for {problem_type}.")

    return models[problem_type][model_name]


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    problem_type: Optional[str] = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    scale_features: bool = True,
) -> Dict:
    """
    Train a model with train/test split and cross-validation.
    Returns a comprehensive results dict.
    """
    if problem_type is None:
        problem_type = detect_problem_type(y)

    logger.info(f"Training '{model_name}' ({problem_type}) on {X.shape[1]} features, {len(X)} samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if problem_type == "classification" else None
    )

    model = get_model(model_name, problem_type, random_state)

    if scale_features and problem_type == "regression":
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    else:
        pipe = Pipeline([("model", model)])

    # Cross-validation
    cv_metric = "neg_mean_squared_error" if problem_type == "regression" else "f1_weighted"
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=cv_metric, n_jobs=-1)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    results = {
        "model_name": model_name,
        "problem_type": problem_type,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "pipeline": pipe,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": list(X.columns),
    }

    # Metrics
    if problem_type == "regression":
        results["metrics"] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred),
        }
    else:
        results["metrics"] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        try:
            y_prob = pipe.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                results["metrics"]["ROC-AUC"] = roc_auc_score(y_test, y_prob[:, 1])
                results["roc_proba"] = y_prob[:, 1]
            else:
                results["metrics"]["ROC-AUC (OvR)"] = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="weighted"
                )
        except Exception:
            pass
        results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        results["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance
    final_model = pipe.named_steps["model"]
    if hasattr(final_model, "feature_importances_"):
        results["feature_importance"] = pd.Series(
            final_model.feature_importances_, index=list(X.columns)
        ).sort_values(ascending=False)
    elif hasattr(final_model, "coef_"):
        coef = final_model.coef_.flatten() if final_model.coef_.ndim > 1 else final_model.coef_
        results["feature_importance"] = pd.Series(
            np.abs(coef), index=list(X.columns)
        ).sort_values(ascending=False)

    logger.info(f"Training complete. Metrics: {results['metrics']}")
    return results


# ─────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────

def plot_feature_importance(results: Dict, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    if "feature_importance" not in results:
        return go.Figure()

    fi = results["feature_importance"].head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker_color="#a78bfa",
        text=[f"{v:.4f}" for v in fi.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances – {results['model_name']}",
        xaxis_title="Importance",
        template="plotly_dark",
        height=max(400, top_n * 25),
    )
    return fig


def plot_residuals(results: Dict) -> go.Figure:
    """Residual plot for regression models."""
    if results.get("problem_type") != "regression":
        return go.Figure()

    y_test = results["y_test"]
    y_pred = results["y_pred"]
    residuals = y_test.values - y_pred

    fig = make_residual_figure(y_pred, residuals)
    return fig


def make_residual_figure(y_pred, residuals) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuals vs Fitted", "Residual Distribution"])

    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                             marker=dict(color="#a78bfa", size=5, opacity=0.6),
                             name="Residuals"), row=1, col=1)
    fig.add_hline(y=0, line_color="#ef4444", line_dash="dash", row=1, col=1)

    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name="Distribution",
                               marker_color="#34d399", opacity=0.7), row=1, col=2)

    fig.update_layout(title="Residual Analysis", template="plotly_dark",
                      height=400, showlegend=False)
    return fig


def plot_actual_vs_predicted(results: Dict) -> go.Figure:
    """Scatter of actual vs predicted for regression."""
    if results.get("problem_type") != "regression":
        return go.Figure()

    y_test = results["y_test"].values
    y_pred = results["y_pred"]
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                             marker=dict(color="#a78bfa", size=5, opacity=0.6),
                             name="Predictions"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="#ef4444", dash="dash"),
                             name="Perfect fit"))
    r2 = results["metrics"].get("R²", 0)
    fig.update_layout(title=f"Actual vs Predicted (R²={r2:.4f})",
                      xaxis_title="Actual", yaxis_title="Predicted",
                      template="plotly_dark", height=450)
    return fig


def plot_roc_curve(results: Dict) -> go.Figure:
    """ROC curve for binary classification."""
    if "roc_proba" not in results:
        return go.Figure()

    y_test = results["y_test"]
    y_prob = results["roc_proba"]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = results["metrics"].get("ROC-AUC", 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC={auc:.4f})", line=dict(color="#a78bfa")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="#ef4444", dash="dash"),
                             name="Random"))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                      template="plotly_dark", height=450)
    return fig


def plot_confusion_matrix(results: Dict) -> go.Figure:
    """Heatmap confusion matrix for classification."""
    if "confusion_matrix" not in results:
        return go.Figure()

    cm = results["confusion_matrix"]
    labels = [str(c) for c in sorted(results["y_test"].unique())]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Purples",
        title="Confusion Matrix", template="plotly_dark",
        labels=dict(x="Predicted", y="Actual"),
        x=labels, y=labels,
    )
    return fig
