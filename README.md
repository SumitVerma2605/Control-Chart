# 📊 DataFlow Analytics

> **Production-ready Data Analysis & Statistical Process Control Platform**

A complete, modular data analysis system built with Python and Streamlit. Supports CSV/Excel/GitHub data ingestion, automated preprocessing, exploratory data analysis (EDA), SPC control charts, and machine learning — all through an interactive web dashboard.

---

## ✨ Features

| Module | Capabilities |
|--------|-------------|
| 📥 **Data Ingestion** | CSV, Excel, GitHub raw URLs, direct HTTP links |
| 🧹 **Preprocessing** | Auto imputation, outlier detection/treatment, encoding, normalization |
| 📊 **EDA** | Summary stats, correlation heatmaps, distributions, boxplots, time series |
| 📈 **SPC Charts** | X-bar/R, X-bar/S, I-MR, P Chart, C Chart with Western Electric rules |
| 🤖 **ML Models** | Linear/Logistic Regression, Random Forest, XGBoost with CV & metrics |
| 📉 **Visualizations** | Interactive Plotly charts, pair plots, residual plots, ROC curves |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/dataflow-analytics.git
cd dataflow-analytics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS/Linux
# OR
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
project/
│── app.py                   # Streamlit dashboard (main entry point)
│── config.yaml              # Configuration (thresholds, methods, settings)
│── requirements.txt         # Python dependencies
│── README.md
│
├── data/                    # Sample datasets (place your data here)
│
├── src/
│   ├── preprocessing.py     # Schema detection, imputation, encoding, scaling
│   ├── eda.py               # Summary stats, correlation, distributions, boxplots
│   ├── control_charts.py    # SPC charts (X-bar, R, S, I-MR, P, C)
│   ├── modeling.py          # ML training, evaluation, feature importance
│   └── visualization.py    # Shared chart utilities and export helpers
│
└── logs/
    └── app.log              # Application logs
```

---

## 📥 Data Input

### Upload Files
- **CSV** and **Excel** (.xlsx, .xls) up to 200 MB
- Drag-and-drop via the Streamlit sidebar

### GitHub / URL
Paste any of the following:
- GitHub blob URL: `https://github.com/user/repo/blob/main/data.csv`
- GitHub raw URL: `https://raw.githubusercontent.com/user/repo/main/data.csv`
- Any direct CSV/Excel link

### Sample Datasets (built-in)
- **Iris** – classification
- **Titanic** – binary classification
- **Car MPG** – regression
- **Apple Stock** – time series

---

## 🧹 Preprocessing Options

| Option | Choices |
|--------|---------|
| Missing values | mean, median, mode, drop, forward_fill |
| Outlier detection | IQR (default threshold 1.5×), Z-score |
| Outlier treatment | clip, remove, winsorize |
| Encoding | one-hot (low cardinality), label encoding |
| Normalization | Standard (z-score), MinMax, Robust |

All steps are logged and a full preprocessing report is generated.

---

## 📈 Control Charts

| Chart | Use Case |
|-------|----------|
| **X-bar & R** | Variable data, subgroup size 2–10 |
| **X-bar & S** | Variable data, larger subgroups |
| **Individuals (I-MR)** | One measurement per time period |
| **P Chart** | Proportion defective (variable sample sizes) |
| **C Chart** | Count of defects per unit |

All charts:
- Auto-calculate UCL, CL, LCL using standard SPC constants
- Highlight out-of-control points in red
- Detect Western Electric violations (8-run rule, 6-trend rule, 2-of-3 rule)
- Are fully interactive (zoom, hover, export)

---

## 🤖 ML Models

### Regression
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

**Metrics:** RMSE, MAE, R²

### Classification
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

**Metrics:** Accuracy, F1 (weighted), ROC-AUC

All models include:
- Stratified train/test split
- k-fold cross-validation
- Feature importance visualization
- Downloadable predictions CSV

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
preprocessing:
  missing_value_strategy: "mean"
  outlier_method: "iqr"
  outlier_threshold: 1.5

control_charts:
  sigma_level: 3        # Control limit multiplier
  detection_runs_rule: 8

modeling:
  test_size: 0.2
  cv_folds: 5
```

---

## 🐍 Using as a Library

```python
from src.preprocessing import run_preprocessing_pipeline
from src.eda import summary_statistics, correlation_matrix
from src.control_charts import imr_chart
from src.modeling import train_model, prepare_features

import pandas as pd

# Load data
df = pd.read_csv("data/your_data.csv")

# Preprocess
proc_df, report = run_preprocessing_pipeline(df, missing_strategy="median")

# EDA
stats = summary_statistics(proc_df)

# Control chart
fig, stats = imr_chart(proc_df, value_col="measurement")
fig.show()

# ML model
X, y = prepare_features(proc_df, target_col="target")
results = train_model(X, y, model_name="Random Forest Regressor")
print(results["metrics"])
```

---

## 📦 Dependencies

```
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
plotly >= 5.14
scikit-learn >= 1.3
scipy >= 1.10
streamlit >= 1.28
openpyxl >= 3.1
requests >= 2.28
pyyaml >= 6.0
xgboost >= 1.7
statsmodels >= 0.14
```

---

## 📸 Screenshots

> Run the app and load the Iris or Titanic sample datasets to see the dashboard in action.

**Tabs available:**
1. 👁️ **Preview** – Raw data with schema detection
2. 🧹 **Preprocess** – Configure and run the pipeline
3. 📊 **EDA** – 7 sub-panels: summary, correlation, distributions, boxplots, time series, missing, pair plot
4. 📈 **Control Charts** – 5 chart types with auto control limits
5. 🤖 **ML Model** – Train, evaluate, and download predictions

---

## 📄 License

MIT License. Free to use and modify.
