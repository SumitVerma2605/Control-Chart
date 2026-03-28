"""
generate_sample_data.py
-----------------------
Generates sample datasets for testing DataFlow Analytics.
Run: python data/generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT = Path(__file__).parent


def make_manufacturing_spc():
    """Simulated manufacturing process measurements for SPC demo."""
    n = 200
    # Simulate in-control process with a few deliberate shifts
    base = 10 + np.random.normal(0, 0.5, n)
    base[80:90]  += 2.0   # process shift
    base[150:156] = [12.5, 13.0, 13.5, 14.0, 14.5, 15.0]  # trend
    base[180]    = 18.0   # single outlier

    df = pd.DataFrame({
        "timestamp":    pd.date_range("2024-01-01", periods=n, freq="h"),
        "sample_id":    [f"S{i:04d}" for i in range(n)],
        "subgroup":     np.repeat(np.arange(n // 5), 5),
        "measurement":  base,
        "machine":      np.random.choice(["Machine_A", "Machine_B", "Machine_C"], n),
        "operator":     np.random.choice(["Alice", "Bob", "Charlie"], n),
        "defects":      np.random.poisson(2, n),
        "n_inspected":  np.random.randint(50, 200, n),
        "n_defective":  np.random.binomial(50, 0.05, n),
    })
    df["sample_size"] = 50
    df.to_csv(OUT / "manufacturing_spc.csv", index=False)
    print(f"✅ manufacturing_spc.csv ({len(df)} rows)")


def make_sales_regression():
    """Simulated sales data for regression modeling."""
    n = 500
    advertising = np.random.uniform(10, 200, n)
    price       = np.random.uniform(5, 50, n)
    seasonality = np.sin(np.arange(n) * 2 * np.pi / 52) * 20
    noise       = np.random.normal(0, 10, n)
    sales       = 50 + 1.5 * advertising - 0.8 * price + seasonality + noise

    df = pd.DataFrame({
        "date":        pd.date_range("2022-01-01", periods=n, freq="W"),
        "advertising": advertising,
        "price":       price,
        "region":      np.random.choice(["North", "South", "East", "West"], n),
        "channel":     np.random.choice(["Online", "Retail", "Wholesale"], n),
        "sales":       np.maximum(0, sales),
        "returns":     np.random.poisson(3, n),
    })
    # Inject some missing values
    for col in ["advertising", "price"]:
        idx = np.random.choice(n, 15, replace=False)
        df.loc[idx, col] = np.nan

    df.to_csv(OUT / "sales_regression.csv", index=False)
    print(f"✅ sales_regression.csv ({len(df)} rows)")


def make_quality_classification():
    """Simulated quality inspection data for classification."""
    n = 800
    hardness     = np.random.normal(65, 8, n)
    tensile      = np.random.normal(400, 50, n)
    surface_finish = np.random.uniform(0.5, 3.0, n)
    thickness    = np.random.normal(5.0, 0.3, n)
    temperature  = np.random.normal(200, 15, n)

    # Defect probability driven by features
    log_odds = (-8
                + 0.05 * np.abs(hardness - 65)
                + 0.01 * np.abs(tensile - 400)
                + 0.8  * surface_finish
                - 0.5  * thickness
                + 0.02 * np.abs(temperature - 200))
    prob = 1 / (1 + np.exp(-log_odds))
    defect = (np.random.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "batch_id":      [f"B{i:05d}" for i in range(n)],
        "hardness":      hardness,
        "tensile_strength": tensile,
        "surface_finish": surface_finish,
        "thickness":     thickness,
        "temperature":   temperature,
        "material":      np.random.choice(["Steel", "Aluminum", "Copper"], n),
        "shift":         np.random.choice(["Morning", "Afternoon", "Night"], n),
        "defect":        defect,
    })
    df.to_csv(OUT / "quality_classification.csv", index=False)
    print(f"✅ quality_classification.csv ({len(df)} rows)")


if __name__ == "__main__":
    make_manufacturing_spc()
    make_sales_regression()
    make_quality_classification()
    print("\n🎉 All sample datasets generated in ./data/")
