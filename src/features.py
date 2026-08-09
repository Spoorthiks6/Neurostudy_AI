"""Shared EEG feature derivation.

Import this everywhere instead of redefining alpha/beta ratios in
multiple files — that duplication was how the old codebase drifted
(feature_engineering.py and generate_dashboard_report.py each had
their own slightly different copy of this logic).
"""
import pandas as pd


def derive_band_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["alpha"] = df["alphaLow"] + df["alphaHigh"]
    df["beta"] = df["betaLow"] + df["betaHigh"]

    df["alpha_beta"] = df["alpha"] / (df["beta"] + 1e-8)
    df["theta_beta"] = df["theta"] / (df["beta"] + 1e-8)
    df["alpha_theta"] = df["alpha"] / (df["theta"] + 1e-8)
    df["attention_meditation"] = df["attention"] - df["meditation"]
    df["engagement"] = df["attention"] + df["meditation"]

    return df
