import pandas as pd


# Center selected signals around each recording session's own baseline.
def normalize_session_baselines(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    df = df.copy()
    for feature in feature_cols:
        if feature not in df.columns:
            raise KeyError(f"Cannot normalize missing feature: {feature}")
        df[feature] = df[feature] - df.groupby("source_file")[feature].transform("mean")
    return df


# Add derived EEG band and focus metrics used by the modeling and reporting steps.
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
