"""
Step 6 — use the trained model to predict activity for a NEW session
you haven't labeled yet.

This is also the logic you'd reuse later for live/real-time detection:
same windowing -> same features -> same model.predict(), just called
on a rolling buffer of the last WINDOW_SIZE samples instead of a whole
file.

Usage:
    python src/predict_activity.py path/to/new_session.xlsx
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from config import (
    MODEL_FILE,
    FEATURE_COLUMNS_FILE,
    EEG_CORE_COLUMNS,
    POOR_SIGNAL_THRESHOLD,
    WINDOW_SIZE,
    STEP_SIZE,
)
from features import derive_band_features
from windowing import window_stats, FEATURE_COLS


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[df["poorSignal"].notna()]
    df = df[df["poorSignal"] <= POOR_SIGNAL_THRESHOLD]
    for col in EEG_CORE_COLUMNS + ["blinkStrength", "eegRawValue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=EEG_CORE_COLUMNS)
    return df.sort_values("timestampMs").reset_index(drop=True)


def build_windows_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df.iloc[start : start + WINDOW_SIZE]
        feats = {}
        for col in FEATURE_COLS:
            stats = window_stats(window[col].values)
            for stat_name, val in stats.items():
                feats[f"{col}_{stat_name}"] = val
        rows.append(feats)
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/predict_activity.py path/to/new_session.xlsx")
        sys.exit(1)

    session_path = sys.argv[1]

    if not MODEL_FILE.exists():
        raise FileNotFoundError("No trained model found — run train_baseline.py first.")

    model = joblib.load(MODEL_FILE)
    with open(FEATURE_COLUMNS_FILE, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    df = load_and_clean(session_path)
    df = derive_band_features(df)

    if len(df) < WINDOW_SIZE:
        raise RuntimeError(f"Session too short ({len(df)} rows) to form even one window.")

    windows_df = build_windows_for_prediction(df)
    X = windows_df[feature_cols].values

    predictions = model.predict(X)
    probs = model.predict_proba(X)
    confidences = probs.max(axis=1)

    print(f"Windows predicted: {len(predictions)}")
    vote_counts = Counter(predictions)
    majority_activity, majority_count = vote_counts.most_common(1)[0]

    print("\nPer-window vote breakdown:")
    for activity, count in vote_counts.most_common():
        pct = 100 * count / len(predictions)
        print(f"  {activity}: {count} windows ({pct:.1f}%)")

    print(f"\nPredicted overall activity: {majority_activity}")
    print(f"Majority confidence: {100 * majority_count / len(predictions):.1f}% of windows")
    print(f"Average model confidence per window: {np.mean(confidences):.3f}")


if __name__ == "__main__":
    main()
