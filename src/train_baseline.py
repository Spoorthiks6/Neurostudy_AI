"""
Step 5 — train a baseline activity classifier on windowed features.

IMPORTANT: the train/test split is done by SESSION (source_file), not
by individual window. If you split randomly by window, windows from
the same session end up in both train and test — the model then
partly "memorizes" that session's specific baseline instead of
learning general activity patterns, and your accuracy looks
artificially high. Splitting by session is the honest way to evaluate
this, and it's exactly the kind of methodological detail examiners
look for in a final-year project.

Run: python src/train_baseline.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import (
    WINDOW_FEATURES_FILE,
    MODEL_FILE,
    FEATURE_COLUMNS_FILE,
    CONFUSION_MATRIX_IMAGE,
    FEATURE_IMPORTANCE_IMAGE,
    REPORTS_DIR,
    TEST_SIZE,
    RANDOM_STATE,
)

NON_FEATURE_COLS = {"source_file", "activity"}


def load_data():
    df = pd.read_csv(WINDOW_FEATURES_FILE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df["activity"].values
    groups = df["source_file"].values
    return X, y, groups, feature_cols, df


def session_aware_split(X, y, groups):
    n_sessions = len(set(groups))
    if n_sessions < 4:
        print(
            f"WARNING: only {n_sessions} distinct sessions total. A session-aware "
            "split needs several sessions per activity to be meaningful — results "
            "below should be treated as a rough sanity check, not a final metric, "
            "until you've recorded more sessions."
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx


def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Activity Classifier")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_IMAGE)
    plt.close(fig)


def plot_feature_importance(model, feature_cols, top_n=20):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_cols[i] for i in order]
    top_vals = importances[order]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(top_feats[::-1], top_vals[::-1], color="tab:purple")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_IMAGE)
    plt.close(fig)


def main() -> None:
    if not WINDOW_FEATURES_FILE.exists():
        raise FileNotFoundError("Run windowing.py first.")

    X, y, groups, feature_cols, df = load_data()

    if len(set(y)) < 2:
        raise RuntimeError(
            f"Only 1 activity class ({set(y)}) in the data — need at least 2 "
            "distinct activities to train a classifier."
        )

    train_idx, test_idx = session_aware_split(X, y, groups)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train windows: {len(X_train)}  |  Test windows: {len(X_test)}")
    print(f"Train sessions: {sorted(set(groups[train_idx]))}")
    print(f"Test sessions:  {sorted(set(groups[test_idx]))}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = sorted(set(y))

    print("\nClassification report (test sessions, never seen during training):")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    plot_confusion_matrix(y_test, y_pred, labels)
    plot_feature_importance(model, feature_cols)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    with open(FEATURE_COLUMNS_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved feature column order: {FEATURE_COLUMNS_FILE}")
    print(f"Saved: {CONFUSION_MATRIX_IMAGE}")
    print(f"Saved: {FEATURE_IMPORTANCE_IMAGE}")


if __name__ == "__main__":
    main()
