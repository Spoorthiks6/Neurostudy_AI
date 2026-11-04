import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

from config import (
    WINDOW_FEATURES_FILE,
    MODEL_FILE,
    FEATURE_COLUMNS_FILE,
    CONFUSION_MATRIX_IMAGE,
    FEATURE_IMPORTANCE_IMAGE,
    RANDOM_STATE,
    MODEL_FEATURE_BASES,
    WINDOW_STAT_NAMES,
)

NON_FEATURE_COLS = {"source_file", "activity", "person"}
MIN_SESSIONS_PER_ACTIVITY_WARNING = 2


# Read the windowed feature table and treat all non-label columns as model inputs.
def load_data():
    df = pd.read_csv(WINDOW_FEATURES_FILE)
    feature_cols = [
        f"{base}_{stat}"
        for base in MODEL_FEATURE_BASES
        for stat in WINDOW_STAT_NAMES
    ]
    missing = sorted(set(feature_cols) - set(df.columns))
    if missing:
        raise RuntimeError(f"Configured model features are missing: {missing}")
    return df, feature_cols


# Warn if some activities do not have enough sessions for stable cross-validation.
def check_class_coverage(df: pd.DataFrame) -> None:
    sessions_per_activity = df.groupby("activity")["source_file"].nunique()
    print("Sessions per activity:")
    print(sessions_per_activity.to_string())
    thin = sessions_per_activity[sessions_per_activity < MIN_SESSIONS_PER_ACTIVITY_WARNING]
    if len(thin) > 0:
        print(
            "\nNOTE: these activities have fewer than "
            f"{MIN_SESSIONS_PER_ACTIVITY_WARNING} sessions, so LOSO-CV folds "
            "involving them will be skipped or high-variance until you record more:"
        )
        print(thin.to_string())


# Hold out one session at a time to evaluate how well the model generalizes across sessions.
def leave_one_session_out_cv(df: pd.DataFrame, feature_cols: list[str]):
    sessions = df["source_file"].unique()
    all_true, all_pred, per_session_acc = [], [], {}

    for held_out in sessions:
        train_df = df[df["source_file"] != held_out]
        test_df = df[df["source_file"] == held_out]

        activity = test_df["activity"].iloc[0]
        if activity not in train_df["activity"].unique():
            print(f"Skipping '{held_out}': '{activity}' has no other session to train on.")
            continue

        X_train, y_train = train_df[feature_cols].values, train_df["activity"].values
        X_test, y_test = test_df[feature_cols].values, test_df["activity"].values

        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        per_session_acc[held_out] = (activity, acc)
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        print(f"Held out '{held_out}' [{activity}] ({len(y_test)} windows): accuracy = {acc:.3f}")

    return all_true, all_pred, per_session_acc


# Render a pooled confusion matrix for the cross-validation predictions.
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Leave-One-Session-Out CV (pooled)")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_IMAGE)
    plt.close(fig)


# Plot the most important features in the final model to explain what it is learning.
def plot_feature_importance(model, feature_cols, top_n=20):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_cols[i] for i in order]
    top_vals = importances[order]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(top_feats[::-1], top_vals[::-1], color="tab:purple")
    ax.set_title("Top Feature Importances (final model, trained on all data)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_IMAGE)
    plt.close(fig)


# Train the baseline random forest, evaluate it, save the model, and export plots.
def main() -> None:
    if not WINDOW_FEATURES_FILE.exists():
        raise FileNotFoundError("Run windowing.py first.")

    df, feature_cols = load_data()

    if df["activity"].nunique() < 2:
        raise RuntimeError("Need at least 2 distinct activities to train a classifier.")

    check_class_coverage(df)

    print(f"\nRunning Leave-One-Session-Out CV across {df['source_file'].nunique()} sessions...\n")
    all_true, all_pred, per_session_acc = leave_one_session_out_cv(df, feature_cols)

    if not all_true:
        raise RuntimeError(
            "No valid LOSO folds ran — every activity needs at least 2 sessions "
            "for this evaluation to work at all. Record more sessions per activity."
        )

    labels = sorted(set(all_true) | set(all_pred))
    report_text = classification_report(all_true, all_pred, labels=labels, zero_division=0)
    overall_acc = accuracy_score(all_true, all_pred)

    print("\nPooled classification report (every session held out once):")
    print(report_text)
    print(f"Overall LOSO-CV accuracy: {overall_acc:.3f}")

    plot_confusion_matrix(all_true, all_pred, labels)

    X_all = df[feature_cols].values
    y_all = df["activity"].values
    final_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X_all, y_all)

    plot_feature_importance(final_model, feature_cols)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)
    with open(FEATURE_COLUMNS_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nSaved final model (trained on ALL data): {MODEL_FILE}")
    print(f"Saved: {CONFUSION_MATRIX_IMAGE}")
    print(f"Saved: {FEATURE_IMPORTANCE_IMAGE}")
    print(
        "\nCite the LOSO-CV classification report above as your evaluation result — "
        "not the final model's fit, since that model has seen all its own training data."
    )


if __name__ == "__main__":
    main()
