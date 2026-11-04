import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import (
    WINDOW_FEATURES_FILE,
    LOPO_REPORT_FILE,
    LOPO_CONFUSION_MATRIX_IMAGE,
    SESSION_SUMMARY_FILE,
    MIN_PERSONS_FOR_LOPO,
    RANDOM_STATE,
    MODEL_FEATURE_BASES,
    WINDOW_STAT_NAMES,
)

NON_FEATURE_COLS = {"source_file", "activity", "person"}


def plot_split_comparison(loso_acc: float, lopo_acc: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Leave-one-session-out", "Leave-one-person-out"],
        [loso_acc, lopo_acc],
        color=["#4c78a8", "#e45756"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pooled accuracy")
    ax.set_title("Session-Split vs. Person-Independent Evaluation")
    ax.tick_params(axis="x", rotation=10)
    for bar, accuracy in zip(bars, [loso_acc, lopo_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
        )
    ax.text(
        0.5,
        -0.2,
        "Session-split evaluation excludes the single reading session; LOPO includes all windows.",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(LOPO_REPORT_FILE.with_name("lopo_vs_session_split.png"), dpi=160)
    plt.close(fig)


# Load windowed data and remove unknown-person rows before checking person independence.
def load_data():
    if not WINDOW_FEATURES_FILE.exists():
        raise FileNotFoundError("Run windowing.py first.")

    df = pd.read_csv(WINDOW_FEATURES_FILE)
    if "person" not in df.columns:
        raise RuntimeError(
            "window_features.csv has no 'person' column — re-run prepare_dataset.py "
            "and windowing.py after filling in config/person_map.json."
        )

    unknown_mask = df["person"] == "unknown"
    if unknown_mask.any():
        n = df.loc[unknown_mask, "source_file"].nunique()
        print(f"WARNING: {n} session(s) have person='unknown' and will be excluded from LOPO.")
        df = df[~unknown_mask]

    feature_cols = [
        f"{base}_{stat}"
        for base in MODEL_FEATURE_BASES
        for stat in WINDOW_STAT_NAMES
    ]
    missing = sorted(set(feature_cols) - set(df.columns))
    if missing:
        raise RuntimeError(f"Configured model features are missing: {missing}")
    return df, feature_cols


# Leave one session out to provide the session-split comparison beside LOPO.
def run_loso(df: pd.DataFrame, feature_cols: list[str]):
    all_true, all_pred = [], []

    for held_out in df["source_file"].unique():
        train_df = df[df["source_file"] != held_out]
        test_df = df[df["source_file"] == held_out]
        activity = test_df["activity"].iloc[0]

        if activity not in train_df["activity"].unique():
            print(f"Skipping LOSO session '{held_out}': no training session for '{activity}'.")
            continue

        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train_df[feature_cols].values, train_df["activity"].values)
        y_test = test_df["activity"].values
        y_pred = model.predict(test_df[feature_cols].values)
        all_true.extend(y_test)
        all_pred.extend(y_pred)

    return all_true, all_pred


# Leave one person out at a time to test whether the model generalizes to unseen participants.
def run_lopo(df: pd.DataFrame, feature_cols: list[str]):
    people = sorted(df["person"].unique())

    if len(people) < MIN_PERSONS_FOR_LOPO:
        print(
            f"Only {len(people)} distinct person(s) found ({people}). "
            f"LOPO needs at least {MIN_PERSONS_FOR_LOPO} to be meaningful at all — "
            "recording one or two more people's sessions would make this evaluation "
            "far more convincing for your report. Running anyway on what's available."
        )

    all_true, all_pred, per_person_acc = [], [], {}

    for held_out in people:
        train_df = df[df["person"] != held_out]
        test_df = df[df["person"] == held_out]

        if train_df["activity"].nunique() < 2 or test_df.empty:
            print(f"Skipping '{held_out}': not enough data on either side of the split.")
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df["activity"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["activity"].values

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
        per_person_acc[held_out] = acc
        all_true.extend(y_test)
        all_pred.extend(y_pred)

        print(f"Held out '{held_out}' ({len(y_test)} windows): accuracy = {acc:.3f}")

    return all_true, all_pred, per_person_acc


# Run the LOPO evaluation, save the report, and generate a confusion matrix figure.
def main() -> None:
    df, feature_cols = load_data()
    loso_true, loso_pred = run_loso(df, feature_cols)
    all_true, all_pred, per_person_acc = run_lopo(df, feature_cols)

    if not all_true:
        raise RuntimeError("LOPO produced no results — check person_map.json and data volume.")

    labels = sorted(set(all_true) | set(all_pred))
    report_text = classification_report(all_true, all_pred, labels=labels, zero_division=0)
    overall_acc = accuracy_score(all_true, all_pred)
    loso_acc = accuracy_score(loso_true, loso_pred) if loso_true else float("nan")
    plot_split_comparison(loso_acc, overall_acc)
    print(f"Session-split LOSO accuracy (pooled): {loso_acc:.3f}")

    print("\nOverall LOPO classification report (pooled across all held-out people):")
    print(report_text)
    print(f"Overall LOPO accuracy: {overall_acc:.3f}")

    cm = confusion_matrix(all_true, all_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("LOPO Confusion Matrix (person-independent)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(LOPO_CONFUSION_MATRIX_IMAGE)
    plt.close(fig)

    with open(LOPO_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("Leave-One-Person-Out Evaluation\n")
        f.write("=" * 40 + "\n\n")
        f.write("Per-person accuracy (this person held out entirely from training):\n")
        for person, acc in per_person_acc.items():
            f.write(f"  {person}: {acc:.3f}\n")
        f.write(f"\nOverall pooled accuracy: {overall_acc:.3f}\n\n")
        f.write(f"Session-split LOSO pooled accuracy: {loso_acc:.3f}\n\n")
        f.write("Pooled classification report:\n")
        f.write(report_text)
        f.write(
            "\nInterpretation: LOSO allows other sessions from the held-out person in training, "
            "whereas LOPO holds out that person's sessions entirely. The difference between "
            "these estimates is the observed person-generalization gap."
        )

    print(f"\nSaved: {LOPO_REPORT_FILE}")
    print(f"Saved: {LOPO_CONFUSION_MATRIX_IMAGE}")
    print(f"Saved: {LOPO_REPORT_FILE.with_name('lopo_vs_session_split.png')}")


if __name__ == "__main__":
    main()
