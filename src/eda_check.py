"""
Step 2 — do this BEFORE building any model or trusting the dashboard.

Plots the distribution of key EEG-derived features across your
activities and prints the median per activity. If the medians look
nearly identical across activities, that means either:
  - NeuroSky's attention/meditation output isn't sensitive enough to
    tell these activities apart, or
  - you don't have enough distinct sessions per activity yet, or
  - the recordings themselves have contact/signal-quality issues.

No amount of downstream modeling fixes a lack of separable signal —
check this first, every time you add new sessions.

Run: python src/eda_check.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CLEANED_SESSIONS_FILE, EDA_IMAGE_FILE
from features import derive_band_features

FEATURES_TO_CHECK = ["attention", "meditation", "alpha_beta", "theta_beta", "engagement"]


def main() -> None:
    if not CLEANED_SESSIONS_FILE.exists():
        raise FileNotFoundError("Run prepare_dataset.py first.")

    df = pd.read_csv(CLEANED_SESSIONS_FILE)
    df = derive_band_features(df)

    n_activities = df["activity"].nunique()
    if n_activities < 2:
        print(f"Only {n_activities} distinct activity found — add more sessions before checking separability.")
        return

    n = len(FEATURES_TO_CHECK)
    fig, axs = plt.subplots(n, 1, figsize=(10, 4 * n), constrained_layout=True)
    if n == 1:
        axs = [axs]

    for ax, feat in zip(axs, FEATURES_TO_CHECK):
        df.boxplot(column=feat, by="activity", ax=ax, rot=30)
        ax.set_title(f"{feat} by activity")
        ax.set_xlabel("")

    fig.suptitle("")
    fig.savefig(EDA_IMAGE_FILE)
    plt.close(fig)

    print(f"Saved: {EDA_IMAGE_FILE}")
    print("\nMedian feature values per activity:")
    print(df.groupby("activity")[FEATURES_TO_CHECK].median().round(3))

    print(
        "\nLook at feature_separability.png: if the boxes for each activity "
        "overlap heavily with no visible shift in the median line, there's "
        "not yet enough separation to model reliably — collect more/longer "
        "sessions or revisit the recording setup before moving to Step 3."
    )


if __name__ == "__main__":
    main()
