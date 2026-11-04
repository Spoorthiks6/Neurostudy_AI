import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import CLEANED_SESSIONS_FILE, EDA_IMAGE_FILE
from feature_math import derive_band_features

FEATURES_TO_CHECK = ["attention", "meditation", "alpha_beta", "theta_beta", "engagement"]


# Check whether each feature differs meaningfully across activities with one-way ANOVA.
def run_separability_test(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    
    activities = df["activity"].unique()
    rows = []
    for feat in feature_cols:
        groups = [df.loc[df["activity"] == a, feat].dropna().values for a in activities]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        rows.append({"feature": feat, "F_statistic": f_stat, "p_value": p_val})

    result = pd.DataFrame(rows).sort_values("p_value")
    return result


# Load cleaned data, compute derived features, and generate a visual separability report.
def main() -> None:
    if not CLEANED_SESSIONS_FILE.exists():
        raise FileNotFoundError("Run prepare_dataset.py first.")

    df = pd.read_csv(CLEANED_SESSIONS_FILE)
    df = derive_band_features(df)

    n_activities = df["activity"].nunique()
    if n_activities < 2:
        print(f"Only {n_activities} distinct activity found — add more sessions before checking separability.")
        return

    anova_results = run_separability_test(df, FEATURES_TO_CHECK)
    p_map = {row["feature"]: row["p_value"] for _, row in anova_results.iterrows()}

    n = len(FEATURES_TO_CHECK)
    fig, axs = plt.subplots(n, 1, figsize=(10, 4 * n), constrained_layout=True)
    if n == 1:
        axs = [axs]

    for ax, feat in zip(axs, FEATURES_TO_CHECK):
        df.boxplot(column=feat, by="activity", ax=ax, rot=30)
        ax.set_title(f"{feat} by activity")
        ax.set_xlabel("")
        p_val = p_map.get(feat, 1.0)
        ax.text(0.02, 0.95, f"ANOVA p={p_val:.3f}", transform=ax.transAxes, fontsize=8)

    fig.suptitle("")
    fig.savefig(EDA_IMAGE_FILE)
    plt.close(fig)

    print(f"Saved: {EDA_IMAGE_FILE}")
    print("\nMedian feature values per activity:")
    print(df.groupby("activity")[FEATURES_TO_CHECK].median().round(3))

    print("\n" + "=" * 60)
    print("Statistical separability test (one-way ANOVA per feature)")
    print("=" * 60)
    print(anova_results.to_string(index=False))
    print(
        "\np < 0.05 means that feature's mean genuinely differs across "
        "activities (statistically). p >= 0.05 means no detectable "
        "difference — the model has nothing to learn from that feature "
        "for telling activities apart, regardless of how it looks visually.\n"
        "(See the caveat in run_separability_test() docstring about "
        "within-session autocorrelation before over-trusting small p-values.)"
    )

    print(
        "\nLook at feature_separability.png: if the boxes for each activity "
        "overlap heavily with no visible shift in the median line, there's "
        "not yet enough separation to model reliably — collect more/longer "
        "sessions or revisit the recording setup before moving to Step 3."
    )


if __name__ == "__main__":
    main()
