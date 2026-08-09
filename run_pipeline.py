"""Run the full NeuroStudy AI pipeline end to end, stopping on the first
failure so you fix problems in order rather than debugging everything
at once."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"

STEPS = [
    ("Step 1: Preparing dataset", "prepare_dataset.py"),
    ("Step 2: Checking feature separability", "eda_check.py"),
    ("Step 3: Generating session report", "session_report.py"),
    ("Step 4: Windowing features for ML", "windowing.py"),
    ("Step 5: Training activity classifier", "train_baseline.py"),
    ("Step 6: Generating LLM narratives", "llm_report.py"),
]


def main():
    for label, script in STEPS:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(SRC / script)])
        if result.returncode != 0:
            print(f"\nStopped: {script} failed. Fix the issue above before continuing.")
            sys.exit(1)

    print("\nAll done. Open reports/index.html in a browser to view the dashboard.")


if __name__ == "__main__":
    main()
