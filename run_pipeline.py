import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"


def use_project_venv():
    venv_dir = PROJECT_ROOT / ".venv"
    venv_python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    if not venv_dir.exists() or Path(sys.prefix).resolve() == venv_dir.resolve():
        return
    if not venv_python.is_file():
        raise FileNotFoundError(
            f"Found {venv_dir}, but its Python interpreter is missing: {venv_python}"
        )

    print(f"Using project environment: {venv_python}", flush=True)
    result = subprocess.run([str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.exit(result.returncode)

REQUIRED_STEPS = [
    ("Step 1: Cleaning and labeling data", "01_clean_and_label_data.py"),
    ("Step 2: Checking feature separability", "02_check_feature_separability.py"),
    ("Step 3: Extracting window features", "03_extract_window_features.py"),
    ("Step 4: Training Random Forest baseline", "04_train_random_forest_baseline.py"),
]

OPTIONAL_STEPS = [
    ("Step 5: Person-independent (LOPO) evaluation", "05_evaluate_person_independence.py"),
]

FINAL_STEPS = [
    ("Step 6: Generating session dashboard", "06_generate_session_dashboard.py"),
    ("Step 7: Generating LLM coaching notes", "07_generate_llm_coaching_notes.py"),
    ("Updating dashboard with coaching notes", "06_generate_session_dashboard.py"),
]


def run_step(index, total, label, script, required=True):
    print(f"\n[{index}/{total}] {label}", flush=True)
    result = subprocess.run([sys.executable, str(SRC / script)])
    if result.returncode != 0:
        if required:
            print(f"  Failed: {script} (exit code {result.returncode})")
            sys.exit(result.returncode)
        else:
            print(f"  Warning: optional step failed; continuing.")
    else:
        print("  Done")


def main():
    steps = (
        [(label, script, True) for label, script in REQUIRED_STEPS]
        + [(label, script, False) for label, script in OPTIONAL_STEPS]
        + [(label, script, True) for label, script in FINAL_STEPS]
    )
    total = len(steps)

    for index, (label, script, required) in enumerate(steps, start=1):
        clean_label = label.split(": ", 1)[-1]
        run_step(index, total, clean_label, script, required)

    print("\nPipeline complete. Dashboard: reports/index.html")


if __name__ == "__main__":
    use_project_venv()
    main()
