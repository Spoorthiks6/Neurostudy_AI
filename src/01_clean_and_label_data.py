from pathlib import Path
import json
import pandas as pd

from config import (
    RAW_DIR,
    ACTIVITY_MAP_FILE,
    PERSON_MAP_FILE,
    CLEANED_SESSIONS_FILE,
    EXPECTED_COLUMNS,
    EEG_CORE_COLUMNS,
    POOR_SIGNAL_THRESHOLD,
)


# Load a JSON lookup table and ignore internal metadata keys.
def load_json_map(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    return {k: v for k, v in raw_map.items() if not k.startswith("_")}


# Read the activity name mapping used to translate raw filenames into clean labels.
def load_activity_map() -> dict:
    if not ACTIVITY_MAP_FILE.exists():
        raise FileNotFoundError(
            f"{ACTIVITY_MAP_FILE} not found. See README.md to set it up."
        )
    return load_json_map(ACTIVITY_MAP_FILE)


# Read optional person labels for later LOPO evaluation; missing labels are allowed.
def load_person_map() -> dict:
    """Person labels are optional — used only for Leave-One-Person-Out
    evaluation later. Missing entries print a warning but don't stop
    the pipeline, since activity classification alone doesn't need them."""
    if not PERSON_MAP_FILE.exists():
        print(f"NOTE: {PERSON_MAP_FILE} not found — 'person' column will be 'unknown' for all rows.")
        return {}
    return load_json_map(PERSON_MAP_FILE)


# Match a session file name to a canonical activity label from the config map.
def resolve_activity(path: Path, activity_map: dict) -> str:
    stem = path.stem.strip().lower()
    if stem in activity_map:
        return activity_map[stem]
    raise KeyError(
        f"No activity mapping for '{path.name}'. "
        f'Add "{stem}": "<activity>" to {ACTIVITY_MAP_FILE}.'
    )


# Match a session file name to a participant label if it has been provided.
def resolve_person(path: Path, person_map: dict) -> str:
    stem = path.stem.strip().lower()
    person = person_map.get(stem, "unknown")
    if person == "REPLACE_ME":
        return "unknown"
    return person


# Read one Excel session and attach the derived activity and person metadata.
def load_raw_file(path: Path, activity_map: dict, person_map: dict) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.insert(0, "source_file", path.name)
    df["activity"] = resolve_activity(path, activity_map)
    df["person"] = resolve_person(path, person_map)
    return df


# Drop empty or bad EEG rows so the dataset only keeps usable measurements.
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep_cols = [c for c in EXPECTED_COLUMNS if c in df.columns] + ["source_file", "activity", "person"]
    df = df[keep_cols]

    df = df[df["poorSignal"].notna()]
    df = df[df["poorSignal"] <= POOR_SIGNAL_THRESHOLD]

    for col in EEG_CORE_COLUMNS + ["poorSignal", "blinkStrength", "eegRawValue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=EEG_CORE_COLUMNS)
    return df


# Loop through every raw session, clean each one, and concatenate valid rows together.
def combine_and_clean_all() -> pd.DataFrame:
    activity_map = load_activity_map()
    person_map = load_person_map()
    xlsx_files = sorted(
        p for p in RAW_DIR.glob("*.xlsx")
        if not p.name.startswith("~$") and not p.name.startswith(".~")
    )

    if not xlsx_files:
        raise FileNotFoundError(
            f"No .xlsx files found in {RAW_DIR}. Copy your recorded sessions there first."
        )

    frames, skipped, unknown_person = [], [], []
    for path in xlsx_files:
        try:
            raw = load_raw_file(path, activity_map, person_map)
        except KeyError as exc:
            print(f"SKIPPING {path.name}: {exc}")
            skipped.append(path.name)
            continue
        except Exception as exc:
            print(f"SKIPPING {path.name}: could not read file — {type(exc).__name__}: {exc}")
            skipped.append(path.name)
            continue

        try:
            if raw["person"].iloc[0] == "unknown":
                unknown_person.append(path.name)
            cleaned = clean_dataframe(raw)
            activity_shown = cleaned["activity"].iloc[0] if len(cleaned) else "N/A"
            print(f"{path.name}: {len(raw)} rows -> {len(cleaned)} kept  [activity={activity_shown}]")
            if len(cleaned) == 0:
                print(f"  NOTE: 0 rows survived cleaning for {path.name} — check poorSignal values in this file.")
                skipped.append(path.name)
                continue
            frames.append(cleaned)
        except Exception as exc:
            print(f"SKIPPING {path.name}: error while cleaning — {type(exc).__name__}: {exc}")
            skipped.append(path.name)

    if not frames:
        raise RuntimeError(
            "No usable sessions after cleaning. Check the activity map and "
            "the poorSignal threshold in config.py."
        )

    combined = pd.concat(frames, ignore_index=True)

    if skipped:
        print(f"\n{len(skipped)} file(s) skipped — fix and re-run: {skipped}")

    if unknown_person:
        print(
            f"\n{len(unknown_person)} file(s) have no person label (needed only for "
            f"LOPO evaluation later): {unknown_person}\nFill these in {PERSON_MAP_FILE} when ready."
        )

    return combined


# Save the cleaned dataset and print a compact summary of the final record counts.
def main() -> None:
    combined = combine_and_clean_all()
    combined.to_csv(CLEANED_SESSIONS_FILE, index=False)

    print(f"\nSaved: {CLEANED_SESSIONS_FILE}")
    print(f"Total rows: {len(combined)}")
    print("\nRows per activity:")
    print(combined["activity"].value_counts())
    print("\nSessions per activity:")
    print(combined.groupby("activity")["source_file"].nunique())
    print("\nSessions per person:")
    print(combined.groupby("person")["source_file"].nunique())


if __name__ == "__main__":
    main()
