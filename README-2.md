# NeuroStudy AI — Data Pipeline

Rebuilt from scratch to fix two bugs from the earlier version:

1. **Activity labels were guessed from filenames** via keyword matching, so
   files like `lekhana.xlsx` or `ranjini ma'am.xlsx` fell through every
   check and the raw filename became the "activity" — which is why the old
   dashboard showed bars named after people instead of `reels` / `studying`
   / `sleep` / `shloka` / `listening`.
   **Fix:** `config/activity_map.json` is now an explicit, required lookup
   table. The pipeline refuses to run on any file that isn't listed there,
   so mislabeling can't happen silently.

2. **Fatigue/break/retention used fixed absolute thresholds**, but
   NeuroSky's `attention`/`meditation` values are compressed into a narrow
   real-world range — so almost every session landed in the same bucket
   (`recommended_break_min = 20` for everyone).
   **Fix:** `session_report.py` now rescales fatigue and attention against
   the **actual min/max seen in your own dataset** (percentile-based)
   before computing break/retention/duration, so sessions differentiate
   from each other.

## Folder structure

```
neurostudy-ai/
├── config/
│   └── activity_map.json      # EDIT THIS — maps each filename to an activity
├── data/
│   ├── raw/                   # put your recorded .xlsx sessions here
│   └── processed/             # prepare_dataset.py writes cleaned_sessions.csv here
├── reports/                   # session_report.py + llm_report.py write output here
├── models/                    # reserved — see "What's deliberately missing" below
├── src/
│   ├── config.py               # all paths & constants in one place
│   ├── prepare_dataset.py      # Step 1: load, clean, label
│   ├── features.py             # shared feature math (imported everywhere)
│   ├── eda_check.py            # Step 2: DO NOT SKIP — check separability first
│   ├── session_report.py       # Step 3: dashboard + report generation
│   └── llm_report.py           # Step 4: LLM narrative layer (optional)
├── run_pipeline.py             # runs steps 1-4 in order
├── requirements.txt
└── README.md
```

## Setup

```bash
cd neurostudy-ai
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Before your first run

1. Copy your recorded `.xlsx` sessions into `data/raw/`.
2. Open `config/activity_map.json` and make sure **every** filename in
   `data/raw/` (without `.xlsx`, lowercase) has an entry. I've pre-filled
   it with guesses based on the filenames you'd shared earlier
   (`ranjini ma'am` → `listening`, `abhipsa` → `reels`, etc.) — **check
   these are actually correct**, I was guessing from names alone. Also
   decide whether `meditation`, `demo`, and `talking` should map into your
   five target categories (`reels`, `studying`, `sleep`, `shloka`,
   `listening`) or stay as their own categories — that's a judgment call
   only you can make about what those sessions actually were.
3. (Optional) `export ANTHROPIC_API_KEY=your_key_here` for LLM narratives.
   Without it, `llm_report.py` writes a simpler rule-based summary instead.

## Running

```bash
python run_pipeline.py
```

This runs, in order:
- `prepare_dataset.py` — clean + label
- `eda_check.py` — **check this before trusting anything downstream.**
  It plots attention/meditation/alpha-beta/theta-beta/engagement by
  activity. If the boxes overlap heavily with no shift in the median,
  there isn't yet enough separable signal to model reliably — collect
  more or longer sessions per activity before going further.
- `session_report.py` — generates `reports/index.html` (the dashboard)
- `llm_report.py` — generates natural-language coaching notes

Then open `reports/index.html` in a browser.

You can also run any step individually, e.g. `python src/eda_check.py`,
once `data/processed/cleaned_sessions.csv` exists.

## Step 5 — Activity classifier (added once EDA confirmed separability)

- `src/windowing.py` — slides a window (default 20 rows, 50% overlap) over
  each session and computes mean/std/min/max/slope per feature, producing
  one row per window in `data/processed/window_features.csv`.
- `src/train_baseline.py` — trains a `RandomForestClassifier` on those
  windows. **The train/test split is by session (`source_file`), not by
  window** — this matters: splitting randomly by window lets windows from
  the same session leak into both train and test, which inflates accuracy
  because the model partly memorizes that session's personal baseline
  instead of learning general activity patterns. Session-aware splitting
  is the honest evaluation, and it's the kind of methodological detail
  that shows rigor in a viva. Outputs:
  - `reports/confusion_matrix.png`
  - `reports/feature_importance.png` — which EEG features actually drove
    the classification (great for your report's "results" section)
  - `models/activity_classifier.joblib` + `models/feature_columns.json`
- `src/predict_activity.py` — run the trained model on a brand-new,
  unlabeled session: `python src/predict_activity.py path/to/file.xlsx`.
  This is also the exact logic you'd reuse for live/real-time detection
  later — same windowing, same features, same `model.predict()`, just
  called on a rolling buffer instead of a finished file.

### A note on how much data you need

Random Forest on windowed features works with far less data than a
CNN/LSTM — a few sessions per activity is enough for a first honest
result, though more sessions (especially more *distinct* sessions, not
just longer ones) will make the session-aware split more meaningful. If
`train_baseline.py` warns about too few sessions, that's it telling you
the accuracy number is a sanity check, not a final metric yet.

### What's still deliberately missing

`train_cnn_lstm.py` and raw-sequence deep learning are left out for now.
Once the Random Forest baseline is solid (or once you have enough data
that a baseline plateaus below what you need), a CNN/LSTM on raw
windows is a natural next step and a good "stretch" section for your
report — ask and I'll build it reusing the same `windowing.py` output.
