NeuroStudy AI — minimal README

This repository processes NeuroSky EEG session data, extracts features,
trains a baseline activity classifier, and generates reports.

Quick run (after placing .xlsx session files in `data/raw/`):

```bash
python3 run_pipeline.py
```

The runner automatically uses the project's `.venv` when it exists, so you
do not need to activate it first.

See `config/` for mapping files and `reports/` for outputs.

Session coaching notes identify the participant using `config/person_map.json`
and name the recorded activity. Focus, distraction, fatigue, and next-session
recommendations are experimental heuristics, not validated EEG assessments.
Set `GROQ_API_KEY` in the environment to generate notes with Groq; without it,
the pipeline uses a local coaching template.
