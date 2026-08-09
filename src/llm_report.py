"""
Step 4 (optional).

Turns the per-session numeric metrics into short natural-language
coaching text using an LLM. The LLM only ever sees the ALREADY-COMPUTED
numbers from session_report.py — never raw EEG — so it can't invent
numbers, only phrase the ones you've already validated.

Requires: pip install anthropic
Set ANTHROPIC_API_KEY in your environment before running. Without it,
a plain-text fallback summary is written instead (no API call).

Run: python src/llm_report.py
"""
import os
import pandas as pd

from config import SESSION_SUMMARY_FILE, REPORTS_DIR

SESSION_TEXT_FILE = REPORTS_DIR / "llm_session_summary.txt"
DASHBOARD_TEXT_FILE = REPORTS_DIR / "llm_dashboard_insights.txt"

SYSTEM_PROMPT = (
    "You are a calm, encouraging study coach. You are given structured "
    "metrics from a student's EEG-tracked study session. Write 2-4 short "
    "sentences of practical, specific advice based ONLY on the numbers "
    "given. Never invent numbers that aren't in the input. Never make "
    "medical or diagnostic claims — this is a wellness/coaching tool, "
    "not a clinical device."
)


def _client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed — run: pip install anthropic")
        return None
    return anthropic.Anthropic(api_key=api_key)


def _call_llm(client, prompt: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def fallback_session_text(row: pd.Series) -> str:
    return (
        f"Session '{row['activity']}' ({row['source_file']}) lasted {row['duration_min']:.1f} min. "
        f"Deep focus: {row['deep_focus_min']:.1f} min, distraction: {row['distraction_min']:.1f} min. "
        f"Fatigue: {row['mental_fatigue']}. Predicted retention: {row['predicted_retention_pct']:.0f}%. "
        f"Recommended break: {row['recommended_break_min']} min. "
        f"Suggested duration tomorrow: {row['tomorrow_suggested_duration_min']} min."
    )


def generate_narratives(summary_df: pd.DataFrame):
    client = _client()

    session_lines = []
    for _, row in summary_df.iterrows():
        if client:
            prompt = f"Session metrics:\n{row.to_json()}\n\nWrite the coaching note."
            text = _call_llm(client, prompt)
        else:
            text = fallback_session_text(row)
        session_lines.append(text)

    if client:
        dashboard_prompt = (
            "Here are metrics for all sessions this student has recorded:\n"
            + summary_df.to_json(orient="records")
            + "\n\nWrite a short overall insight (3-5 sentences) comparing "
              "which activities were associated with higher focus/retention "
              "and what the student should try next."
        )
        dashboard_text = _call_llm(client, dashboard_prompt)
    else:
        top = summary_df.sort_values("predicted_retention_pct", ascending=False).iloc[0]
        low = summary_df.sort_values("predicted_retention_pct").iloc[0]
        dashboard_text = (
            f"Across {len(summary_df)} sessions, '{top['activity']}' showed the highest "
            f"predicted retention ({top['predicted_retention_pct']:.0f}%), while "
            f"'{low['activity']}' showed the lowest ({low['predicted_retention_pct']:.0f}%). "
            "(Set ANTHROPIC_API_KEY for a richer LLM-generated summary.)"
        )

    return "\n\n".join(session_lines), dashboard_text


def main() -> None:
    if not SESSION_SUMMARY_FILE.exists():
        raise FileNotFoundError("Run session_report.py first.")

    df = pd.read_csv(SESSION_SUMMARY_FILE)
    session_text, dashboard_text = generate_narratives(df)

    SESSION_TEXT_FILE.write_text(session_text, encoding="utf-8")
    DASHBOARD_TEXT_FILE.write_text(dashboard_text, encoding="utf-8")

    print(f"Saved: {SESSION_TEXT_FILE}")
    print(f"Saved: {DASHBOARD_TEXT_FILE}")


if __name__ == "__main__":
    main()
