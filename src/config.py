from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
CONFIG_DIR = ROOT_DIR / "config"

ACTIVITY_MAP_FILE = CONFIG_DIR / "activity_map.json"
PERSON_MAP_FILE = CONFIG_DIR / "person_map.json"
CLEANED_SESSIONS_FILE = PROCESSED_DIR / "cleaned_sessions.csv"
SESSION_SUMMARY_FILE = REPORTS_DIR / "session_summary.csv"
DASHBOARD_IMAGE_FILE = REPORTS_DIR / "dashboard_summary.png"
DASHBOARD_HTML_FILE = REPORTS_DIR / "index.html"
EDA_IMAGE_FILE = REPORTS_DIR / "feature_separability.png"

WINDOW_FEATURES_FILE = PROCESSED_DIR / "window_features.csv"
MODEL_FILE = MODELS_DIR / "activity_classifier.joblib"
FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.json"
CONFUSION_MATRIX_IMAGE = REPORTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_IMAGE = REPORTS_DIR / "feature_importance.png"

WINDOW_SIZE = 20          # raw samples per window (NeuroSky ~1Hz -> ~20s window)
STEP_SIZE = 10            # 50% overlap between consecutive windows
MIN_ROWS_PER_SESSION = 30  # skip sessions too short to window reliably
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Keep the modeling experiment focused on features that showed the clearest
# activity separation in the ANOVA check.
MODEL_FEATURE_BASES = ["alpha_beta", "theta_beta"]
WINDOW_STAT_NAMES = ["mean", "std", "min", "max", "slope"]
SESSION_BASELINE_FEATURES = ["attention"]

LOPO_REPORT_FILE = REPORTS_DIR / "lopo_report.txt"
LOPO_CONFUSION_MATRIX_IMAGE = REPORTS_DIR / "lopo_confusion_matrix.png"
MIN_PERSONS_FOR_LOPO = 3

for _d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, MODELS_DIR, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
    "timestampMs", "poorSignal", "eegRawValue", "attention", "meditation",
    "blinkStrength", "delta", "theta", "alphaLow", "alphaHigh",
    "betaLow", "betaHigh", "gammaLow", "gammaMid", "tagEvent", "location",
]

EEG_CORE_COLUMNS = [
    "attention", "meditation", "delta", "theta",
    "alphaLow", "alphaHigh", "betaLow", "betaHigh",
]

POOR_SIGNAL_THRESHOLD = 50  # rows with poorSignal above this are dropped

MIN_BREAK_MIN = 5
MAX_BREAK_MIN = 20
BASE_TOMORROW_MIN = 40

THRESHOLD_FOCUS_ATTENTION = 70
THRESHOLD_DISTRACTION_ATTENTION = 50
