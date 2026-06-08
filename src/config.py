from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MATCH_DATA_PATH = PROCESSED_DATA_DIR / "matches_clean.csv"
PLAYER_DATA_PATH = PROCESSED_DATA_DIR / "player_stats.csv"
MODEL_PATH = MODEL_DIR / "match_predictor.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42

RESULT_LABELS = {
    -1: "Away Win",
    0: "Draw",
    1: "Home Win",
    2: "Home Win",
}

MODEL_FEATURES = [
    "home_goals_form",
    "away_goals_form",
    "home_conceded_form",
    "away_conceded_form",
    "home_shots_form",
    "away_shots_form",
    "home_sot_form",
    "away_sot_form",
    "home_gd_form",
    "away_gd_form",
    "home_points_form5",
    "away_points_form5",
    "home_points_form10",
    "away_points_form10",
    "home_win_rate",
    "away_win_rate",
    "home_cs_rate",
    "away_cs_rate",
    "home_fts_rate",
    "away_fts_rate",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_streak",
    "away_streak",
    "home_attack_strength",
    "away_attack_strength",
    "home_defence_strength",
    "away_defence_strength",
    "season_stage",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_total",
    "h2h_home_rate",
]