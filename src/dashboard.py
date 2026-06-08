import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.prediction import predict_match
from src.utils import format_season, sorted_seasons 


try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
except Exception:
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    cosine_similarity = None
    MinMaxScaler = None
    StandardScaler = None


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLING
# =============================================================================
st.markdown(
    """
<style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] {
        background-color: #f5f5f7 !important;
        border-right: 1px solid #e0e0e5 !important;
    }
    [data-testid="stSidebar"] * { color: #1d1d1f !important; }
    h1 {
        font-size: 48px !important;
        font-weight: 700 !important;
        color: #1d1d1f !important;
        letter-spacing: -1.5px !important;
        line-height: 1.05 !important;
    }
    h2 {
        font-size: 26px !important;
        font-weight: 600 !important;
        color: #1d1d1f !important;
        letter-spacing: -0.5px !important;
    }
    h3 {
        font-size: 19px !important;
        font-weight: 600 !important;
        color: #1d1d1f !important;
        letter-spacing: -0.3px !important;
    }
    [data-testid="metric-container"] {
        background: #f5f5f7 !important;
        border-radius: 18px !important;
        padding: 24px !important;
        border: none !important;
        transition: transform 0.2s ease !important;
    }
    [data-testid="metric-container"]:hover { transform: scale(1.02) !important; }
    [data-testid="metric-container"] label {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #6e6e73 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 34px !important;
        font-weight: 700 !important;
        color: #1d1d1f !important;
        letter-spacing: -1px !important;
    }
    .stButton > button {
        background-color: #0071e3 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 980px !important;
        padding: 12px 28px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #0077ed !important;
        transform: scale(1.02) !important;
        box-shadow: 0 4px 20px rgba(0,113,227,0.3) !important;
    }
    [data-testid="stSelectbox"] > div > div {
        background-color: #f5f5f7 !important;
        border: 1px solid #d2d2d7 !important;
        border-radius: 12px !important;
        font-size: 15px !important;
    }
    [data-testid="stTextInput"] > div > div > input {
        background-color: #f5f5f7 !important;
        border: 1px solid #d2d2d7 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
    }
    hr {
        border: none !important;
        border-top: 1px solid #e0e0e5 !important;
        margin: 36px 0 !important;
    }
    [data-testid="stDataFrame"] {
        border-radius: 18px !important;
        overflow: hidden !important;
        border: 1px solid #e0e0e5 !important;
    }
    .main .block-container {
        padding: 48px 60px !important;
        max-width: 1280px !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .insight-card {
        background: linear-gradient(135deg, #0071e3 0%, #0051a0 100%);
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 8px;
        color: white;
    }
    .insight-card p {
        color: white !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    .insight-card span { font-size: 22px; }
    .small-muted {
        color: #6e6e73;
        font-size: 14px;
        line-height: 1.5;
    }
    @media (max-width: 768px) {
        .main .block-container { padding: 24px 16px !important; }
        h1 { font-size: 32px !important; }
    }
</style>
""",
    unsafe_allow_html=True,
)


BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Helvetica Neue'", color="#1d1d1f"),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=12, color="#6e6e73")),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f5", zeroline=False, tickfont=dict(size=12, color="#6e6e73")),
    margin=dict(l=0, r=0, t=24, b=0),
    hoverlabel=dict(bgcolor="white", bordercolor="#e0e0e5", font=dict(size=13, color="#1d1d1f")),
)

COLORS = [
    "#0071e3",
    "#34c759",
    "#ff9f0a",
    "#ff3b30",
    "#bf5af2",
    "#5ac8fa",
    "#ffcc00",
    "#ff6b35",
    "#32ade6",
    "#30b0c7",
]

MATCH_PATH = Path("data/processed/matches_clean.csv")
PLAYER_PATH = Path("data/processed/player_stats.csv")
MODEL_PATH = Path("models/match_predictor.pkl")
METRICS_PATH = Path("models/metrics.json")

PAGES = [
    "🏠  Home",
    "⚽  Overview",
    "🔮  Match Predictor",
    "👤  Player Stats",
    "⚔️  Player Comparison",
    "🏟️  Team Analysis",
    "💰  Transfer Analysis",
    "📈  Model Performance",
]

BASIC_MODEL_FEATURES = [
    "home_form",
    "away_form",
    "home_conceded_form",
    "away_conceded_form",
    "home_shots_form",
    "away_shots_form",
]

MODEL_FEATURES = [
    "home_goals_form", "away_goals_form",
    "home_conceded_form", "away_conceded_form",
    "home_shots_form", "away_shots_form",
    "home_sot_form", "away_sot_form",
    "home_gd_form", "away_gd_form",
    "home_points_form5", "away_points_form5",
    "home_points_form10", "away_points_form10",
    "home_win_rate", "away_win_rate",
    "home_cs_rate", "away_cs_rate",
    "home_fts_rate", "away_fts_rate",
    "home_elo", "away_elo", "elo_diff",
    "home_streak", "away_streak",
    "home_attack_strength", "away_attack_strength",
    "home_defence_strength", "away_defence_strength",
    "season_stage",
    "h2h_home_wins", "h2h_away_wins",
    "h2h_draws", "h2h_total", "h2h_home_rate",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = (
        frame.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return frame


def apply_column_aliases(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    frame = frame.copy()
    for target, possible_names in aliases.items():
        if target in frame.columns:
            continue
        for name in possible_names:
            if name in frame.columns:
                frame = frame.rename(columns={name: target})
                break
    return frame


def ensure_columns(frame: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    frame = frame.copy()
    for col, default in defaults.items():
        if col not in frame.columns:
            frame[col] = default
    return frame


def numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return frame


def safe_mean(series: pd.Series, fallback: float = 0.0) -> float:
    value = pd.to_numeric(series, errors="coerce").mean()
    if pd.isna(value):
        return float(fallback)
    return float(value)


def insight_card(emoji: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <span>{emoji}</span>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "0%"
    return f"{value:.0%}"


def result_label_from_scores(gf: float, ga: float) -> str:
    if gf > ga:
        return "✅ Win"
    if gf == ga:
        return "🟡 Draw"
    return "❌ Loss"


def result_code_from_scores(home_goals: float, away_goals: float) -> int:
    if home_goals > away_goals:
        return 1
    if home_goals == away_goals:
        return 0
    return -1


def feature_value(team_frame: pd.DataFrame, column: str, fallback: float) -> float:
    if column in team_frame.columns:
        return safe_mean(team_frame[column].tail(5), fallback)
    return fallback


def get_expected_model_features(fitted_model) -> list[str]:
    """Return the feature names the saved model was trained with.

    This prevents prediction crashes when the dashboard feature list changes
    but the saved pickle was trained on a smaller or older feature set.
    """
    if fitted_model is None:
        return MODEL_FEATURES

    if hasattr(fitted_model, "feature_names_in_"):
        return [str(feature) for feature in fitted_model.feature_names_in_]

    try:
        booster = fitted_model.get_booster()
        if booster.feature_names:
            return [str(feature) for feature in booster.feature_names]
    except Exception:
        pass

    try:
        estimator = fitted_model.named_steps.get("model")
        if hasattr(estimator, "feature_names_in_"):
            return [str(feature) for feature in estimator.feature_names_in_]
    except Exception:
        pass

    # Some pickled models only remember the number of input features, not names.
    # In that situation, avoid passing 35 features to an old 6-feature model.
    try:
        n_features = int(getattr(fitted_model, "n_features_in_"))
        if n_features == len(BASIC_MODEL_FEATURES):
            return BASIC_MODEL_FEATURES
        if n_features == len(MODEL_FEATURES):
            return MODEL_FEATURES
        if 0 < n_features <= len(MODEL_FEATURES):
            return MODEL_FEATURES[:n_features]
    except Exception:
        pass

    return MODEL_FEATURES


def get_model_classes(fitted_model, probability_count: int | None = None) -> list:
    """Return model classes safely, including models wrapped inside pipelines."""
    if fitted_model is not None and hasattr(fitted_model, "classes_"):
        return list(getattr(fitted_model, "classes_"))

    try:
        estimator = fitted_model.named_steps.get("model")
        if hasattr(estimator, "classes_"):
            return list(getattr(estimator, "classes_"))
    except Exception:
        pass

    if probability_count == 3:
        return [0, 1, 2]
    return [-1, 0, 1]


def class_name(value, classes: list | None = None) -> str:
    """Convert model class labels into readable football outcomes.

    Handles both common encodings:
    - [-1, 0, 1] means Away Win, Draw, Home Win
    - [0, 1, 2] means Away Win, Draw, Home Win
    """
    try:
        class_values = [int(v) for v in classes] if classes is not None else []
    except Exception:
        class_values = []

    try:
        numeric_value = int(value)
    except Exception:
        numeric_value = None

    if set(class_values) == {0, 1, 2}:
        encoded_mapping = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        if numeric_value in encoded_mapping:
            return encoded_mapping[numeric_value]

    if set(class_values) == {-1, 0, 1}:
        signed_mapping = {-1: "Away Win", 0: "Draw", 1: "Home Win"}
        if numeric_value in signed_mapping:
            return signed_mapping[numeric_value]

    fallback_mapping = {
        -1: "Away Win",
        0: "Draw",
        1: "Home Win",
        2: "Home Win",
        "-1": "Away Win",
        "0": "Draw",
        "1": "Home Win",
        "2": "Home Win",
        "A": "Away Win",
        "D": "Draw",
        "H": "Home Win",
        "away": "Away Win",
        "draw": "Draw",
        "home": "Home Win",
        "Away Win": "Away Win",
        "Draw": "Draw",
        "Home Win": "Home Win",
    }
    return fallback_mapping.get(value, str(value))


def team_recent_stats(frame: pd.DataFrame, team: str, venue: str | None = None, n: int = 5) -> dict:
    """Build recent form stats from raw match rows for one team."""
    team = str(team)

    if venue == "home":
        recent = frame[frame["home_team"].astype(str) == team].tail(n).copy()
        gf = recent["home_goals"] if "home_goals" in recent.columns else pd.Series(dtype=float)
        ga = recent["away_goals"] if "away_goals" in recent.columns else pd.Series(dtype=float)
        shots = recent["home_shots"] if "home_shots" in recent.columns else recent.get("home_shots_form", pd.Series(dtype=float))
        sot = recent["home_shots_on_target"] if "home_shots_on_target" in recent.columns else recent.get("home_sot_form", pd.Series(dtype=float))
    elif venue == "away":
        recent = frame[frame["away_team"].astype(str) == team].tail(n).copy()
        gf = recent["away_goals"] if "away_goals" in recent.columns else pd.Series(dtype=float)
        ga = recent["home_goals"] if "home_goals" in recent.columns else pd.Series(dtype=float)
        shots = recent["away_shots"] if "away_shots" in recent.columns else recent.get("away_shots_form", pd.Series(dtype=float))
        sot = recent["away_shots_on_target"] if "away_shots_on_target" in recent.columns else recent.get("away_sot_form", pd.Series(dtype=float))
    else:
        home = frame[frame["home_team"].astype(str) == team].copy()
        away = frame[frame["away_team"].astype(str) == team].copy()

        home["gf"] = home["home_goals"]
        home["ga"] = home["away_goals"]
        away["gf"] = away["away_goals"]
        away["ga"] = away["home_goals"]

        recent = pd.concat([home, away], ignore_index=True)
        if "date" in recent.columns:
            recent = recent.sort_values("date", na_position="last")
        recent = recent.tail(n)
        gf = recent["gf"] if "gf" in recent.columns else pd.Series(dtype=float)
        ga = recent["ga"] if "ga" in recent.columns else pd.Series(dtype=float)
        shots = pd.Series(dtype=float)
        sot = pd.Series(dtype=float)

    gf = pd.to_numeric(gf, errors="coerce").fillna(0)
    ga = pd.to_numeric(ga, errors="coerce").fillna(0)

    wins = (gf > ga).astype(int)
    draws = (gf == ga).astype(int)
    points = wins * 3 + draws
    result_sign = np.where(gf > ga, 1, np.where(gf == ga, 0, -1))

    return {
        "goals": float(gf.mean()) if len(gf) else 0.0,
        "conceded": float(ga.mean()) if len(ga) else 0.0,
        "gd": float((gf - ga).mean()) if len(gf) else 0.0,
        "points": float(points.sum()) if len(points) else 0.0,
        "win_rate": float(wins.mean()) if len(wins) else 0.0,
        "cs_rate": float((ga == 0).mean()) if len(ga) else 0.0,
        "fts_rate": float((gf == 0).mean()) if len(gf) else 0.0,
        "shots": safe_mean(shots, 0.0),
        "sot": safe_mean(sot, 0.0),
        "streak": float(np.sum(result_sign)) if len(result_sign) else 0.0,
    }


def build_prediction_features(frame: pd.DataFrame, fitted_model, home_team: str, away_team: str) -> tuple[pd.DataFrame, list[str]]:
    """Create the exact model input row while correctly separating home and away form."""
    expected_features = get_expected_model_features(fitted_model)

    home_home_5 = team_recent_stats(frame, home_team, venue="home", n=5)
    away_away_5 = team_recent_stats(frame, away_team, venue="away", n=5)
    home_all_5 = team_recent_stats(frame, home_team, venue=None, n=5)
    away_all_5 = team_recent_stats(frame, away_team, venue=None, n=5)
    home_all_10 = team_recent_stats(frame, home_team, venue=None, n=10)
    away_all_10 = team_recent_stats(frame, away_team, venue=None, n=10)

    league_avg_goals = safe_mean(frame["total_goals"], 2.5) / 2 if "total_goals" in frame.columns else 1.25
    league_avg_conceded = league_avg_goals

    pair_matches = frame[
        ((frame["home_team"].astype(str) == str(home_team)) & (frame["away_team"].astype(str) == str(away_team)))
        | ((frame["home_team"].astype(str) == str(away_team)) & (frame["away_team"].astype(str) == str(home_team)))
    ].copy()

    h2h_total = len(pair_matches)
    h2h_home_wins = 0
    h2h_away_wins = 0
    h2h_draws = 0
    for _, row in pair_matches.iterrows():
        hg = row["home_goals"]
        ag = row["away_goals"]
        if hg == ag:
            h2h_draws += 1
        elif str(row["home_team"]) == str(home_team) and hg > ag:
            h2h_home_wins += 1
        elif str(row["away_team"]) == str(home_team) and ag > hg:
            h2h_home_wins += 1
        else:
            h2h_away_wins += 1

    home_latest = frame[(frame["home_team"].astype(str) == str(home_team)) | (frame["away_team"].astype(str) == str(home_team))].tail(1)
    away_latest = frame[(frame["home_team"].astype(str) == str(away_team)) | (frame["away_team"].astype(str) == str(away_team))].tail(1)

    def latest_value(feature: str, fallback: float = 0.0) -> float:
        if feature in home_latest.columns and len(home_latest):
            return safe_mean(home_latest[feature], fallback)
        if feature in away_latest.columns and len(away_latest):
            return safe_mean(away_latest[feature], fallback)
        if feature in frame.columns:
            return safe_mean(frame[feature].tail(10), fallback)
        return fallback

    home_elo = latest_value("home_elo", 1500.0)
    away_elo = latest_value("away_elo", 1500.0)

    computed = {
        "home_form": home_home_5["goals"],
        "away_form": away_away_5["goals"],
        "home_goals_form": home_home_5["goals"],
        "away_goals_form": away_away_5["goals"],
        "home_conceded_form": home_home_5["conceded"],
        "away_conceded_form": away_away_5["conceded"],
        "home_shots_form": home_home_5["shots"],
        "away_shots_form": away_away_5["shots"],
        "home_sot_form": home_home_5["sot"],
        "away_sot_form": away_away_5["sot"],
        "home_gd_form": home_all_5["gd"],
        "away_gd_form": away_all_5["gd"],
        "home_points_form5": home_all_5["points"],
        "away_points_form5": away_all_5["points"],
        "home_points_form10": home_all_10["points"],
        "away_points_form10": away_all_10["points"],
        "home_win_rate": home_all_5["win_rate"],
        "away_win_rate": away_all_5["win_rate"],
        "home_cs_rate": home_all_5["cs_rate"],
        "away_cs_rate": away_all_5["cs_rate"],
        "home_fts_rate": home_all_5["fts_rate"],
        "away_fts_rate": away_all_5["fts_rate"],
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_streak": home_all_5["streak"],
        "away_streak": away_all_5["streak"],
        "home_attack_strength": home_all_5["goals"] / league_avg_goals if league_avg_goals else 0.0,
        "away_attack_strength": away_all_5["goals"] / league_avg_goals if league_avg_goals else 0.0,
        "home_defence_strength": home_all_5["conceded"] / league_avg_conceded if league_avg_conceded else 0.0,
        "away_defence_strength": away_all_5["conceded"] / league_avg_conceded if league_avg_conceded else 0.0,
        "season_stage": latest_value("season_stage", 0.5),
        "h2h_home_wins": float(h2h_home_wins),
        "h2h_away_wins": float(h2h_away_wins),
        "h2h_draws": float(h2h_draws),
        "h2h_total": float(h2h_total),
        "h2h_home_rate": float(h2h_home_wins / h2h_total) if h2h_total else 0.0,
    }

    feature_row = {}
    for feature in expected_features:
        if feature in computed:
            feature_row[feature] = computed[feature]
        elif feature in frame.columns:
            feature_row[feature] = safe_mean(frame[feature].tail(10), 0.0)
        else:
            feature_row[feature] = 0.0

    return pd.DataFrame([feature_row], columns=expected_features), expected_features

def stop_if_empty(frame: pd.DataFrame, message: str) -> None:
    if frame is None or len(frame) == 0:
        st.warning(message)
        st.stop()


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_match_data() -> pd.DataFrame:
    if not MATCH_PATH.exists():
        st.error(f"Match data not found at {MATCH_PATH}")
        st.stop()

    frame = pd.read_csv(MATCH_PATH)
    frame = normalise_columns(frame)
    frame = apply_column_aliases(
        frame,
        {
            "home_team": ["hometeam", "home", "home_side"],
            "away_team": ["awayteam", "away", "away_side"],
            "home_goals": ["fthg", "home_score", "hg"],
            "away_goals": ["ftag", "away_score", "ag"],
            "league": ["division", "div", "competition"],
            "season": ["year", "season_name"],
            "date": ["match_date"],
        },
    )

    required = {
        "home_team": "Unknown Home",
        "away_team": "Unknown Away",
        "home_goals": 0,
        "away_goals": 0,
        "league": "Unknown League",
        "season": "Unknown Season",
    }
    frame = ensure_columns(frame, required)
    frame = numeric_columns(frame, ["home_goals", "away_goals"])

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date", na_position="last")
    else:
        frame["date"] = pd.NaT

    if "result" not in frame.columns:
        frame["result"] = frame.apply(lambda row: result_code_from_scores(row["home_goals"], row["away_goals"]), axis=1)
    else:
        frame["result"] = pd.to_numeric(frame["result"], errors="coerce").fillna(0).astype(int)

    frame["total_goals"] = frame["home_goals"] + frame["away_goals"]

    fallback_features = {
        "home_form": frame["home_goals"].mean(),
        "away_form": frame["away_goals"].mean(),
        "home_conceded_form": frame["away_goals"].mean(),
        "away_conceded_form": frame["home_goals"].mean(),
        "home_shots_form": 0.0,
        "away_shots_form": 0.0,
        "home_goals_form": frame["home_goals"].mean(),
        "away_goals_form": frame["away_goals"].mean(),
    }
    for feature in MODEL_FEATURES:
        fallback_features.setdefault(feature, 0.0)

    frame = ensure_columns(frame, fallback_features)
    frame = numeric_columns(frame, BASIC_MODEL_FEATURES + MODEL_FEATURES)

    return frame


@st.cache_data
def load_player_data() -> pd.DataFrame | None:
    if not PLAYER_PATH.exists():
        st.warning(f"Player data not found at {PLAYER_PATH}. Player pages will be limited.")
        return None

    try:
        frame = pd.read_csv(PLAYER_PATH)
    except pd.errors.EmptyDataError:
        st.error("Player data file is empty.")
        return None
    except Exception as exc:
        st.error(f"Unexpected error loading player data: {exc}")
        return None

    frame = normalise_columns(frame)
    frame = apply_column_aliases(
        frame,
        {
            "name": ["player", "player_name", "fullname", "full_name"],
            "team": ["club", "squad", "current_team"],
            "position": ["pos", "player_position"],
            "season": ["year", "season_name"],
            "nationality": ["nation", "country"],
            "appearances": ["apps", "matches", "games"],
            "minutes": ["mins", "minutes_played"],
            "shots_total": ["shots", "total_shots"],
            "shots_on_target": ["sot", "shots_target"],
            "pass_accuracy": ["passing_accuracy", "passes_accuracy"],
            "yellow_cards": ["yellow", "cards_yellow"],
            "red_cards": ["red", "cards_red"],
            "duels_won": ["duels", "duels_won_total"],
        },
    )

    frame = ensure_columns(
        frame,
        {
            "name": "Unknown Player",
            "team": "Unknown Team",
            "position": "Unknown Position",
            "season": "Unknown Season",
            "nationality": "Unknown",
            "age": 0,
            "goals": 0,
            "assists": 0,
            "minutes": 0,
            "appearances": 0,
            "shots_total": 0,
            "shots_on_target": 0,
            "pass_accuracy": 0,
            "dribbles": 0,
            "tackles": 0,
            "interceptions": 0,
            "rating": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "duels_won": 0,
        },
    )

    numeric = [
        "age",
        "goals",
        "assists",
        "minutes",
        "appearances",
        "shots_total",
        "shots_on_target",
        "pass_accuracy",
        "dribbles",
        "tackles",
        "interceptions",
        "rating",
        "yellow_cards",
        "red_cards",
        "duels_won",
    ]
    frame = numeric_columns(frame, numeric)

    minutes_safe = frame["minutes"].replace(0, np.nan)
    frame["goals_p90"] = (frame["goals"] / minutes_safe * 90).round(2)
    frame["assists_p90"] = (frame["assists"] / minutes_safe * 90).round(2)
    frame["shots_p90"] = (frame["shots_total"] / minutes_safe * 90).round(2)
    frame["sot_p90"] = (frame["shots_on_target"] / minutes_safe * 90).round(2)
    frame["tackles_p90"] = (frame["tackles"] / minutes_safe * 90).round(2)
    frame["interc_p90"] = (frame["interceptions"] / minutes_safe * 90).round(2)
    frame["contrib_p90"] = ((frame["goals"] + frame["assists"]) / minutes_safe * 90).round(2)
    frame["cards_p90"] = ((frame["yellow_cards"] + frame["red_cards"]) / minutes_safe * 90).round(2)
    frame = frame.fillna(0)

    return frame


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.warning(f"Model could not be loaded: {exc}")
        return None


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


matches = load_match_data()
players = load_player_data()
model = load_model()
metrics = load_metrics()

LEAGUES = ["All"] + sorted(matches["league"].dropna().astype(str).unique().tolist())


def filter_matches_by_league(league: str) -> pd.DataFrame:
    if league == "All":
        return matches.copy()
    return matches[matches["league"].astype(str) == str(league)].copy()


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown(
    """
<div style='padding:28px 8px 20px 8px;'>
    <p style='font-size:11px;font-weight:600;color:#6e6e73;letter-spacing:1.2px;text-transform:uppercase;margin:0;'>Sports</p>
    <p style='font-size:26px;font-weight:700;color:#1d1d1f;letter-spacing:-0.8px;margin:4px 0 0 0;'>Analytics</p>
</div>
""",
    unsafe_allow_html=True,
)

default_page = st.session_state.pop("target_page", PAGES[0])
if default_page not in PAGES:
    default_page = PAGES[0]

page = st.sidebar.radio(
    "",
    PAGES,
    index=PAGES.index(default_page),
)

st.sidebar.markdown(
    f"""
<div style='padding:20px 8px 0 8px;border-top:1px solid #e0e0e5;margin-top:20px;'>
    <p style='font-size:12px;color:#6e6e73;margin:0;line-height:1.6;'>
        {len(matches):,} matches · {matches['league'].nunique()} leagues<br>{matches['season'].nunique()} seasons of data
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HOME
# =============================================================================
if page == "🏠  Home":
    st.markdown(
        """
        <div style='padding: 60px 0 40px 0;'>
            <p style='font-size: 13px; font-weight: 600; color: #0071e3; letter-spacing: 1.5px; text-transform: uppercase; margin: 0 0 16px 0;'>
                Football Intelligence Platform
            </p>
            <h1 style='font-size: 64px; font-weight: 700; color: #1d1d1f; letter-spacing: -2px; line-height: 1.02; margin: 0 0 24px 0;'>
                Analyse. Predict.<br>Scout. Decide.
            </h1>
            <p style='font-size: 21px; color: #6e6e73; font-weight: 400; line-height: 1.5; max-width: 680px; margin: 0 0 40px 0;'>
                A machine learning powered football analytics dashboard for match prediction, player comparison, team analysis and recruitment insights across English football.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔮 Try Predictor"):
            st.session_state["target_page"] = "🔮  Match Predictor"
            st.rerun()
    with col2:
        if st.button("👤 View Players"):
            st.session_state["target_page"] = "👤  Player Stats"
            st.rerun()

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches Analysed", f"{len(matches):,}")
    c2.metric("Leagues Covered", matches["league"].nunique())
    c3.metric("Seasons of Data", matches["season"].nunique())
    c4.metric("Player Records", f"{len(players):,}" if players is not None else "Not loaded")

    st.divider()

    st.markdown("## Core Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div style='background:#f5f5f7;border-radius:20px;padding:28px;height:250px;'>
            <div style='font-size:36px;margin-bottom:14px;'>🔮</div>
            <h3>Match Predictor</h3>
            <p class='small-muted'>Predict Home Win, Draw or Away Win using model probabilities and recent form based explanations.</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div style='background:#f5f5f7;border-radius:20px;padding:28px;height:250px;'>
            <div style='font-size:36px;margin-bottom:14px;'>⚔️</div>
            <h3>Player Comparison</h3>
            <p class='small-muted'>Compare players with percentile radar charts, per 90 statistics and similarity matching.</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div style='background:#f5f5f7;border-radius:20px;padding:28px;height:250px;'>
            <div style='font-size:36px;margin-bottom:14px;'>💰</div>
            <h3>Recruitment Scouting</h3>
            <p class='small-muted'>Build shortlists, find hidden gems and connect team weaknesses to player recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("## How to use it")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("1. Open **Match Predictor** and select two teams.")
        st.markdown("2. Open **Player Stats** to filter players by position, team, season and minutes.")
        st.markdown("3. Open **Player Comparison** to compare two players using raw or per 90 metrics.")
    with col2:
        st.markdown("4. Open **Team Analysis** to check team form, goals, clean sheets and last five results.")
        st.markdown("5. Open **Transfer Analysis** to build scouting shortlists.")
        st.markdown("6. Open **Model Performance** to review accuracy, confusion matrix and feature importance.")

    st.divider()
    st.caption("Built by Daniel Olutade · Python · Streamlit · Pandas · Plotly · Scikit Learn · XGBoost")


# =============================================================================
# OVERVIEW
# =============================================================================
elif page == "⚽  Overview":
    st.markdown("# Football Analytics")
    st.markdown("Thirty years of English football. Five leagues. One dashboard.")
    league = st.selectbox("League", LEAGUES, key="overview_league")
    frame = filter_matches_by_league(league)
    stop_if_empty(frame, "No matches found for this league.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches", f"{len(frame):,}")
    c2.metric("Leagues", frame["league"].nunique())
    c3.metric("Teams", len(set(frame["home_team"]) | set(frame["away_team"])))
    c4.metric("Seasons", frame["season"].nunique())
    st.divider()

    home_pct = (frame["result"] == 1).mean()
    avg_goals = frame["total_goals"].mean()
    top_home_team = frame.groupby("home_team")["home_goals"].mean().idxmax()

    insight_card("🏠", f"Home teams win <b>{home_pct:.0%}</b> of all matches in this selection.")
    col1, col2 = st.columns(2)
    with col1:
        insight_card("⚽", f"<b>{top_home_team}</b> average the most home goals per match.")
    with col2:
        insight_card("📊", f"Average of <b>{avg_goals:.2f}</b> goals per match.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Match Outcomes")
        result_counts = frame["result"].map({1: "Home Win", 0: "Draw", -1: "Away Win"}).value_counts()
        fig = go.Figure(
            go.Pie(
                values=result_counts.values,
                labels=result_counts.index,
                hole=0.6,
                marker=dict(colors=["#0071e3", "#ff9f0a", "#ff3b30"], line=dict(color="#ffffff", width=2)),
            )
        )
        fig.update_layout(**BASE_LAYOUT, showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Matches Per League")
        league_counts = frame["league"].value_counts()
        fig = go.Figure(
            go.Bar(
                x=league_counts.index,
                y=league_counts.values,
                marker=dict(color=COLORS[: len(league_counts)], line=dict(width=0)),
                text=league_counts.values,
                textposition="outside",
            )
        )
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("League Entertainment Profile")

    frame = frame.copy()

    if "total_goals" not in frame.columns:
        frame["total_goals"] = frame["home_goals"] + frame["away_goals"]

    frame["over_2_5_goals"] = frame["total_goals"] > 2.5
    frame["both_teams_scored"] = (frame["home_goals"] > 0) & (frame["away_goals"] > 0)
    frame["home_win"] = frame["result"] == 1
    frame["draw"] = frame["result"] == 0
    frame["away_win"] = frame["result"] == -1

    league_profile = (
        frame.groupby("league")
        .agg(
            matches=("league", "count"),
            avg_goals=("total_goals", "mean"),
            home_win_rate=("home_win", "mean"),
            draw_rate=("draw", "mean"),
            away_win_rate=("away_win", "mean"),
            over_2_5_rate=("over_2_5_goals", "mean"),
            btts_rate=("both_teams_scored", "mean"),
        )
        .reset_index()
    )

    league_profile["entertainment_score"] = (
        league_profile["avg_goals"] * 30
        + league_profile["over_2_5_rate"] * 40
        + league_profile["btts_rate"] * 30
    ).round(1)

    league_profile = league_profile.sort_values("entertainment_score", ascending=False)

    top_league = league_profile.iloc[0]

    insight_card(
        "🔥",
        f"<b>{top_league['league']}</b> is the most entertaining league in the dataset, "
        f"averaging <b>{top_league['avg_goals']:.2f}</b> goals per match with "
        f"<b>{top_league['over_2_5_rate']:.0%}</b> of games going over 2.5 goals."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=league_profile["league"],
                y=league_profile["avg_goals"],
                name="Average Goals",
                marker=dict(color="#0071e3", line=dict(width=0)),
                text=league_profile["avg_goals"].round(2),
                textposition="outside",
                hovertemplate="%{x}<br>Average goals: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            **BASE_LAYOUT,
            height=420,
            yaxis_title="Goals per Match",
            xaxis_title="League",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Key Stats")

        best_goals = league_profile.loc[league_profile["avg_goals"].idxmax()]
        most_draws = league_profile.loc[league_profile["draw_rate"].idxmax()]
        most_btts = league_profile.loc[league_profile["btts_rate"].idxmax()]

        st.metric("Highest Scoring", best_goals["league"], f"{best_goals['avg_goals']:.2f} goals")
        st.metric("Most Draw Heavy", most_draws["league"], f"{most_draws['draw_rate']:.0%} draws")
        st.metric("Most BTTS", most_btts["league"], f"{most_btts['btts_rate']:.0%} BTTS")

    st.markdown("### League Breakdown")

    display_profile = league_profile.copy()

    display_profile["avg_goals"] = display_profile["avg_goals"].round(2)
    display_profile["home_win_rate"] = (display_profile["home_win_rate"] * 100).round(1)
    display_profile["draw_rate"] = (display_profile["draw_rate"] * 100).round(1)
    display_profile["away_win_rate"] = (display_profile["away_win_rate"] * 100).round(1)
    display_profile["over_2_5_rate"] = (display_profile["over_2_5_rate"] * 100).round(1)
    display_profile["btts_rate"] = (display_profile["btts_rate"] * 100).round(1)

    display_profile = display_profile.rename(
        columns={
            "league": "League",
            "matches": "Matches",
            "avg_goals": "Avg Goals",
            "home_win_rate": "Home Win %",
            "draw_rate": "Draw %",
            "away_win_rate": "Away Win %",
            "over_2_5_rate": "Over 2.5 %",
            "btts_rate": "BTTS %",
            "entertainment_score": "Entertainment Score",
        }
    )

    st.dataframe(
        display_profile[
            [
                "League",
                "Matches",
                "Avg Goals",
                "Home Win %",
                "Draw %",
                "Away Win %",
                "Over 2.5 %",
                "BTTS %",
                "Entertainment Score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# MATCH PREDICTOR
# =============================================================================
elif page == "🔮  Match Predictor":
    st.markdown("# Match Predictor")
    st.markdown("Select two teams to forecast the outcome.")
    st.divider()

    if model is None:
        st.warning("Model file not found. Run python scripts/train_model.py first.")
        st.stop()

    teams = sorted(
        set(matches["home_team"].astype(str))
        | set(matches["away_team"].astype(str))
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏠 Home Team**")
        home_team = st.selectbox(
            "Home Team",
            teams,
            index=0,
            label_visibility="collapsed",
            key="predictor_home_team",
        )

    with col2:
        st.markdown("**✈️ Away Team**")
        away_team = st.selectbox(
            "Away Team",
            teams,
            index=min(1, len(teams) - 1),
            label_visibility="collapsed",
            key="predictor_away_team",
        )

    if home_team == away_team:
        st.warning("Choose two different teams.")
        st.stop()

    if st.button("Predict Match →", key="predict_match_button"):
        try:
            prediction = predict_match(model, matches, home_team, away_team, metrics)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        outcome = prediction["prediction"]
        confidence = prediction["confidence"]
        probabilities = prediction["probabilities"]
        feature_row = prediction.get("explanation_features", prediction["features"]).iloc[0]

        def feature(name, default=0.0):
            return float(feature_row[name]) if name in feature_row.index else default

        if confidence >= 0.60:
            confidence_label = "High"
        elif confidence >= 0.45:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"

        st.divider()

        insight_card(
            "🔮",
            f"Prediction: <b>{outcome}</b> with <b>{confidence_label}</b> confidence ({confidence:.0%}).",
        )

        st.subheader("Outcome Probabilities")

        home_probability = probabilities.loc[
            probabilities["Outcome"] == "Home Win",
            "Probability",
        ].sum()

        draw_probability = probabilities.loc[
            probabilities["Outcome"] == "Draw",
            "Probability",
        ].sum()

        away_probability = probabilities.loc[
            probabilities["Outcome"] == "Away Win",
            "Probability",
        ].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("🏠 Home Win", f"{home_probability:.0%}")
        c2.metric("🤝 Draw", f"{draw_probability:.0%}")
        c3.metric("✈️ Away Win", f"{away_probability:.0%}")

        fig = go.Figure(
            go.Bar(
                x=probabilities["Outcome"],
                y=probabilities["Probability"],
                marker=dict(
                    color=["#0071e3", "#ff9f0a", "#ff3b30"],
                    line=dict(width=0),
                ),
                text=[f"{value:.0%}" for value in probabilities["Probability"]],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            )
        )

        fig.update_layout(
            **BASE_LAYOUT,
            height=420,
            yaxis_title="Probability",
        )

        fig.update_yaxes(
            tickformat=".0%",
            range=[0, max(probabilities["Probability"].max() + 0.15, 0.6)],
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Why this prediction?")

        reasons = []

        elo_diff = feature("elo_diff")
        home_points_form5 = feature("home_points_form5")
        away_points_form5 = feature("away_points_form5")
        home_goals_form = feature("home_goals_form")
        away_goals_form = feature("away_goals_form")
        home_conceded_form = feature("home_conceded_form")
        away_conceded_form = feature("away_conceded_form")

        if elo_diff > 75:
            reasons.append(f"🏠 {home_team} have the stronger Elo rating advantage.")
        elif elo_diff < -75:
            reasons.append(f"✈️ {away_team} have the stronger Elo rating advantage.")
        else:
            reasons.append("⚖️ The Elo ratings are fairly close, so the model sees this as competitive.")

        if home_points_form5 > away_points_form5:
            reasons.append(f"📈 {home_team} have better recent points form.")
        elif away_points_form5 > home_points_form5:
            reasons.append(f"📈 {away_team} have better recent points form.")

        if home_goals_form > away_goals_form:
            reasons.append(f"⚽ {home_team} have stronger recent scoring form.")
        elif away_goals_form > home_goals_form:
            reasons.append(f"⚽ {away_team} have stronger recent scoring form.")

        if home_conceded_form < away_conceded_form:
            reasons.append(f"🛡️ {home_team} have conceded fewer goals recently.")
        elif away_conceded_form < home_conceded_form:
            reasons.append(f"🛡️ {away_team} have conceded fewer goals recently.")

        if confidence < 0.45:
            reasons.append("⚠️ The model confidence is low, so this should be treated as a close match.")

        for reason in reasons:
            st.markdown(reason)

        st.divider()

        st.subheader("Model Feature Snapshot")

        feature_display = pd.DataFrame(
            {
                "Feature": [
                    "Home Goals Form",
                    "Away Goals Form",
                    "Home Conceded Form",
                    "Away Conceded Form",
                    "Home Points Form 5",
                    "Away Points Form 5",
                    "Home Elo",
                    "Away Elo",
                    "Elo Difference",
                ],
                "Value": [
                    feature("home_goals_form"),
                    feature("away_goals_form"),
                    feature("home_conceded_form"),
                    feature("away_conceded_form"),
                    feature("home_points_form5"),
                    feature("away_points_form5"),
                    feature("home_elo"),
                    feature("away_elo"),
                    feature("elo_diff"),
                ],
            }
        )

        st.dataframe(
            feature_display,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Recent Matches: {home_team}")
            recent_home = matches[
                (matches["home_team"].astype(str) == home_team)
                | (matches["away_team"].astype(str) == home_team)
            ].tail(5)

            st.dataframe(
                recent_home[
                    ["date", "home_team", "away_team", "home_goals", "away_goals"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            st.subheader(f"Recent Matches: {away_team}")
            recent_away = matches[
                (matches["home_team"].astype(str) == away_team)
                | (matches["away_team"].astype(str) == away_team)
            ].tail(5)

            st.dataframe(
                recent_away[
                    ["date", "home_team", "away_team", "home_goals", "away_goals"]
                ],
                use_container_width=True,
                hide_index=True,
            )

# =============================================================================
# PLAYER STATS
# =============================================================================
elif page == "👤  Player Stats":
    st.markdown("# Player Statistics")
    st.markdown("Filter players by position, team, season and minutes.")
    st.divider()

    if players is None:
        st.error("Player data is not available. Add data/processed/player_stats.csv first.")
        st.stop()

    min_450 = players[players["minutes"] >= 450]
    top_scorer = players.loc[players["goals"].idxmax()]
    top_assister = players.loc[players["assists"].idxmax()]
    top_p90 = min_450.loc[min_450["goals_p90"].idxmax()] if len(min_450) else players.loc[players["goals_p90"].idxmax()]

    insight_card("⚽", f"<b>{top_scorer['name']}</b> leads with <b>{int(top_scorer['goals'])}</b> goals.")
    insight_card("🎯", f"<b>{top_assister['name']}</b> leads with <b>{int(top_assister['assists'])}</b> assists.")
    insight_card("⏱️", f"<b>{top_p90['name']}</b> has the best goals per 90 among players with 450+ minutes: <b>{top_p90['goals_p90']:.2f}</b>.")
    st.divider()

    st.subheader("Filters")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        positions = ["All"] + sorted(players["position"].dropna().astype(str).unique().tolist())
        pos_filter = st.selectbox("Position", positions, key="players_position")

    with col2:
        teams_list = ["All"] + sorted(players["team"].dropna().astype(str).unique().tolist())
        team_filter = st.selectbox("Team", teams_list, key="players_team")

    with col3:
        seasons_list = ["All"] + sorted(players["season"].dropna().astype(str).unique().tolist(), reverse=True)
        season_filter = st.selectbox("Season", seasons_list, key="players_season")

    with col4:
        max_minutes = int(max(players["minutes"].max(), 90))
        min_minutes = st.number_input("Min Minutes", min_value=0, max_value=max_minutes, value=0, step=90)

    with col5:
        stat_view = st.selectbox("Stat View", ["Raw Totals", "Per 90"], key="players_stat_view")

    filtered = players.copy()
    if pos_filter != "All":
        filtered = filtered[filtered["position"].astype(str) == pos_filter]
    if team_filter != "All":
        filtered = filtered[filtered["team"].astype(str) == team_filter]
    if season_filter != "All":
        filtered = filtered[filtered["season"].astype(str) == season_filter]
    filtered = filtered[filtered["minutes"] >= min_minutes]

    search = st.text_input("🔍 Search player", placeholder="Example: Salah, Haaland, Palmer")
    if search:
        filtered = filtered[filtered["name"].astype(str).str.contains(search, case=False, na=False)]

    st.markdown(f"**{len(filtered):,} players found**")
    if filtered.empty:
        st.warning("No players match these filters.")
        st.stop()

    if search and len(filtered) == 1:
        player = filtered.iloc[0]
        st.subheader(f"📋 {player['name']} · {player['team']}")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Goals", int(player["goals"]))
        p2.metric("Assists", int(player["assists"]))
        p3.metric("Goals/90", f"{player['goals_p90']:.2f}")
        p4.metric("Assists/90", f"{player['assists_p90']:.2f}")
        p5.metric("Minutes", int(player["minutes"]))
        p6.metric("Rating", f"{player['rating']:.1f}")
        st.divider()

    goal_col = "goals_p90" if stat_view == "Per 90" else "goals"
    assist_col = "assists_p90" if stat_view == "Per 90" else "assists"

    st.subheader(f"Top Scorers: {stat_view}")
    top = filtered.nlargest(10, goal_col)
    fig = go.Figure(
        go.Bar(
            x=top["name"],
            y=top[goal_col],
            marker=dict(color="#0071e3", line=dict(width=0)),
            text=top[goal_col].round(2),
            textposition="outside",
        )
    )
    fig.update_layout(**BASE_LAYOUT)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Assisters")
        st.dataframe(filtered.nlargest(10, assist_col)[["name", "team", assist_col]], use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Most Booked")
        booked = filtered.copy()
        booked["total_cards"] = booked["yellow_cards"] + booked["red_cards"]
        st.dataframe(booked.nlargest(10, "total_cards")[["name", "team", "yellow_cards", "red_cards", "total_cards"]], use_container_width=True, hide_index=True)

    st.subheader("Full Stats Table")
    if stat_view == "Per 90":
        display_cols = ["name", "team", "position", "season", "goals_p90", "assists_p90", "shots_p90", "sot_p90", "tackles_p90", "contrib_p90", "rating"]
    else:
        display_cols = ["name", "team", "position", "season", "goals", "assists", "appearances", "minutes", "yellow_cards", "red_cards", "shots_on_target", "rating"]
    st.dataframe(filtered[display_cols].sort_values(goal_col, ascending=False), use_container_width=True, hide_index=True)


# =============================================================================
# PLAYER COMPARISON
# =============================================================================
elif page == "⚔️  Player Comparison":
    st.markdown("# Player Comparison")
    st.markdown("Compare two players side by side with percentile radar charts.")
    st.divider()

    if players is None:
        st.error("Player data is not available. Add data/processed/player_stats.csv first.")
        st.stop()

    if len(players) < 2:
        st.warning("At least two players are needed for comparison.")
        st.stop()

    player_names = sorted(players["name"].dropna().astype(str).unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        first_name = st.selectbox("Player 1", player_names, index=0)
    with col2:
        second_name = st.selectbox("Player 2", player_names, index=min(1, len(player_names) - 1))

    first_player = players[players["name"].astype(str) == first_name].iloc[0]
    second_player = players[players["name"].astype(str) == second_name].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"👤 {first_name}")
        st.caption(f"{first_player['team']} · {first_player['position']} · {int(first_player['minutes'])} mins")
        a, b, c, d = st.columns(4)
        a.metric("Goals", int(first_player["goals"]))
        b.metric("Assists", int(first_player["assists"]))
        c.metric("Goals/90", f"{first_player['goals_p90']:.2f}")
        d.metric("Rating", f"{first_player['rating']:.1f}")
    with col2:
        st.subheader(f"👤 {second_name}")
        st.caption(f"{second_player['team']} · {second_player['position']} · {int(second_player['minutes'])} mins")
        a, b, c, d = st.columns(4)
        a.metric("Goals", int(second_player["goals"]))
        b.metric("Assists", int(second_player["assists"]))
        c.metric("Goals/90", f"{second_player['goals_p90']:.2f}")
        d.metric("Rating", f"{second_player['rating']:.1f}")

    st.divider()
    stat_mode = st.radio("Radar Mode", ["Raw Stats", "Per 90"], horizontal=True)
    if stat_mode == "Per 90":
        radar_cols = ["goals_p90", "assists_p90", "sot_p90", "tackles_p90", "interc_p90", "contrib_p90"]
        radar_labels = ["Goals/90", "Assists/90", "Shots on Target/90", "Tackles/90", "Interceptions/90", "Contributions/90"]
    else:
        radar_cols = ["goals", "assists", "shots_on_target", "pass_accuracy", "dribbles", "tackles"]
        radar_labels = ["Goals", "Assists", "Shots on Target", "Pass Accuracy", "Dribbles", "Tackles"]

    def percentile(player_row: pd.Series, col: str) -> float:
        return float((players[col] <= player_row[col]).mean() * 100)

    first_values = [percentile(first_player, col) for col in radar_cols]
    second_values = [percentile(second_player, col) for col in radar_cols]

    st.subheader("Radar Comparison")
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=first_values + [first_values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(0,113,227,0.15)",
            line=dict(color="#0071e3", width=2.5),
            name=first_name,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=second_values + [second_values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(255,59,48,0.15)",
            line=dict(color="#ff3b30", width=2.5),
            name=second_name,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color="#6e6e73"), gridcolor="#e0e0e5"),
            angularaxis=dict(tickfont=dict(size=12, color="#1d1d1f"), gridcolor="#e0e0e5"),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Percentile Rankings")
    rows = []
    for col, label in zip(radar_cols, radar_labels):
        first_pct = percentile(first_player, col)
        second_pct = percentile(second_player, col)
        rows.append(
            {
                "Stat": label,
                f"{first_name} Raw": round(float(first_player[col]), 2),
                f"{first_name} Percentile": f"{first_pct:.0f}th",
                f"{second_name} Raw": round(float(second_player[col]), 2),
                f"{second_name} Percentile": f"{second_pct:.0f}th",
                "Better": first_name if first_pct >= second_pct else second_name,
            }
        )
    pct_frame = pd.DataFrame(rows)
    st.dataframe(pct_frame, use_container_width=True, hide_index=True)

    first_wins = (pct_frame["Better"] == first_name).sum()
    second_wins = (pct_frame["Better"] == second_name).sum()
    if first_wins > second_wins:
        insight_card("🏆", f"<b>{first_name}</b> wins <b>{first_wins}</b> of {len(pct_frame)} categories.")
    elif second_wins > first_wins:
        insight_card("🏆", f"<b>{second_name}</b> wins <b>{second_wins}</b> of {len(pct_frame)} categories.")
    else:
        insight_card("🤝", f"These players are evenly matched across the selected categories.")

    st.divider()
    st.subheader(f"Players Similar to {first_name}")
    if StandardScaler is None or cosine_similarity is None:
        st.warning("Install scikit learn to enable similarity matching.")
    else:
        similarity_cols = ["goals", "assists", "shots_on_target", "pass_accuracy", "dribbles", "tackles"]
        scaler = StandardScaler()
        matrix = scaler.fit_transform(players[similarity_cols].fillna(0))
        player_index = players[players["name"].astype(str) == first_name].index[0]
        matrix_position = players.index.get_loc(player_index)
        similarities = cosine_similarity([matrix[matrix_position]], matrix)[0]
        similar = players.copy()
        similar["similarity"] = similarities
        similar = similar[similar["name"].astype(str) != first_name].nlargest(5, "similarity")
        similar["similarity"] = similar["similarity"].apply(lambda value: f"{value:.0%}")
        st.dataframe(similar[["name", "team", "position", "goals", "assists", "rating", "similarity"]], use_container_width=True, hide_index=True)


# =============================================================================
# TEAM ANALYSIS
# =============================================================================
elif page == "🏟️  Team Analysis":
    st.markdown("# Team Analysis")
    st.markdown("Deep dive into any team's performance.")
    st.divider()

    league = st.selectbox("League", LEAGUES, key="team_league")
    frame = filter_matches_by_league(league)
    teams = sorted(set(frame["home_team"].astype(str)) | set(frame["away_team"].astype(str)))
    team = st.selectbox("Select Team", teams, key="team_select")

    home_frame = frame[frame["home_team"].astype(str) == team].copy()
    away_frame = frame[frame["away_team"].astype(str) == team].copy()

    home_frame["gf"] = home_frame["home_goals"]
    home_frame["ga"] = home_frame["away_goals"]
    home_frame["venue"] = "Home"

    away_frame["gf"] = away_frame["away_goals"]
    away_frame["ga"] = away_frame["home_goals"]
    away_frame["venue"] = "Away"

    team_matches = pd.concat([home_frame, away_frame], ignore_index=True)
    if "date" in team_matches.columns:
        team_matches = team_matches.sort_values("date", na_position="last")

    stop_if_empty(team_matches, "No matches found for this team.")

    team_matches["result_label"] = team_matches.apply(lambda row: result_label_from_scores(row["gf"], row["ga"]), axis=1)
    team_matches["win"] = (team_matches["gf"] > team_matches["ga"]).astype(int)
    team_matches["draw"] = (team_matches["gf"] == team_matches["ga"]).astype(int)
    team_matches["loss"] = (team_matches["gf"] < team_matches["ga"]).astype(int)

    total = len(team_matches)
    wins = int(team_matches["win"].sum())
    draws = int(team_matches["draw"].sum())
    losses = int(team_matches["loss"].sum())
    goals_for = int(team_matches["gf"].sum())
    goals_against = int(team_matches["ga"].sum())

    insight_card("🏆", f"<b>{team}</b>: <b>{wins}W {draws}D {losses}L</b> from {total} matches. Win rate: <b>{wins / total:.0%}</b>.")
    insight_card("⚽", f"Scored <b>{goals_for}</b>, conceded <b>{goals_against}</b>, goal difference <b>{goals_for - goals_against:+}</b>.")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games", total)
    c2.metric("Wins", wins)
    c3.metric("Draws", draws)
    c4.metric("Goals For", goals_for)
    c5.metric("Goals Against", goals_against)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Home vs Away")
        venue_stats = team_matches.groupby("venue", as_index=False).agg(Wins=("win", "sum"), Draws=("draw", "sum"), Losses=("loss", "sum"), Goals=("gf", "mean"), Conceded=("ga", "mean"))
        fig = go.Figure()
        for i, metric in enumerate(["Wins", "Draws", "Losses", "Goals", "Conceded"]):
            fig.add_trace(go.Bar(name=metric, x=venue_stats["venue"], y=venue_stats[metric], marker=dict(color=COLORS[i], line=dict(width=0))))
        fig.update_layout(**BASE_LAYOUT, barmode="group", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Last 5 Matches")
        last_five = team_matches.tail(5)[["date", "venue", "gf", "ga", "result_label"]].copy()
        last_five = last_five.rename(columns={"date": "Date", "venue": "Venue", "gf": "GF", "ga": "GA", "result_label": "Result"})
        st.dataframe(last_five, use_container_width=True, hide_index=True)

    st.subheader("Goals Per Season")
    goals_by_season = team_matches.groupby("season", as_index=False)["gf"].sum()
    fig = go.Figure(
        go.Bar(
            x=goals_by_season["season"],
            y=goals_by_season["gf"],
            marker=dict(color="#0071e3", line=dict(width=0)),
            hovertemplate="Season %{x}<br>Goals: %{y}<extra></extra>",
        )
    )
    fig.update_layout(**BASE_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Team Strength Profile")
    clean_sheets = int((team_matches["ga"] == 0).sum())
    failed_to_score = int((team_matches["gf"] == 0).sum())
    home_win_rate = (home_frame["home_goals"] > home_frame["away_goals"]).mean() if len(home_frame) else 0
    away_win_rate = (away_frame["away_goals"] > away_frame["home_goals"]).mean() if len(away_frame) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean Sheets", f"{clean_sheets} ({clean_sheets / total:.0%})")
    c2.metric("Failed to Score", f"{failed_to_score} ({failed_to_score / total:.0%})")
    c3.metric("Home Win Rate", format_percent(home_win_rate))
    c4.metric("Away Win Rate", format_percent(away_win_rate))

    if home_win_rate > away_win_rate * 1.3 and home_win_rate > 0:
        insight_card("🏠", f"<b>{team}</b>'s home form is significantly stronger than their away form.")
    elif away_win_rate > home_win_rate * 1.3 and away_win_rate > 0:
        insight_card("✈️", f"<b>{team}</b> are surprisingly strong away from home.")
    else:
        insight_card("⚖️", f"<b>{team}</b> perform fairly consistently home and away.")


# =============================================================================
# TRANSFER ANALYSIS
# =============================================================================
elif page == "💰  Transfer Analysis":
    st.markdown("# Transfer & Scouting")
    st.markdown("Find hidden gems, analyse team weaknesses and build recruitment shortlists.")
    st.divider()

    if players is None:
        st.error("Player data is not available. Add data/processed/player_stats.csv first.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Player Scouting", "💎 Hidden Gems", "🏟️ Team Weaknesses", "📊 Attack vs Defence"])

    with tab1:
        st.subheader("Recruitment Shortlist Builder")
        st.markdown("Filter players by your exact requirements to build a ranked shortlist.")
        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            scout_positions = ["All"] + sorted(players["position"].dropna().astype(str).unique().tolist())
            scout_position = st.selectbox("Position", scout_positions, key="scout_position")
            min_age = st.number_input("Min Age", min_value=0, max_value=50, value=18, key="scout_min_age")
            max_age = st.number_input("Max Age", min_value=0, max_value=50, value=28, key="scout_max_age")
        with col2:
            min_goals_p90 = st.number_input("Min Goals/90", min_value=0.0, max_value=5.0, value=0.0, step=0.05, key="scout_min_goals")
            min_assists_p90 = st.number_input("Min Assists/90", min_value=0.0, max_value=5.0, value=0.0, step=0.05, key="scout_min_assists")
            min_rating = st.number_input("Min Rating", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="scout_min_rating")
        with col3:
            min_minutes = st.number_input("Min Minutes Played", min_value=0, max_value=int(max(players["minutes"].max(), 90)), value=90, step=90, key="scout_min_minutes")
            scout_teams = ["All"] + sorted(players["team"].dropna().astype(str).unique().tolist())
            scout_team = st.selectbox("Team", scout_teams, key="scout_team")
            scout_nations = ["All"] + sorted(players["nationality"].dropna().astype(str).unique().tolist())
            scout_nation = st.selectbox("Nationality", scout_nations, key="scout_nation")

        if st.button("Build Shortlist →", key="build_shortlist"):
            shortlist = players.copy()
            if scout_position != "All":
                shortlist = shortlist[shortlist["position"].astype(str) == scout_position]
            if scout_team != "All":
                shortlist = shortlist[shortlist["team"].astype(str) == scout_team]
            if scout_nation != "All":
                shortlist = shortlist[shortlist["nationality"].astype(str) == scout_nation]

            shortlist = shortlist[
                (shortlist["age"] >= min_age)
                & (shortlist["age"] <= max_age)
                & (shortlist["minutes"] >= min_minutes)
                & (shortlist["goals_p90"] >= min_goals_p90)
                & (shortlist["assists_p90"] >= min_assists_p90)
                & (shortlist["rating"] >= min_rating)
            ].copy()

            if shortlist.empty:
                st.warning("No players match your criteria. Relax the filters slightly.")
            else:
                score_cols = ["goals_p90", "assists_p90", "sot_p90", "rating", "contrib_p90"]
                if MinMaxScaler is not None and len(shortlist) > 1:
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(shortlist[score_cols].fillna(0))
                    shortlist["score"] = (scaled.mean(axis=1) * 100).round(1)
                else:
                    shortlist["score"] = (shortlist[score_cols].fillna(0).mean(axis=1)).round(1)

                shortlist = shortlist.sort_values("score", ascending=False)
                display_cols = ["name", "team", "age", "position", "goals_p90", "assists_p90", "rating", "minutes", "score"]
                insight_card("🎯", f"Found <b>{len(shortlist)}</b> players matching your criteria.")
                st.dataframe(shortlist[display_cols], use_container_width=True, hide_index=True)

                csv = shortlist[display_cols].to_csv(index=False)
                st.download_button("⬇️ Download Shortlist CSV", data=csv, file_name="recruitment_shortlist.csv", mime="text/csv")

                top = shortlist.iloc[0]
                insight_card("⭐", f"Top recommendation: <b>{top['name']}</b> from <b>{top['team']}</b>. Score: <b>{top['score']}</b>/100.")

    with tab2:
        st.subheader("Hidden Gems Finder")
        st.markdown("Players with strong output, young age or low minutes.")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Best Under 23s**")
            under_23 = players[(players["age"] <= 23) & (players["minutes"] >= 90)].copy()
            if under_23.empty:
                st.info("No under 23 players found with the current data.")
            else:
                under_23["score"] = (under_23["goals_p90"] + under_23["assists_p90"] + under_23["rating"] / 10).round(2)
                st.dataframe(under_23.nlargest(10, "score")[["name", "team", "age", "goals_p90", "assists_p90", "rating", "score"]], use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**High Output Low Minutes**")
            hidden = players[(players["minutes"] >= 90) & (players["minutes"] <= 900)].copy()
            if hidden.empty:
                st.info("No low minute players found with the current data.")
            else:
                hidden["score"] = (hidden["goals_p90"] + hidden["assists_p90"]).round(2)
                st.dataframe(hidden.nlargest(10, "score")[["name", "team", "minutes", "goals_p90", "assists_p90", "rating"]], use_container_width=True, hide_index=True)

        st.divider()
        col1, col2 = st.columns(2)
        overperformers = players[players["minutes"] >= 90].copy()
        overperformers["expected_goals_proxy"] = overperformers["shots_on_target"] * 0.35
        overperformers["overperformance"] = (overperformers["goals"] - overperformers["expected_goals_proxy"]).round(2)

        with col1:
            st.markdown("**Overperformers**")
            st.dataframe(overperformers.nlargest(10, "overperformance")[["name", "team", "goals", "expected_goals_proxy", "overperformance"]], use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Underperformers**")
            st.dataframe(overperformers.nsmallest(10, "overperformance")[["name", "team", "goals", "expected_goals_proxy", "overperformance"]], use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Team Needs Analysis")
        st.markdown("Select a team to find where they need reinforcement.")
        st.divider()

        league_tw = st.selectbox("League", LEAGUES, key="needs_league")
        frame_tw = filter_matches_by_league(league_tw)
        team_options = sorted(set(frame_tw["home_team"].astype(str)) | set(frame_tw["away_team"].astype(str)))
        team_tw = st.selectbox("Select Team", team_options, key="needs_team")

        home_tw = frame_tw[frame_tw["home_team"].astype(str) == team_tw]
        away_tw = frame_tw[frame_tw["away_team"].astype(str) == team_tw]

        avg_scored = (safe_mean(home_tw["home_goals"], 0.0) + safe_mean(away_tw["away_goals"], 0.0)) / 2
        avg_conceded = (safe_mean(home_tw["away_goals"], 0.0) + safe_mean(away_tw["home_goals"], 0.0)) / 2
        league_avg_scored = frame_tw["total_goals"].mean() / 2
        league_avg_conceded = frame_tw["total_goals"].mean() / 2
        attack_diff = avg_scored - league_avg_scored
        defence_diff = avg_conceded - league_avg_conceded

        all_team_results = pd.concat(
            [
                home_tw.assign(gf=home_tw["home_goals"], ga=home_tw["away_goals"]),
                away_tw.assign(gf=away_tw["away_goals"], ga=away_tw["home_goals"]),
            ],
            ignore_index=True,
        )

        home_wr = (home_tw["result"] == 1).mean() if len(home_tw) else 0
        away_wr = (away_tw["result"] == -1).mean() if len(away_tw) else 0
        clean_sheet_rate = (all_team_results["ga"] == 0).mean() if len(all_team_results) else 0
        fail_score_rate = (all_team_results["gf"] == 0).mean() if len(all_team_results) else 0

        weaknesses = []
        strengths = []
        if attack_diff < -0.2:
            weaknesses.append(f"⚠️ **Weak attack**: scoring {avg_scored:.2f} goals per game vs league average {league_avg_scored:.2f}.")
        elif attack_diff > 0.2:
            strengths.append(f"✅ **Strong attack**: scoring {avg_scored:.2f} goals per game vs league average {league_avg_scored:.2f}.")

        if defence_diff > 0.2:
            weaknesses.append(f"⚠️ **Leaky defence**: conceding {avg_conceded:.2f} goals per game vs league average {league_avg_conceded:.2f}.")
        elif defence_diff < -0.2:
            strengths.append(f"✅ **Solid defence**: conceding {avg_conceded:.2f} goals per game vs league average {league_avg_conceded:.2f}.")

        if home_wr < 0.35:
            weaknesses.append(f"⚠️ **Poor home form**: winning only {home_wr:.0%} of home games.")
        if away_wr < 0.25:
            weaknesses.append(f"⚠️ **Poor away form**: winning only {away_wr:.0%} of away games.")
        if fail_score_rate > 0.25:
            weaknesses.append(f"⚠️ **Scoring issue**: fails to score in {fail_score_rate:.0%} of games.")
        if clean_sheet_rate < 0.20:
            weaknesses.append(f"⚠️ **Defensive frailty**: keeps a clean sheet in only {clean_sheet_rate:.0%} of games.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**⚠️ Areas Needing Improvement**")
            if weaknesses:
                for item in weaknesses:
                    st.markdown(item)
            else:
                st.markdown("No major weaknesses identified.")
        with col2:
            st.markdown("**✅ Team Strengths**")
            if strengths:
                for item in strengths:
                    st.markdown(item)
            else:
                st.markdown("No standout strengths identified.")

        st.divider()
        st.subheader("💡 Transfer Suggestions")
        if any("attack" in item.lower() or "score" in item.lower() for item in weaknesses):
            insight_card("⚽", f"<b>{team_tw}</b> need attacking reinforcement. Look for high goals per 90 and contribution per 90.")
            striker_targets = players[(players["position"].astype(str).str.contains("forward|attacker|striker", case=False, na=False)) & (players["minutes"] >= 90)].nlargest(5, "goals_p90")
            st.dataframe(striker_targets[["name", "team", "age", "goals_p90", "assists_p90", "rating"]], use_container_width=True, hide_index=True)
        if any("defence" in item.lower() or "defensive" in item.lower() for item in weaknesses):
            insight_card("🛡️", f"<b>{team_tw}</b> need defensive reinforcement. Look for high tackles and interceptions per 90.")
            defender_targets = players[(players["position"].astype(str).str.contains("defender|back", case=False, na=False)) & (players["minutes"] >= 90)].nlargest(5, "tackles_p90")
            st.dataframe(defender_targets[["name", "team", "age", "tackles_p90", "interc_p90", "rating"]], use_container_width=True, hide_index=True)
        if not weaknesses:
            insight_card("✅", f"<b>{team_tw}</b> appear balanced. Focus on depth signings rather than emergency starters.")

    with tab4:
        st.subheader("Attack vs Defence")
        st.markdown("Team efficiency across the selected league.")
        st.divider()

        league_scatter = st.selectbox("League", LEAGUES, key="scatter_league")
        frame_scatter = filter_matches_by_league(league_scatter)

        home_stats = frame_scatter.groupby("home_team").agg(home_attack=("home_goals", "mean"), home_defence=("away_goals", "mean"))
        away_stats = frame_scatter.groupby("away_team").agg(away_attack=("away_goals", "mean"), away_defence=("home_goals", "mean"))
        team_stats = pd.DataFrame(
            {
                "attack": (home_stats["home_attack"] + away_stats["away_attack"]) / 2,
                "defence": (home_stats["home_defence"] + away_stats["away_defence"]) / 2,
            }
        ).dropna().reset_index()
        team_stats.columns = ["team", "attack", "defence"]
        team_stats["score"] = team_stats["attack"] - team_stats["defence"]

        stop_if_empty(team_stats, "Not enough data to build attack vs defence chart.")

        best_attack = team_stats.loc[team_stats["attack"].idxmax()]
        best_defence = team_stats.loc[team_stats["defence"].idxmin()]
        insight_card("⚔️", f"<b>{best_attack['team']}</b> have the strongest attack at <b>{best_attack['attack']:.2f}</b> goals per game.")
        insight_card("🛡️", f"<b>{best_defence['team']}</b> have the strongest defence, conceding <b>{best_defence['defence']:.2f}</b> per game.")

        highlight = st.text_input("🔍 Highlight a team", placeholder="Example: Chelsea", key="scatter_search")
        team_stats["highlight"] = team_stats["team"].astype(str).str.contains(highlight, case=False, na=False) if highlight else False

        fig = go.Figure()
        normal = team_stats[~team_stats["highlight"]]
        fig.add_trace(
            go.Scatter(
                x=normal["defence"],
                y=normal["attack"],
                mode="markers",
                name="Teams",
                marker=dict(size=9, color=normal["score"], colorscale=[[0, "#ff3b30"], [0.5, "#ff9f0a"], [1, "#34c759"]], line=dict(width=0)),
                text=normal["team"],
                hovertemplate="<b>%{text}</b><br>Attack: %{y:.2f}<br>Defence: %{x:.2f}<extra></extra>",
            )
        )
        highlighted = team_stats[team_stats["highlight"]]
        if not highlighted.empty:
            fig.add_trace(
                go.Scatter(
                    x=highlighted["defence"],
                    y=highlighted["attack"],
                    mode="markers+text",
                    name="Highlighted",
                    marker=dict(size=16, color="#0071e3", line=dict(color="white", width=2)),
                    text=highlighted["team"],
                    textposition="top center",
                )
            )
        fig.update_layout(**BASE_LAYOUT, xaxis_title="Avg Goals Conceded", yaxis_title="Avg Goals Scored", height=520)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 Attack")
            st.dataframe(team_stats.nlargest(10, "attack")[["team", "attack", "defence"]], use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Top 10 Defence")
            st.dataframe(team_stats.nsmallest(10, "defence")[["team", "attack", "defence"]], use_container_width=True, hide_index=True)


# =============================================================================
# MODEL PERFORMANCE
# =============================================================================
# =============================================================================
# MODEL PERFORMANCE PAGE — Replace your existing elif page == "📈  Model Performance": block
# =============================================================================
elif page == "📈  Model Performance":
    st.markdown("# Model Performance")
    st.markdown("Review how the match prediction model performs on recent unseen seasons.")
    st.divider()

    if not metrics:
        st.warning("No metrics found. Run python scripts/train_model.py first.")
        st.stop()

    def first_metric_value(data, keys, default=0):
        for key in keys:
            if key in data:
                return data[key]
        return default

    def as_list(value):
        if value is None:
            return []

        if isinstance(value, list):
            return [str(item) for item in value]

        if isinstance(value, tuple):
            return [str(item) for item in value]

        if isinstance(value, set):
            return [str(item) for item in sorted(value)]

        return [str(value)]

    def build_importance_frame(value):
        if value is None:
            return pd.DataFrame(columns=["feature", "importance"])

        if isinstance(value, list):
            frame = pd.DataFrame(value)

            if frame.empty:
                return pd.DataFrame(columns=["feature", "importance"])

            if "feature" not in frame.columns or "importance" not in frame.columns:
                return pd.DataFrame(columns=["feature", "importance"])

            frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0)
            return frame[["feature", "importance"]]

        if isinstance(value, dict):
            if "feature" in value and "importance" in value:
                frame = pd.DataFrame([value])
                frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0)
                return frame[["feature", "importance"]]

            frame = pd.DataFrame(
                [{"feature": key, "importance": item} for key, item in value.items()]
            )

            frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0)
            return frame[["feature", "importance"]]

        return pd.DataFrame(columns=["feature", "importance"])

    accuracy = float(first_metric_value(metrics, ["accuracy", "test_accuracy", "model_accuracy"], 0))
    train_rows = int(first_metric_value(metrics, ["train_rows", "training_rows"], 0))
    test_rows = int(first_metric_value(metrics, ["test_rows", "testing_rows"], 0))

    raw_train_seasons = first_metric_value(metrics, ["train_seasons", "training_seasons"], [])
    raw_test_seasons = first_metric_value(metrics, ["test_seasons", "testing_seasons"], [])
    
    train_seasons = [
    format_season(season)
    for season in sorted_seasons(as_list(raw_train_seasons))
]
    test_seasons = [
    format_season(season)
    for season in sorted_seasons(as_list(raw_test_seasons))
]

    report = first_metric_value(metrics, ["classification_report", "report"], {})
    matrix = first_metric_value(metrics, ["confusion_matrix", "matrix"], [])
    feature_importance = first_metric_value(metrics, ["feature_importance", "importances"], [])
    model_features = as_list(first_metric_value(metrics, ["model_features", "features"], []))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy:.1%}")
    c2.metric("Training Rows", f"{train_rows:,}")
    c3.metric("Testing Rows", f"{test_rows:,}")
    c4.metric("Model Features", len(model_features))

    st.divider()

    if accuracy >= 0.50:
        insight_card("✅", f"The model achieved <b>{accuracy:.1%}</b> accuracy, which is strong for a three outcome football prediction task.")
    elif accuracy >= 0.42:
        insight_card("📊", f"The model achieved <b>{accuracy:.1%}</b> accuracy. This is realistic for football because draws and away wins are difficult to predict.")
    else:
        insight_card("⚠️", f"The model achieved <b>{accuracy:.1%}</b> accuracy. This suggests the feature set or training approach needs improvement.")

    if test_seasons:
        insight_card("🧪", f"The model was tested on recent unseen seasons: <b>{', '.join(test_seasons)}</b>.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Train Test Split")

        split_frame = pd.DataFrame(
            {
                "Dataset": ["Training", "Testing"],
                "Rows": [train_rows, test_rows],
            }
        )

        fig = go.Figure(
            go.Bar(
                x=split_frame["Dataset"],
                y=split_frame["Rows"],
                marker=dict(color=["#0071e3", "#ff9f0a"], line=dict(width=0)),
                text=split_frame["Rows"],
                textposition="outside",
                hovertemplate="%{x}<br>Rows: %{y:,}<extra></extra>",
            )
        )

        fig.update_layout(
            **BASE_LAYOUT,
            height=360,
            yaxis_title="Rows",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Seasons Used")

        st.markdown("**Training seasons**")
        st.write(", ".join(train_seasons) if train_seasons else "Not available")

        st.markdown("**Testing seasons**")
        st.write(", ".join(test_seasons) if test_seasons else "Not available")

    st.divider()

    st.subheader("Classification Report")

    report_rows = []

    if isinstance(report, dict):
        for label in ["Away Win", "Draw", "Home Win", "macro avg", "weighted avg"]:
            if label in report and isinstance(report[label], dict):
                values = report[label]

                report_rows.append(
                    {
                        "Class": label,
                        "Precision": round(float(values.get("precision", 0)), 4),
                        "Recall": round(float(values.get("recall", 0)), 4),
                        "F1 Score": round(float(values.get("f1-score", 0)), 4),
                        "Support": int(values.get("support", 0)),
                    }
                )

    if report_rows:
        report_frame = pd.DataFrame(report_rows)
        st.dataframe(report_frame, use_container_width=True, hide_index=True)
    else:
        st.info("Classification report not available. Run python scripts/train_model.py again to regenerate metrics.json.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")

        if matrix and isinstance(matrix, list):
            labels = ["Away Win", "Draw", "Home Win"]

            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=[f"Predicted {label}" for label in labels],
                    y=[f"Actual {label}" for label in labels],
                    text=matrix,
                    texttemplate="%{text}",
                    hovertemplate="%{y}<br>%{x}<br>Matches: %{z}<extra></extra>",
                    colorscale="Blues",
                )
            )

            fig.update_layout(
                **BASE_LAYOUT,
                height=420,
            )

            fig.update_xaxes(side="top")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Confusion matrix not available. Run python scripts/train_model.py again to regenerate metrics.json.")

    with col2:
        st.subheader("Model Reality Check")

        st.markdown(
            """
            Football prediction is difficult because the model has to choose between three outcomes:

            **Home Win**, **Draw**, and **Away Win**.

            Draws are usually the hardest class because they sit between both win outcomes. A realistic football model should be judged on whether it beats a simple baseline, not whether it predicts every match correctly.
            """
        )

        if report_rows:
            draw_row = next((row for row in report_rows if row["Class"] == "Draw"), None)

            if draw_row:
                st.metric("Draw F1 Score", f"{draw_row['F1 Score']:.3f}")

    st.divider()

    st.subheader("Feature Importance")

    importance_frame = build_importance_frame(feature_importance)

    if not importance_frame.empty:
        importance_frame = importance_frame.sort_values("importance", ascending=False).head(15)

        fig = go.Figure(
            go.Bar(
                x=importance_frame["importance"],
                y=importance_frame["feature"],
                orientation="h",
                marker=dict(color="#0071e3", line=dict(width=0)),
                text=importance_frame["importance"].round(4),
                textposition="outside",
                hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            **BASE_LAYOUT,
            height=520,
            xaxis_title="Importance",
            yaxis_title="Feature",
        )

        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)

        top_feature = importance_frame.iloc[0]

        insight_card(
            "🔥",
            f"The most important feature is <b>{top_feature['feature']}</b>, with an importance score of <b>{top_feature['importance']:.4f}</b>.",
        )

        st.markdown("### Top Features Table")

        st.dataframe(
            importance_frame.rename(
                columns={
                    "feature": "Feature",
                    "importance": "Importance",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Feature importance not available. Run python scripts/train_model.py again to regenerate metrics.json.")

    st.divider()

    st.subheader("Limitations")

    st.markdown(
        """
        This model does not currently include injuries, confirmed lineups, tactical changes, transfer news, weather, betting market odds, or live form updates.

        The predictions are based on historical match patterns, engineered team form, Elo strength, recent points, goals, shots, clean sheets, failed to score rate, streaks, and head to head history.
        """
    )