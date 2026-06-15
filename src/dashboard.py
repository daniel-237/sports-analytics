import json
import html
import re
import unicodedata
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
import streamlit.components.v1 as components
from src.prediction import predict_match, latest_team_elo
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


st.set_page_config(
    page_title="Football Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif !important;
        color: #1d1d1f !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: #ffffff !important;
        color: #1d1d1f !important;
    }

    .main .block-container {
        padding: 44px 56px !important;
        max-width: 1280px !important;
    }

    [data-testid="stSidebar"] {
        background-color: #f5f5f7 !important;
        border-right: 1px solid #e0e0e5 !important;
    }

    [data-testid="stSidebar"] * {
        color: #1d1d1f !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1d1d1f !important;
        letter-spacing: -0.6px !important;
    }

    h1 {
        font-size: 48px !important;
        font-weight: 700 !important;
        line-height: 1.05 !important;
    }

    h2 {
        font-size: 28px !important;
        font-weight: 650 !important;
    }

    h3 {
        font-size: 21px !important;
        font-weight: 650 !important;
    }

    p, li, span, label, div {
        color: #1d1d1f !important;
    }

    .small-muted {
        color: #515154 !important;
        font-size: 15px !important;
        line-height: 1.55 !important;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #1d1d1f !important;
    }

    [data-testid="stMarkdownContainer"] li {
        color: #1d1d1f !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: #1d1d1f !important;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stTextInput"] label {
        color: #1d1d1f !important;
        font-weight: 500 !important;
    }

    [data-testid="stSelectbox"] div,
    [data-testid="stNumberInput"] div,
    [data-testid="stTextInput"] div {
        color: #1d1d1f !important;
    }

    [data-baseweb="select"] {
        background-color: #f5f5f7 !important;
        border-radius: 14px !important;
    }

    [data-baseweb="select"] * {
        color: #1d1d1f !important;
        -webkit-text-fill-color: #1d1d1f !important;
    }

    [data-baseweb="input"] {
        background-color: #f5f5f7 !important;
        border-radius: 14px !important;
    }

    input {
        color: #1d1d1f !important;
        -webkit-text-fill-color: #1d1d1f !important;
        background-color: #f5f5f7 !important;
    }

    input::placeholder {
        color: #86868b !important;
        -webkit-text-fill-color: #86868b !important;
        opacity: 1 !important;
    }

    [data-testid="stNumberInput"] button {
        color: #1d1d1f !important;
        background-color: #f5f5f7 !important;
    }

    [data-testid="stMetric"] {
        background: #f5f5f7 !important;
        border-radius: 18px !important;
        padding: 20px !important;
    }

    [data-testid="metric-container"] {
        background: #f5f5f7 !important;
        border-radius: 18px !important;
        padding: 22px !important;
        border: none !important;
    }

    [data-testid="metric-container"] label {
        color: #515154 !important;
        font-size: 12px !important;
        font-weight: 650 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
    }

    [data-testid="stMetricValue"] {
        color: #1d1d1f !important;
        font-size: 32px !important;
        font-weight: 750 !important;
    }

    .stButton > button {
        background-color: #0071e3 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 12px 28px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    .stButton > button * {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #0077ed !important;
        transform: scale(1.01) !important;
        box-shadow: 0 4px 18px rgba(0,113,227,0.25) !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid #e0e0e5 !important;
        margin: 34px 0 !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 18px !important;
        overflow: hidden !important;
        border: 1px solid #e0e0e5 !important;
    }

    .insight-card {
        background: linear-gradient(135deg, #0071e3 0%, #0051a0 100%) !important;
        border-radius: 18px !important;
        padding: 20px 24px !important;
        margin-bottom: 10px !important;
    }

    .insight-card p,
    .insight-card span,
    .insight-card b {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .insight-card p {
        font-size: 15px !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    header {
        visibility: hidden !important;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding: 28px 18px 96px 18px !important;
            max-width: 100% !important;
        }

        h1 {
            font-size: 40px !important;
            line-height: 1.05 !important;
            letter-spacing: -1.4px !important;
        }

        h2 {
            font-size: 27px !important;
        }

        h3 {
            font-size: 21px !important;
        }

        p, li {
            font-size: 16px !important;
            line-height: 1.55 !important;
        }

        [data-testid="stSidebar"] {
        background-color: #f5f5f7 !important;
        }


        [data-testid="stSelectbox"],
        [data-testid="stNumberInput"],
        [data-testid="stTextInput"] {
            width: 100% !important;
        }

        [data-baseweb="select"] {
            min-height: 54px !important;
        }

        input {
            min-height: 52px !important;
            font-size: 16px !important;
        }

        .stButton > button {
            width: 100% !important;
            min-height: 56px !important;
            font-size: 18px !important;
        }

        [data-testid="metric-container"] {
            padding: 18px !important;
            margin-bottom: 12px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 28px !important;
        }

        .js-plotly-plot .modebar {
            display: none !important;
        }

        iframe {
            max-width: 100% !important;
        }

        .stDataFrame {
            max-width: 100% !important;
            overflow-x: auto !important;
        }
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

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

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
LINEUPS_PATH = Path("data/processed/lineups_clean.csv")
MODEL_PATH = Path("models/match_predictor.pkl")
METRICS_PATH = Path("models/metrics.json")

PAGES = [
    "  Home",
    "  Overview",
    "  Match Predictor",
    "  Player Stats",
    "  Player Comparison",
    "  Team Analysis",
    "  Transfer Analysis",
    "  Model Performance",
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
        return " Win"
    if gf == ga:
        return " Draw"
    return " Loss"


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




def player_position_group(value: str) -> str:
    text = str(value).upper()
    if any(token in text for token in ["GK", "KEEPER", "GOALKEEPER"]):
        return "Goalkeeper"
    if any(token in text for token in ["DF", "CB", "BACK", "DEF"]):
        return "Defender"
    if any(token in text for token in ["DM", "CM", "MF", "MID"]):
        return "Midfielder"
    if any(token in text for token in ["FW", "ST", "ATT", "WING", "FORWARD"]):
        return "Forward"
    return "Outfield"


def percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.nunique() <= 1:
        return pd.Series(0.5, index=series.index)
    return numeric.rank(pct=True).fillna(0.5)


def weighted_percentile_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    total = 0.0
    for column, weight in weights.items():
        if column in frame.columns:
            score += percentile(frame[column]) * abs(weight) * (1 if weight >= 0 else -1)
            total += abs(weight)
    if total == 0:
        return pd.Series(50.0, index=frame.index)
    return (score / total * 100).clip(0, 100)


def add_performance_scores(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["position_group"] = frame["position"].apply(player_position_group)

    for column in [
        "starts", "minutes", "goals", "assists", "shots_total", "shots_on_target", "tackles", "tackles_won",
        "interceptions", "blocks", "shots_blocked", "passes_blocked", "clearances", "errors", "yellow_cards", "red_cards",
        "goals_p90", "assists_p90", "contrib_p90", "shots_p90", "sot_p90", "tackles_p90", "interc_p90",
        "blocks_p90", "clearances_p90", "tackles_won_p90", "errors_p90", "cards_p90"
    ]:
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    if "tackles" in frame.columns and "tackles_won" in frame.columns:
        frame["tackles"] = np.where(frame["tackles"] <= 0, frame["tackles_won"], frame["tackles"])
        minutes_safe = frame["minutes"].replace(0, np.nan)
        frame["tackles_p90"] = (frame["tackles"] / minutes_safe * 90).fillna(0).round(2)

    frame["attacking_score"] = 0.0
    frame["creative_score"] = 0.0
    frame["defensive_score"] = 0.0
    frame["performance_score"] = 0.0

    group_profiles = {
        "Forward": {
            "attack": {"goals_p90": 2.6, "contrib_p90": 2.1, "shots_p90": 1.2, "sot_p90": 1.5, "shots_on_target": 0.5},
            "creative": {"assists_p90": 1.6, "assists": 0.7, "contrib_p90": 0.5},
            "defence": {"tackles_p90": 0.5, "interc_p90": 0.3, "blocks_p90": 0.2},
            "mix": (0.66, 0.22, 0.04, 0.08),
        },
        "Midfielder": {
            "attack": {"goals_p90": 0.8, "contrib_p90": 1.0, "shots_p90": 0.6, "sot_p90": 0.7},
            "creative": {"assists_p90": 1.4, "assists": 0.8, "contrib_p90": 0.5},
            "defence": {"tackles_p90": 1.3, "tackles_won_p90": 0.9, "interc_p90": 1.3, "blocks_p90": 0.5, "clearances_p90": 0.3, "errors_p90": -0.7},
            "mix": (0.22, 0.32, 0.34, 0.12),
        },
        "Defender": {
            "attack": {"goals_p90": 0.3, "assists_p90": 0.5, "shots_p90": 0.2},
            "creative": {"assists_p90": 0.6, "passes_blocked": 0.2},
            "defence": {"tackles_p90": 0.9, "tackles_won_p90": 1.1, "interc_p90": 1.2, "blocks_p90": 1.0, "clearances_p90": 1.1, "errors_p90": -1.0},
            "mix": (0.08, 0.12, 0.46, 0.34),
        },
        "Goalkeeper": {
            "attack": {"minutes": 0.2},
            "creative": {"minutes": 1.0},
            "defence": {"minutes": 1.0, "errors_p90": -0.8},
            "mix": (0.00, 0.10, 0.55, 0.35),
        },
        "Outfield": {
            "attack": {"goals_p90": 1.1, "contrib_p90": 1.2, "shots_p90": 0.8},
            "creative": {"assists_p90": 1.0, "assists": 0.6},
            "defence": {"tackles_p90": 1.0, "interc_p90": 1.0, "blocks_p90": 0.6, "clearances_p90": 0.5},
            "mix": (0.28, 0.22, 0.34, 0.16),
        },
    }

    for group, profile in group_profiles.items():
        mask = frame["position_group"] == group
        if not mask.any():
            continue
        part = frame.loc[mask]
        attack = weighted_percentile_score(part, profile["attack"])
        creative = weighted_percentile_score(part, profile["creative"])
        defence = weighted_percentile_score(part, profile["defence"])
        minutes_component = percentile(part["minutes"]) * 100 if "minutes" in part.columns else pd.Series(50.0, index=part.index)
        starts_component = percentile(part["starts"]) * 100 if "starts" in part.columns else minutes_component
        availability = (minutes_component * 0.60 + starts_component * 0.40).clip(0, 100)
        a, c, d, av = profile["mix"]
        if group == "Forward":
            defensive_bonus = ((defence - 50).clip(lower=0) * 0.08)
            overall = attack * a + creative * c + availability * av + defensive_bonus
        else:
            overall = attack * a + creative * c + defence * d + availability * av
        if group == "Defender":
            display_defence = (defence * 0.35 + availability * 0.65).clip(0, 100)
            overall = attack * 0.08 + creative * 0.12 + display_defence * 0.46 + availability * 0.34
        elif group == "Goalkeeper":
            display_defence = (defence * 0.65 + availability * 0.35).clip(0, 100)
            overall = display_defence * 0.55 + availability * 0.35 + creative * 0.10
        else:
            display_defence = defence
        frame.loc[mask, "attacking_score"] = attack.round(1)
        frame.loc[mask, "creative_score"] = creative.round(1)
        frame.loc[mask, "defensive_score"] = display_defence.round(1)
        frame.loc[mask, "performance_score"] = overall.clip(0, 100).round(1)

    return frame

@st.cache_data
def load_match_data(file_version: float) -> pd.DataFrame:
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
def load_player_data(file_version: float) -> pd.DataFrame | None:
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
            "competition": ["comp", "league", "division", "competition_name"],
            "appearances": ["apps", "matches", "games"],
            "minutes": ["mins", "minutes_played"],
            "shots_total": ["shots", "total_shots"],
            "shots_on_target": ["sot", "shots_target"],
            "pass_accuracy": ["passing_accuracy", "passes_accuracy"],
            "yellow_cards": ["yellow", "cards_yellow"],
            "red_cards": ["red", "cards_red"],
            "duels_won": ["duels", "duels_won_total"],
            "starts": ["start", "starts_total"],
            "tackles_won": ["tklw", "tackles_won_total"],
            "blocks": ["blocks_total", "blocked"],
            "shots_blocked": ["shot_blocks", "shots_blocks"],
            "passes_blocked": ["pass_blocks", "passes_blocks"],
            "clearances": ["clr", "clearance"],
            "errors": ["err", "mistakes"],
            "challenge_success_rate": ["tkl_pct", "challenge_tackle_pct"],
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
            "competition": "Unknown League",
            "age": 0,
            "goals": 0,
            "assists": 0,
            "minutes": 0,
            "appearances": 0,
            "starts": 0,
            "shots_total": 0,
            "shots_on_target": 0,
            "pass_accuracy": 0,
            "dribbles": 0,
            "tackles": 0,
            "tackles_won": 0,
            "interceptions": 0,
            "blocks": 0,
            "shots_blocked": 0,
            "passes_blocked": 0,
            "clearances": 0,
            "errors": 0,
            "challenge_success_rate": 0,
            "performance_score": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "duels_won": 0,
            "crosses": 0,
            "key_passes": 0,
            "progressive_carries": 0,
            "progressive_passes": 0,
            "expected_goals": 0,
            "expected_assists": 0,
            "clean_sheets": 0,
            "saves": 0,
            "goals_against": 0,
        },
    )

    numeric = [
        "age",
        "goals",
        "assists",
        "minutes",
        "appearances",
        "starts",
        "shots_total",
        "shots_on_target",
        "pass_accuracy",
        "dribbles",
        "tackles",
        "tackles_won",
        "interceptions",
        "blocks",
        "shots_blocked",
        "passes_blocked",
        "clearances",
        "errors",
        "challenge_success_rate",
        "performance_score",
        "yellow_cards",
        "red_cards",
        "duels_won",
        "crosses",
        "key_passes",
        "progressive_carries",
        "progressive_passes",
        "expected_goals",
        "expected_assists",
        "clean_sheets",
        "saves",
        "goals_against",
    ]
    frame = numeric_columns(frame, numeric)

    minutes_safe = frame["minutes"].replace(0, np.nan)
    frame["goals_p90"] = (frame["goals"] / minutes_safe * 90).round(2)
    frame["assists_p90"] = (frame["assists"] / minutes_safe * 90).round(2)
    frame["shots_p90"] = (frame["shots_total"] / minutes_safe * 90).round(2)
    frame["sot_p90"] = (frame["shots_on_target"] / minutes_safe * 90).round(2)
    frame["tackles"] = np.where(frame["tackles"] <= 0, frame["tackles_won"], frame["tackles"])
    frame["tackles_p90"] = (frame["tackles"] / minutes_safe * 90).round(2)
    frame["tackles_won_p90"] = (frame["tackles_won"] / minutes_safe * 90).round(2)
    frame["interc_p90"] = (frame["interceptions"] / minutes_safe * 90).round(2)
    frame["blocks_p90"] = (frame["blocks"] / minutes_safe * 90).round(2)
    frame["clearances_p90"] = (frame["clearances"] / minutes_safe * 90).round(2)
    frame["errors_p90"] = (frame["errors"] / minutes_safe * 90).round(3)
    frame["contrib_p90"] = ((frame["goals"] + frame["assists"]) / minutes_safe * 90).round(2)
    frame["cards_p90"] = ((frame["yellow_cards"] + frame["red_cards"]) / minutes_safe * 90).round(2)
    frame = frame.fillna(0)
    frame = add_performance_scores(frame)

    return frame




@st.cache_data
def load_lineup_data(file_version: float) -> pd.DataFrame:
    if not LINEUPS_PATH.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(LINEUPS_PATH, low_memory=False)
    except Exception:
        return pd.DataFrame()
    frame = normalise_columns(frame)
    frame = ensure_columns(
        frame,
        {
            "match_id": "Unknown",
            "date": pd.NaT,
            "competition": "Unknown",
            "season": "Unknown",
            "stage": "Unknown",
            "home_team": "Unknown",
            "away_team": "Unknown",
            "home_score": 0,
            "away_score": 0,
            "team": "Unknown",
            "opponent": "Unknown",
            "venue": "Unknown",
            "player": "Unknown Player",
            "player_display_name": "Unknown Player",
            "jersey_number": "",
            "position": "Unknown",
            "starter": False,
        },
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["starter"] = frame["starter"].astype(str).str.lower().isin(["true", "1", "yes"])
    frame = numeric_columns(frame, ["home_score", "away_score"])
    return frame.sort_values("date", ascending=False, na_position="last")

@st.cache_resource
def load_model(file_version: float):
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.warning(f"Model could not be loaded: {exc}")
        return None


@st.cache_data
def load_metrics(file_version: float) -> dict:
    if not METRICS_PATH.exists():
        return {}

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def path_version(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def season_start(value) -> int:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown season", "all"}:
        return -1
    if "/" in text:
        first = text.split("/")[0]
        if first.isdigit():
            return int(first)
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) >= 8 and digits[:4].isdigit():
        return int(digits[:4])
    if len(digits) >= 4:
        value_4 = digits[:4]
        if value_4.isdigit() and 1800 <= int(value_4) <= 2200:
            return int(value_4)
        yy = int(value_4[:2])
        return 1900 + yy if yy >= 70 else 2000 + yy
    return -1


def sorted_season_values(values) -> list[str]:
    cleaned = [str(value).strip() for value in values if str(value).strip() and str(value).strip().lower() not in {"nan", "none", "unknown season", "all"}]
    return sorted(list(dict.fromkeys(cleaned)), key=lambda value: (season_start(value), value), reverse=True)


def display_competition(value) -> str:
    text = str(value).strip()
    prefixes = {
        "eng ": "",
        "es ": "",
        "it ": "",
        "de ": "",
        "fr ": "",
    }
    for prefix, replacement in prefixes.items():
        if text.lower().startswith(prefix):
            text = replacement + text[len(prefix):]
            break
    replacements = {
        "1. Bundesliga": "Bundesliga",
        "de Bundesliga": "Bundesliga",
        "eng Premier League": "Premier League",
        "es La Liga": "La Liga",
        "it Serie A": "Serie A",
        "fr Ligue 1": "Ligue 1",
    }
    return replacements.get(str(value).strip(), text)


def competition_key(value) -> str:
    return display_competition(value).strip().lower()




def overview_league_values() -> list[str]:
    values = ["All"] + MATCH_LEAGUES.copy()
    seen_labels = {display_competition(value).lower() for value in values}
    if players is not None and len(players) and "competition" in players.columns:
        for value in sorted(players["competition"].dropna().astype(str).unique().tolist(), key=lambda item: display_competition(item)):
            label = display_competition(value).lower()
            if label not in seen_labels:
                values.append(value)
                seen_labels.add(label)
    return values


def player_seasons_for(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty or "season" not in frame.columns:
        return PLAYER_SEASONS
    values = sorted_season_values(frame["season"].dropna().astype(str).unique().tolist())
    return values if values else PLAYER_SEASONS


def match_seasons_for(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty or "season" not in frame.columns:
        return []
    return sorted_season_values(frame["season"].dropna().astype(str).unique().tolist())


def current_season_default(options: list[str]) -> int:
    return 0 if options else None


def filter_players(frame: pd.DataFrame, league: str = "All", season: str | None = None) -> pd.DataFrame:
    filtered = frame.copy()
    if league != "All" and "competition" in filtered.columns:
        target = competition_key(league)
        filtered = filtered[filtered["competition"].astype(str).map(competition_key) == target]
    if season is not None and season != "All" and "season" in filtered.columns:
        filtered = filtered[filtered["season"].astype(str) == str(season)]
    return filtered


def player_pool_with_defaults(league: str, season: str | None) -> pd.DataFrame:
    if players is None:
        return pd.DataFrame()
    return filter_players(players, league, season)


def team_level_from_league(league: str) -> str:
    label = display_competition(league).lower()
    if label in {"premier league", "la liga", "serie a", "bundesliga", "ligue 1"}:
        return "elite"
    if label == "championship":
        return "upper_efl"
    if label in {"league one", "league two"}:
        return "lower_efl"
    return "non_league"


def realistic_recruitment_pool(team_league: str, pool: pd.DataFrame) -> pd.DataFrame:
    level = team_level_from_league(team_league)
    candidates = pool.copy()
    if candidates.empty:
        return candidates
    if level == "elite":
        return candidates[candidates["minutes"] >= 450]
    if level == "upper_efl":
        return candidates[(candidates["age"] <= 25) & (candidates["minutes"] <= 1800) & (candidates["performance_score"] <= 72)]
    if level == "lower_efl":
        return candidates[(candidates["age"] <= 23) & (candidates["minutes"] <= 1200) & (candidates["performance_score"] <= 68)]
    return candidates[(candidates["age"] <= 21) & (candidates["minutes"] <= 900) & (candidates["performance_score"] <= 64)]


matches = load_match_data(path_version(MATCH_PATH))
players = load_player_data(path_version(PLAYER_PATH))
lineups = load_lineup_data(path_version(LINEUPS_PATH))
model = load_model(path_version(MODEL_PATH))
metrics = load_metrics(path_version(METRICS_PATH))

MATCH_LEAGUES = sorted(matches["league"].dropna().astype(str).unique().tolist())
PREDICTOR_LEAGUES = [league for league in MATCH_LEAGUES if str(league).lower() != "international"]
LEAGUES = ["All"] + MATCH_LEAGUES
PLAYER_LEAGUES = ["All"]
PLAYER_SEASONS = []
if players is not None and len(players):
    player_league_values = sorted(players["competition"].dropna().astype(str).unique().tolist(), key=lambda value: display_competition(value))
    PLAYER_LEAGUES += player_league_values
    PLAYER_SEASONS = sorted_season_values(players["season"].dropna().astype(str).unique().tolist())



def filter_matches_by_league(league: str) -> pd.DataFrame:
    if league == "All":
        return matches.copy()
    return matches[matches["league"].astype(str) == str(league)].copy()


def available_player_teams(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty:
        return []
    return sorted(frame["team"].dropna().astype(str).unique().tolist())


def available_match_teams(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty:
        return []
    return sorted(set(frame["home_team"].astype(str)) | set(frame["away_team"].astype(str)))



def latest_player_season_for(league: str) -> str | None:
    if players is None or players.empty:
        return None
    pool = filter_players(players, league, None)
    seasons = player_seasons_for(pool)
    return seasons[0] if seasons else None


def latest_player_pool(league: str) -> pd.DataFrame:
    season = latest_player_season_for(league)
    return filter_players(players, league, season) if season else pd.DataFrame()


def latest_match_season_for(league: str) -> str | None:
    pool = filter_matches_by_league(league)
    seasons = match_seasons_for(pool)
    return seasons[0] if seasons else None


def recent_team_matches(frame: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    team_frame = frame[(frame["home_team"].astype(str) == str(team)) | (frame["away_team"].astype(str) == str(team))].copy()
    if "date" in team_frame.columns:
        team_frame = team_frame.sort_values("date", na_position="last")
    return team_frame.tail(n)


def result_for_team(row: pd.Series, team: str) -> str:
    is_home = str(row.get("home_team", "")) == str(team)
    gf = row.get("home_goals", 0) if is_home else row.get("away_goals", 0)
    ga = row.get("away_goals", 0) if is_home else row.get("home_goals", 0)
    if gf > ga:
        return "W"
    if gf < ga:
        return "L"
    return "D"


def render_form_badges(team: str, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        st.info("No recent form available for this team.")
        return
    styles = {
        "W": "background:#34c759;color:#ffffff;",
        "L": "background:#ff3b30;color:#ffffff;",
        "D": "background:#8e8e93;color:#ffffff;",
    }
    badges = []
    for _, row in frame.iterrows():
        result = result_for_team(row, team)
        badges.append(f"<span style='display:inline-flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:999px;font-size:16px;font-weight:800;margin-right:8px;{styles[result]}'>{result}</span>")
    st.markdown("<div style='display:flex;align-items:center;margin:8px 0 18px 0;'>" + "".join(badges) + "</div>", unsafe_allow_html=True)


def position_group_from_position(value: str) -> str:
    text = str(value).upper()
    if any(token in text for token in ["GK", "KEEP"]):
        return "Goalkeeper"
    if any(token in text for token in ["DF", "CB", "BACK", "DEF"]):
        return "Defender"
    if any(token in text for token in ["MF", "CM", "DM", "MID"]):
        return "Midfielder"
    if any(token in text for token in ["FW", "ST", "ATT", "WINGER", "FORWARD"]):
        return "Forward"
    return "Outfield"


def estimated_starting_xi(squad: pd.DataFrame) -> pd.DataFrame:
    if squad is None or squad.empty:
        return pd.DataFrame()
    available = squad.copy()
    available["position_group"] = available["position"].apply(position_group_from_position)
    available["selection_score"] = available["minutes"].fillna(0) * 0.7 + available["performance_score"].fillna(0) * 250
    shape = [("Goalkeeper", 1), ("Defender", 4), ("Midfielder", 3), ("Forward", 3)]
    picks = []
    used = set()
    for group, count in shape:
        group_pool = available[(available["position_group"] == group) & (~available.index.isin(used))].sort_values("selection_score", ascending=False).head(count)
        picks.append(group_pool)
        used.update(group_pool.index.tolist())
    selected = pd.concat(picks, ignore_index=False) if picks else pd.DataFrame()
    if len(selected) < 11:
        filler = available[~available.index.isin(used)].sort_values("selection_score", ascending=False).head(11 - len(selected))
        selected = pd.concat([selected, filler], ignore_index=False)
    if selected.empty:
        return selected
    selected = selected.sort_values(["position_group", "selection_score"], ascending=[True, False])
    return selected[["name", "position", "age", "minutes", "goals", "assists", "performance_score"]].head(11)


def show_estimated_starting_xi(squad: pd.DataFrame, title: str = "Estimated Starting XI") -> None:
    xi = estimated_starting_xi(squad)
    st.subheader(title)
    if xi.empty:
        st.info("No squad data available to estimate a starting XI.")
    else:
        st.caption("Estimated from available squad minutes and performance score data. Confirmed lineups are shown separately when matchday lineup data is available.")
        st.dataframe(xi, use_container_width=True, hide_index=True)





FORMATION_SLOTS = {
    "4-3-3": [
        ("GK", "Goalkeeper", 50, 88),
        ("RB", "Right Back", 80, 72),
        ("RCB", "Right Center Back", 62, 74),
        ("LCB", "Left Center Back", 38, 74),
        ("LB", "Left Back", 20, 72),
        ("DM", "Defensive Midfielder", 50, 58),
        ("RCM", "Right Center Midfielder", 63, 45),
        ("LCM", "Left Center Midfielder", 37, 45),
        ("RW", "Right Wing", 80, 24),
        ("LW", "Left Wing", 20, 24),
        ("ST", "Center Forward", 50, 14),
    ],
    "4-2-3-1": [
        ("GK", "Goalkeeper", 50, 88),
        ("RB", "Right Back", 80, 72),
        ("RCB", "Right Center Back", 62, 74),
        ("LCB", "Left Center Back", 38, 74),
        ("LB", "Left Back", 20, 72),
        ("RDM", "Right Defensive Midfielder", 60, 58),
        ("LDM", "Left Defensive Midfielder", 40, 58),
        ("RW", "Right Wing", 80, 35),
        ("AM", "Attacking Midfielder", 50, 34),
        ("LW", "Left Wing", 20, 35),
        ("ST", "Center Forward", 50, 14),
    ],
    "4-4-2": [
        ("GK", "Goalkeeper", 50, 88),
        ("RB", "Right Back", 80, 72),
        ("RCB", "Right Center Back", 62, 74),
        ("LCB", "Left Center Back", 38, 74),
        ("LB", "Left Back", 20, 72),
        ("RM", "Right Midfielder", 80, 48),
        ("RCM", "Right Center Midfielder", 60, 50),
        ("LCM", "Left Center Midfielder", 40, 50),
        ("LM", "Left Midfielder", 20, 48),
        ("RST", "Right Center Forward", 60, 18),
        ("LST", "Left Center Forward", 40, 18),
    ],
    "3-4-3": [
        ("GK", "Goalkeeper", 50, 88),
        ("RCB", "Right Center Back", 67, 74),
        ("CB", "Center Back", 50, 76),
        ("LCB", "Left Center Back", 33, 74),
        ("RM", "Right Wing Back", 82, 54),
        ("RCM", "Right Center Midfielder", 60, 52),
        ("LCM", "Left Center Midfielder", 40, 52),
        ("LM", "Left Wing Back", 18, 54),
        ("RW", "Right Wing", 78, 24),
        ("LW", "Left Wing", 22, 24),
        ("ST", "Center Forward", 50, 14),
    ],
    "3-5-2": [
        ("GK", "Goalkeeper", 50, 88),
        ("RCB", "Right Center Back", 67, 74),
        ("CB", "Center Back", 50, 76),
        ("LCB", "Left Center Back", 33, 74),
        ("RWB", "Right Wing Back", 84, 54),
        ("DM", "Defensive Midfielder", 50, 58),
        ("RCM", "Right Center Midfielder", 62, 45),
        ("LCM", "Left Center Midfielder", 38, 45),
        ("LWB", "Left Wing Back", 16, 54),
        ("RST", "Right Center Forward", 60, 16),
        ("LST", "Left Center Forward", 40, 16),
    ],
}

PLAYER_POSITION_OVERRIDES = {
    "cole palmer": ["AM", "RW", "CM"],
    "moises caicedo": ["DM", "CM", "RB"],
    "enzo fernandez": ["CM", "AM", "DM"],
    "pedro neto": ["RW", "LW", "AM"],
    "alejandro garnacho": ["LW", "RW"],
    "joao pedro": ["ST", "AM", "LW"],
    "liam delap": ["ST"],
    "marc guiu": ["ST"],
    "marc cucurella": ["LB", "LCB", "LWB"],
    "malo gusto": ["RB", "RWB", "LB"],
    "reece james": ["RB", "RWB", "DM"],
    "trevoh chalobah": ["CB", "DM"],
    "wesley fofana": ["CB", "RB"],
    "levi colwill": ["CB", "LB"],
    "benoit badiashile": ["CB"],
    "jorrel hato": ["LB", "CB"],
    "tosin adarabioyo": ["CB"],
    "robert sanchez": ["GK"],
    "filip jorgensen": ["GK"],
    "bukayo saka": ["RW"],
    "martin odegaard": ["AM", "CM"],
    "declan rice": ["DM", "CM"],
    "william saliba": ["CB"],
    "gabriel magalhaes": ["CB"],
    "jurrien timber": ["RB", "LB", "CB"],
    "ben white": ["RB", "CB"],
    "riccardo calafiori": ["LB", "CB"],
    "david raya": ["GK"],
    "kai havertz": ["ST", "AM"],
    "gabriel martinelli": ["LW", "RW"],
    "leandro trossard": ["LW", "ST", "AM"],
    "mohamed salah": ["RW"],
    "virgil van dijk": ["CB"],
    "ibrahima konate": ["CB"],
    "alisson": ["GK"],
    "trent alexander arnold": ["RB", "CM"],
    "trent alexander-arnold": ["RB", "CM"],
    "conor bradley": ["RB"],
    "andy robertson": ["LB"],
    "andrew robertson": ["LB"],
    "alexis mac allister": ["CM", "DM"],
    "ryan gravenberch": ["CM", "DM"],
    "dominik szoboszlai": ["AM", "CM", "RW"],
    "cody gakpo": ["LW", "ST"],
    "darwin nunez": ["ST", "LW"],
    "luis diaz": ["LW", "RW"],
    "erling haaland": ["ST"],
    "rodri": ["DM"],
    "ruben dias": ["CB"],
    "john stones": ["CB", "DM"],
    "josko gvardiol": ["LB", "CB"],
    "manuel akanji": ["CB", "RB"],
    "nathan ake": ["CB", "LB"],
    "rico lewis": ["RB", "DM", "LB"],
    "phil foden": ["AM", "RW", "LW"],
    "bernardo silva": ["CM", "AM", "RW"],
    "kevin de bruyne": ["AM", "CM"],
    "jeremy doku": ["LW", "RW"],
    "savinho": ["RW", "LW"],
    "omar marmoush": ["ST", "LW"],
    "ederson": ["GK"],
    "andre onana": ["GK"],
    "diogo dalot": ["RB", "LB"],
    "noussair mazraoui": ["RB", "LB"],
    "lisandro martinez": ["CB", "LB"],
    "matthijs de ligt": ["CB"],
    "leny yoro": ["CB"],
    "harry maguire": ["CB"],
    "luke shaw": ["LB", "CB"],
    "manuel ugarte": ["DM"],
    "casemiro": ["DM"],
    "kobbie mainoo": ["CM", "DM"],
    "bruno fernandes": ["AM", "CM"],
    "amad diallo": ["RW", "AM"],
    "marcus rashford": ["LW", "ST"],
    "rasmus hojlund": ["ST"],
    "joshua zirkzee": ["ST", "AM"],
    "harry kane": ["ST"],
    "kylian mbappe": ["ST", "LW"],
    "vinicius junior": ["LW", "ST"],
    "rodrygo": ["RW", "LW", "ST"],
    "jude bellingham": ["AM", "CM"],
    "federico valverde": ["CM", "RW", "DM"],
    "aurelien tchouameni": ["DM", "CB"],
    "eduardo camavinga": ["CM", "DM", "LB"],
    "dani carvajal": ["RB"],
    "ferland mendy": ["LB"],
    "antonio rudiger": ["CB"],
    "thibaut courtois": ["GK"],
    "lamine yamal": ["RW"],
    "raphinha": ["RW", "LW"],
    "robert lewandowski": ["ST"],
    "pedri": ["CM", "AM"],
    "gavi": ["CM", "AM"],
    "frenkie de jong": ["CM", "DM"],
    "alejandro balde": ["LB"],
    "jules kounde": ["RB", "CB"],
    "ronald araujo": ["CB", "RB"],
    "pau cubarsi": ["CB"],
    "marc andre ter stegen": ["GK"],
    "marc-andre ter stegen": ["GK"],
    "ousmane dembele": ["RW", "LW", "ST"],
    "khvicha kvaratskhelia": ["LW"],
    "vitinha": ["CM", "AM"],
    "achraf hakimi": ["RB", "RWB"],
    "nuno mendes": ["LB", "LWB"],
    "marquinhos": ["CB"],
    "gianluigi donnarumma": ["GK"],
    "lautaro martinez": ["ST"],
    "nicolo barella": ["CM"],
    "hakan calhanoglu": ["DM", "CM"],
    "alessandro bastoni": ["LCB", "CB"],
    "denzel dumfries": ["RWB", "RB"],
    "theo hernandez": ["LB", "LWB"],
    "rafael leao": ["LW"],
    "christian pulisic": ["RW", "LW", "AM"],
    "mike maignan": ["GK"],
    "victor osimhen": ["ST"],
    "jamal musiala": ["AM", "LW"],
    "leroy sane": ["RW", "LW"],
    "michael olise": ["RW", "AM"],
    "kingsley coman": ["LW", "RW"],
    "alphonso davies": ["LB", "LWB"],
    "joshua kimmich": ["DM", "CM", "RB"],
    "dayot upamecano": ["CB"],
    "min jae kim": ["CB"],
    "min-jae kim": ["CB"],
    "manuel neuer": ["GK"],
}


LINEUP_REPUTATION_BOOST = {
    "william saliba": 18,
    "gabriel magalhaes": 12,
    "gabriel magalhães": 12,
    "virgil van dijk": 18,
    "ruben dias": 15,
    "rúben dias": 15,
    "antonio rudiger": 14,
    "antonio rüdiger": 14,
    "marquinhos": 13,
    "alessandro bastoni": 13,
    "declan rice": 12,
    "rodri": 18,
    "moises caicedo": 12,
    "cole palmer": 14,
    "bukayo saka": 14,
    "mohamed salah": 18,
    "erling haaland": 18,
    "kylian mbappe": 18,
    "kylian mbappé": 18,
    "vinicius junior": 16,
    "vinícius júnior": 16,
    "jude bellingham": 16,
    "lamine yamal": 16,
    "harry kane": 16,
}

ROLE_ALIASES = {
    "GK": "GK",
    "GOALKEEPER": "GK",
    "KEEPER": "GK",
    "RB": "RB",
    "RIGHT BACK": "RB",
    "RIGHT-BACK": "RB",
    "RWB": "RB",
    "RIGHT WING BACK": "RB",
    "LB": "LB",
    "LEFT BACK": "LB",
    "LEFT-BACK": "LB",
    "LWB": "LB",
    "LEFT WING BACK": "LB",
    "CB": "CB",
    "CENTER BACK": "CB",
    "CENTRE BACK": "CB",
    "CENTRAL DEFENDER": "CB",
    "RIGHT CENTER BACK": "CB",
    "LEFT CENTER BACK": "CB",
    "DEFENDER": "CB",
    "DM": "DM",
    "CDM": "DM",
    "DEFENSIVE MIDFIELD": "DM",
    "DEFENSIVE MIDFIELDER": "DM",
    "CENTER DEFENSIVE MIDFIELD": "DM",
    "CENTRE DEFENSIVE MIDFIELD": "DM",
    "CM": "CM",
    "CENTER MIDFIELD": "CM",
    "CENTRE MIDFIELD": "CM",
    "CENTER MIDFIELDER": "CM",
    "CENTRE MIDFIELDER": "CM",
    "CENTRAL MIDFIELD": "CM",
    "MIDFIELDER": "CM",
    "AM": "AM",
    "CAM": "AM",
    "ATTACKING MIDFIELD": "AM",
    "ATTACKING MIDFIELDER": "AM",
    "CENTER ATTACKING MIDFIELD": "AM",
    "CENTRE ATTACKING MIDFIELD": "AM",
    "RW": "RW",
    "RIGHT WING": "RW",
    "RIGHT WINGER": "RW",
    "RM": "RW",
    "RIGHT MIDFIELD": "RW",
    "RIGHT MIDFIELDER": "RW",
    "LW": "LW",
    "LEFT WING": "LW",
    "LEFT WINGER": "LW",
    "LM": "LW",
    "LEFT MIDFIELD": "LW",
    "LEFT MIDFIELDER": "LW",
    "ST": "ST",
    "CF": "ST",
    "FW": "ST",
    "FORWARD": "ST",
    "STRIKER": "ST",
    "CENTER FORWARD": "ST",
    "CENTRE FORWARD": "ST",
    "CENTER-FORWARD": "ST",
}

SLOT_CORE = {
    "GK": "GK",
    "RB": "RB",
    "RWB": "RB",
    "RCB": "CB",
    "CB": "CB",
    "LCB": "CB",
    "LB": "LB",
    "LWB": "LB",
    "DM": "DM",
    "RDM": "DM",
    "LDM": "DM",
    "RCM": "CM",
    "CM": "CM",
    "LCM": "CM",
    "AM": "AM",
    "RM": "RW",
    "RW": "RW",
    "LM": "LW",
    "LW": "LW",
    "ST": "ST",
    "RST": "ST",
    "LST": "ST",
}

POSITION_FIT = {
    "GK": {"GK": 100},
    "RB": {"RB": 100, "LB": 65, "CB": 45, "DM": 38, "CM": 25},
    "LB": {"LB": 100, "RB": 65, "CB": 45, "DM": 38, "CM": 25},
    "CB": {"CB": 100, "RB": 48, "LB": 48, "DM": 44},
    "DM": {"DM": 100, "CM": 78, "CB": 55, "RB": 45, "LB": 45, "AM": 35},
    "CM": {"CM": 100, "DM": 82, "AM": 78, "RW": 42, "LW": 42, "RB": 32, "LB": 32},
    "AM": {"AM": 100, "CM": 76, "RW": 75, "LW": 75, "ST": 56, "DM": 34},
    "RW": {"RW": 100, "LW": 80, "AM": 78, "ST": 35, "CM": 35},
    "LW": {"LW": 100, "RW": 80, "AM": 78, "ST": 35, "CM": 35},
    "ST": {"ST": 100, "AM": 64, "RW": 46, "LW": 46},
}

SLOT_FILL_ORDER = ["GK", "RB", "LB", "RCB", "LCB", "CB", "RWB", "LWB", "DM", "RDM", "LDM", "ST", "RST", "LST", "RW", "LW", "AM", "RCM", "LCM", "CM", "RM", "LM"]


def strip_accents(value):
    return "".join(char for char in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(char))


def player_key(value):
    text = strip_accents(value).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def numeric_value(row, column, default=0.0):
    if column not in row.index:
        return default
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value):
        return default
    return float(value)


def scale_rating(value):
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return 50.0
    if value <= 10:
        return float(np.clip((value - 5.4) / 1.0 * 35 + 55, 35, 95))
    return float(np.clip(value, 0, 100))


def percentile_series(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if values.nunique() <= 1:
        return pd.Series([50.0] * len(values), index=series.index)
    return values.rank(pct=True).fillna(0.5) * 100


def canonical_from_text(value):
    text = str(value).upper().replace(",", " ")
    text = text.replace("/", " ").replace("-", " ")
    found = []

    for token in re.split(r"[\s,;/]+", text):
        if token in ROLE_ALIASES:
            found.append(ROLE_ALIASES[token])

    for raw, canonical in ROLE_ALIASES.items():
        if raw in text:
            found.append(canonical)

    cleaned = []
    for pos in found:
        if pos not in cleaned:
            cleaned.append(pos)

    return cleaned


def infer_positions_from_row(row):
    name = player_key(row.get("name", row.get("player", "")))

    if name in PLAYER_POSITION_OVERRIDES:
        return PLAYER_POSITION_OVERRIDES[name]

    raw_values = []
    for column in ["primary_position", "best_position", "detailed_position", "position", "positions", "player_positions"]:
        if column in row.index:
            raw_values.append(row[column])

    joined = " ".join(str(value) for value in raw_values if str(value).lower() not in {"nan", "none", ""})
    parsed = canonical_from_text(joined)

    if parsed:
        raw_upper = joined.upper()
        if parsed == ["ST"] and "MF" in raw_upper:
            return ["AM", "ST", "RW", "LW"]
        if parsed == ["CM"] and "FW" in raw_upper:
            return ["AM", "RW", "LW", "ST", "CM"]
        if parsed == ["CB"] and "MF" in raw_upper:
            return ["RB", "LB", "DM", "CB"]
        if parsed == ["CM"] and "DF" in raw_upper:
            return ["DM", "CM", "RB", "LB"]
        return parsed

    raw_position = str(row.get("position", "")).upper()
    goals_p90 = numeric_value(row, "goals_p90")
    assists_p90 = numeric_value(row, "assists_p90")
    tackles_p90 = numeric_value(row, "tackles_p90")
    interceptions_p90 = numeric_value(row, "interc_p90")

    if "GK" in raw_position:
        return ["GK"]

    if "FW" in raw_position and "MF" in raw_position:
        return ["AM", "RW", "LW", "ST"]

    if "MF" in raw_position and "FW" in raw_position:
        return ["AM", "RW", "LW", "ST", "CM"]

    if "DF" in raw_position and "MF" in raw_position:
        return ["RB", "LB", "DM", "CB"]

    if "MF" in raw_position and "DF" in raw_position:
        return ["DM", "CM", "RB", "LB"]

    if "FW" in raw_position:
        if assists_p90 > goals_p90 and assists_p90 >= 0.20:
            return ["RW", "LW", "AM", "ST"]
        return ["ST", "LW", "RW", "AM"]

    if "DF" in raw_position:
        return ["CB", "RB", "LB"]

    if "MF" in raw_position:
        if goals_p90 + assists_p90 >= 0.35:
            return ["AM", "RW", "LW", "CM"]
        if tackles_p90 + interceptions_p90 >= 2.0:
            return ["DM", "CM", "AM"]
        return ["CM", "AM", "DM"]

    return ["CM"]


def prepare_lineup_pool(squad):
    frame = squad.copy()

    if frame.empty:
        return frame

    if "name" not in frame.columns and "player" in frame.columns:
        frame["name"] = frame["player"]

    frame["name_key"] = frame["name"].apply(player_key)
    frame["minutes"] = pd.to_numeric(frame.get("minutes", 0), errors="coerce").fillna(0)
    frame["goals"] = pd.to_numeric(frame.get("goals", 0), errors="coerce").fillna(0)
    frame["assists"] = pd.to_numeric(frame.get("assists", 0), errors="coerce").fillna(0)

    if "performance_score" in frame.columns:
        frame["performance_score"] = pd.to_numeric(frame["performance_score"], errors="coerce").fillna(50).clip(0, 100)
    elif "rating" in frame.columns:
        frame["performance_score"] = frame["rating"].apply(scale_rating)
    else:
        frame["performance_score"] = 50.0

    for column in ["goals_p90", "assists_p90", "shots_p90", "sot_p90", "tackles_p90", "interc_p90", "contrib_p90"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    for column in ["shots_total", "shots_on_target", "key_passes", "crosses", "progressive_passes", "progressive_carries", "tackles", "tackles_won", "interceptions", "blocks", "clearances", "errors", "starts", "saves", "clean_sheets", "goals_against"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    frame["tackles"] = np.where(frame["tackles"] <= 0, frame["tackles_won"], frame["tackles"])
    frame["reputation_boost"] = frame["name_key"].map(LINEUP_REPUTATION_BOOST).fillna(0).astype(float)

    frame = frame.sort_values(["name_key", "minutes", "performance_score"], ascending=[True, False, False])
    frame = frame.drop_duplicates("name_key", keep="first")

    frame["position_options"] = frame.apply(infer_positions_from_row, axis=1)
    frame["minutes_score"] = percentile_series(frame["minutes"])
    frame["starts_score"] = percentile_series(frame["starts"])
    frame["availability_score"] = (frame["minutes_score"] * 0.65 + frame["starts_score"] * 0.35).clip(0, 100)
    frame["attacking_score"] = percentile_series(frame["goals"] * 1.25 + frame["assists"] + frame["goals_p90"] * 10 + frame["assists_p90"] * 7 + frame["shots_p90"] * 2 + frame["sot_p90"] * 4)
    frame["creative_score"] = percentile_series(frame["assists"] * 1.5 + frame["assists_p90"] * 10 + frame["key_passes"] + frame["crosses"] * 0.4 + frame["progressive_passes"] * 0.2 + frame["progressive_carries"] * 0.2)
    frame["defensive_score"] = percentile_series(frame["tackles"] * 0.7 + frame["tackles_won"] * 1.0 + frame["interceptions"] * 1.1 + frame["blocks"] * 0.8 + frame["clearances"] * 0.9 + frame["tackles_p90"] * 5 + frame["interc_p90"] * 7 + frame["minutes"] * 0.012 + frame["starts"] * 0.8 - frame["errors"] * 6)
    frame["keeper_score"] = percentile_series(frame["minutes"] + frame["saves"] * 4 + frame["clean_sheets"] * 8 - frame["goals_against"] * 2)

    return frame.reset_index(drop=True)


def slot_core(slot):
    return SLOT_CORE.get(slot, slot)


def role_strength(row, role):
    performance = numeric_value(row, "performance_score", 50)
    minutes = numeric_value(row, "minutes_score", 50)
    availability = numeric_value(row, "availability_score", minutes)
    attack = numeric_value(row, "attacking_score", 50)
    creative = numeric_value(row, "creative_score", 50)
    defensive = numeric_value(row, "defensive_score", 50)
    keeper = numeric_value(row, "keeper_score", 50)
    reputation = numeric_value(row, "reputation_boost", 0)

    if role == "GK":
        return keeper * 0.45 + performance * 0.20 + availability * 0.35 + reputation

    if role == "ST":
        return attack * 0.50 + performance * 0.20 + availability * 0.20 + creative * 0.06 + defensive * 0.04 + reputation

    if role in {"RW", "LW"}:
        return attack * 0.36 + creative * 0.24 + performance * 0.18 + availability * 0.17 + defensive * 0.05 + reputation

    if role == "AM":
        return creative * 0.34 + attack * 0.26 + performance * 0.18 + availability * 0.17 + defensive * 0.05 + reputation

    if role == "CM":
        return performance * 0.22 + availability * 0.30 + creative * 0.22 + defensive * 0.18 + attack * 0.08 + reputation

    if role == "DM":
        return defensive * 0.35 + availability * 0.30 + performance * 0.18 + creative * 0.12 + attack * 0.05 + reputation

    if role in {"RB", "LB"}:
        return defensive * 0.28 + availability * 0.32 + creative * 0.15 + performance * 0.15 + attack * 0.10 + reputation

    if role == "CB":
        return availability * 0.42 + defensive * 0.28 + performance * 0.20 + creative * 0.05 + attack * 0.05 + reputation

    return performance * 0.35 + availability * 0.30 + attack * 0.12 + creative * 0.10 + defensive * 0.13 + reputation


def position_fit_score(row, slot):
    role = slot_core(slot)
    options = row.get("position_options", [])

    if role == "GK":
        return 100 if "GK" in options else -500

    if "GK" in options:
        return -500

    scores = []
    for index, option in enumerate(options):
        base = POSITION_FIT.get(role, {}).get(option, -250)
        scores.append(base - index * 6)

    if not scores:
        return -250

    return max(scores)


def select_player_for_slot(pool, selected_keys, slot):
    role = slot_core(slot)
    candidates = []

    for _, row in pool.iterrows():
        key = row["name_key"]

        if key in selected_keys:
            continue

        fit = position_fit_score(row, slot)

        if fit < 42:
            continue

        strength = role_strength(row, role)
        total = strength + fit * 1.15

        if numeric_value(row, "minutes", 0) < 90:
            total -= 15

        candidates.append((total, fit, strength, row))

    if not candidates:
        for _, row in pool.iterrows():
            key = row["name_key"]

            if key in selected_keys:
                continue

            fit = position_fit_score(row, slot)

            if fit < 20:
                continue

            strength = role_strength(row, role)
            total = strength + fit * 0.80 - 20
            candidates.append((total, fit, strength, row))

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item[0], reverse=True)[0]


def build_predicted_lineup(squad, formation="4-3-3", bench_size=9):
    pool = prepare_lineup_pool(squad)

    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    slots = FORMATION_SLOTS.get(formation, FORMATION_SLOTS["4-3-3"])
    order = sorted(slots, key=lambda item: SLOT_FILL_ORDER.index(item[0]) if item[0] in SLOT_FILL_ORDER else 999)

    selected_keys = set()
    selected_rows = []

    for slot, label, x, y in order:
        chosen = select_player_for_slot(pool, selected_keys, slot)

        if chosen is None:
            continue

        total, fit, strength, row = chosen
        selected_keys.add(row["name_key"])

        record = row.to_dict()
        record["slot"] = slot
        record["slot_label"] = label
        record["plot_x"] = x
        record["plot_y"] = y
        record["position_fit"] = round(float(fit), 1)
        record["selection_score"] = round(float(total), 1)
        record["role_strength"] = round(float(strength), 1)
        record["player_display_name"] = record.get("name", record.get("player", "Unknown"))
        selected_rows.append(record)

    starters = pd.DataFrame(selected_rows)

    if not starters.empty:
        starters["slot_order"] = starters["slot"].apply(lambda value: SLOT_FILL_ORDER.index(value) if value in SLOT_FILL_ORDER else 999)
        starters = starters.sort_values("slot_order").drop(columns=["slot_order"])

    remaining = pool[~pool["name_key"].isin(selected_keys)].copy()

    if not remaining.empty:
        remaining["bench_score"] = remaining.apply(
            lambda row: max([role_strength(row, slot_core(slot[0])) + max(position_fit_score(row, slot[0]), 0) for slot in slots]),
            axis=1,
        )
        bench = remaining.sort_values(["bench_score", "minutes", "performance_score"], ascending=False).head(bench_size)
    else:
        bench = pd.DataFrame()

    return starters.reset_index(drop=True), bench.reset_index(drop=True)


def show_predicted_lineup_visual(squad: pd.DataFrame, team: str, formation: str) -> None:
    starters, bench = build_predicted_lineup(squad, formation)
    if starters.empty:
        st.info("No squad data available to generate a predicted lineup.")
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        render_lineup_pitch(starters, f"{team} Predicted XI", formation)
    with c2:
        st.markdown("### Bench Options")
        bench_table = clean_bench_table(bench)
        if bench_table.empty:
            st.info("No bench options available.")
        else:
            st.dataframe(bench_table, use_container_width=True, hide_index=True)
    st.caption("Predicted lineup based on players from the selected team and season, using position suitability, minutes and performance score. It is not a confirmed matchday lineup.")


def short_player_name(name: str) -> str:
    text = str(name).strip()
    parts = text.split()
    if len(parts) <= 2:
        return text
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def render_lineup_pitch(starters: pd.DataFrame, title: str, formation: str) -> None:
    if starters is None or starters.empty:
        st.info("No lineup could be generated for this team and season.")
        return

    markers = []

    for _, player in starters.iterrows():
        x = float(player.get("plot_x", 50))
        y = float(player.get("plot_y", 50))
        number = str(player.get("jersey_number", "")).replace(".0", "")
        if number.lower() in {"nan", "none"}:
            number = ""
        display_name = html.escape(short_player_name(player.get("player_display_name", player.get("name", "Unknown"))))
        pos_label = html.escape(str(player.get("slot_label", player.get("slot", ""))))

        markers.append(
            f"""
            <div class='lineup-player' style='left:{x}%;top:{y}%;'>
                <div class='lineup-badge'>{html.escape(number)}</div>
                <div class='lineup-name'>{display_name}</div>
                <div class='lineup-position'>{pos_label}</div>
            </div>
            """
        )

    lineup_html = f"""
    <style>
        .lineup-wrap {{
            background:#0b7f45;
            border-radius:26px;
            padding:18px;
            box-shadow:0 18px 42px rgba(0,0,0,0.16);
        }}
        .lineup-head {{
            display:flex;
            justify-content:space-between;
            align-items:center;
            color:white;
            margin-bottom:12px;
            font-weight:900;
            font-size:18px;
        }}
        .lineup-pitch {{
            position:relative;
            height:640px;
            border:3px solid rgba(255,255,255,0.78);
            border-radius:22px;
            overflow:hidden;
            background:linear-gradient(180deg,#10884d 0%,#08733f 100%);
        }}
        .lineup-pitch:before {{
            content:'';
            position:absolute;
            left:5%;
            right:5%;
            top:50%;
            border-top:2px solid rgba(255,255,255,0.55);
        }}
        .lineup-pitch:after {{
            content:'';
            position:absolute;
            left:38%;
            right:38%;
            top:42%;
            bottom:42%;
            border:2px solid rgba(255,255,255,0.55);
            border-radius:999px;
        }}
        .lineup-box-top {{
            position:absolute;
            left:28%;
            right:28%;
            top:0;
            height:16%;
            border-left:2px solid rgba(255,255,255,0.55);
            border-right:2px solid rgba(255,255,255,0.55);
            border-bottom:2px solid rgba(255,255,255,0.55);
        }}
        .lineup-box-bottom {{
            position:absolute;
            left:28%;
            right:28%;
            bottom:0;
            height:16%;
            border-left:2px solid rgba(255,255,255,0.55);
            border-right:2px solid rgba(255,255,255,0.55);
            border-top:2px solid rgba(255,255,255,0.55);
        }}
        .lineup-player {{
            position:absolute;
            transform:translate(-50%,-50%);
            text-align:center;
            width:118px;
            z-index:3;
        }}
        .lineup-badge {{
            width:46px;
            height:46px;
            border-radius:50%;
            background:#ffffff;
            color:#08753f;
            margin:0 auto 6px auto;
            display:flex;
            align-items:center;
            justify-content:center;
            font-weight:900;
            box-shadow:0 4px 10px rgba(0,0,0,0.22);
            font-size:17px;
        }}
        .lineup-name {{
            color:#ffffff;
            font-size:12px;
            font-weight:900;
            text-shadow:0 1px 4px rgba(0,0,0,0.75);
            line-height:1.05;
            white-space:normal;
        }}
        .lineup-position {{
            color:rgba(255,255,255,0.86);
            font-size:10px;
            font-weight:700;
            text-shadow:0 1px 3px rgba(0,0,0,0.75);
            line-height:1.05;
            margin-top:2px;
        }}
    </style>

    <div class='lineup-wrap'>
        <div class='lineup-head'>
            <div>{html.escape(title)}</div>
            <div>Formation: {html.escape(formation)}</div>
        </div>
        <div class='lineup-pitch'>
            <div class='lineup-box-top'></div>
            <div class='lineup-box-bottom'></div>
            {''.join(markers)}
        </div>
    </div>
    """

    components.html(lineup_html, height=740, scrolling=False)


def clean_bench_table(bench: pd.DataFrame) -> pd.DataFrame:
    if bench is None or bench.empty:
        return pd.DataFrame()

    columns = []
    for column in ["name", "position", "age", "minutes", "performance_score", "defensive_score", "position_fit", "goals", "assists", "tackles", "interceptions"]:
        if column in bench.columns:
            columns.append(column)

    table = bench[columns].copy()

    rename_map = {
        "name": "Player",
        "position": "Position",
        "age": "Age",
        "minutes": "Minutes",
        "performance_score": "Performance Score",
        "defensive_score": "Defensive Score",
        "position_fit": "Position Fit",
        "goals": "Goals",
        "assists": "Assists",
    }

    table = table.rename(columns=rename_map)

    if "Performance Score" in table.columns:
        table["Performance Score"] = pd.to_numeric(table["Performance Score"], errors="coerce").round(1)

    if "Minutes" in table.columns:
        table["Minutes"] = pd.to_numeric(table["Minutes"], errors="coerce").fillna(0).round(0).astype(int)

    return table

def lineup_competitions() -> list[str]:
    if lineups is None or lineups.empty:
        return []
    return sorted(lineups["competition"].dropna().astype(str).unique().tolist())


def lineup_seasons_for(competition: str) -> list[str]:
    if lineups is None or lineups.empty:
        return []
    frame = lineups[lineups["competition"].astype(str) == str(competition)].copy()
    return sorted_season_values(frame["season"].dropna().astype(str).unique().tolist())


def lineup_match_label(row: pd.Series) -> str:
    date = pd.to_datetime(row.get("date"), errors="coerce")
    date_text = date.strftime("%Y-%m-%d") if not pd.isna(date) else "Unknown date"
    score = ""
    if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
        score = f" ({int(row.get('home_score'))}-{int(row.get('away_score'))})"
    return f"{date_text}: {row.get('home_team')} vs {row.get('away_team')}{score}"

def season_sort_key(value: str) -> int:
    return sorted_seasons([value])[0] if False else 0


def initialise_navigation() -> None:
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = PAGES[0]
    if "nav_token" not in st.session_state:
        st.session_state["nav_token"] = 0


def go_to_page(page_name: str) -> None:
    if page_name in PAGES:
        st.session_state["active_page"] = page_name
        st.session_state["nav_token"] += 1


initialise_navigation()
active_page = st.session_state["active_page"]
nav_token = st.session_state["nav_token"]

st.markdown(
    """
    <style>
        .top-nav-area {
            display: block;
            margin-bottom: 34px;
        }

        .nav-title {
            font-size: 13px;
            font-weight: 600;
            color: #515154;
            margin-bottom: 8px;
        }

        @media (max-width: 768px) {
            .top-nav-area {
                margin-bottom: 24px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='top-nav-area'>", unsafe_allow_html=True)
st.markdown("<div class='nav-title'>Navigate</div>", unsafe_allow_html=True)

top_page = st.selectbox(
    "Navigate",
    PAGES,
    index=PAGES.index(active_page),
    key=f"top_nav_select_{nav_token}",
    label_visibility="collapsed",
)

st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
<div style='padding:28px 8px 20px 8px;'>
    <p style='font-size:11px;font-weight:600;color:#6e6e73;letter-spacing:1.2px;text-transform:uppercase;margin:0;'>Sports</p>
    <p style='font-size:26px;font-weight:700;color:#1d1d1f;letter-spacing:-0.8px;margin:4px 0 0 0;'>Analytics</p>
</div>
""",
    unsafe_allow_html=True,
)

sidebar_page = st.sidebar.radio(
    "",
    PAGES,
    index=PAGES.index(active_page),
    key=f"sidebar_nav_radio_{nav_token}",
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

if top_page != active_page:
    go_to_page(top_page)
    st.rerun()

if sidebar_page != active_page:
    go_to_page(sidebar_page)
    st.rerun()

page = st.session_state["active_page"]


if page == "  Home":
    st.markdown(
        """
        <div style='padding: 56px 0 20px 0; max-width: 960px;'>
            <p style='font-size: 13px; font-weight: 700; color: #0071e3; letter-spacing: 1.5px; text-transform: uppercase; margin: 0 0 16px 0;'>
                Football Intelligence Platform
            </p>
            <h1 style='font-size: 58px; font-weight: 700; color: #1d1d1f; line-height: 1.04; margin: 0 0 22px 0;'>
                Analyse. Predict.<br>Scout. Decide.
            </h1>
            <p style='font-size: 19px; color: #52525c; line-height: 1.7; max-width: 760px; margin: 0;'>
                A machine learning powered football analytics dashboard for match prediction, player comparison, team analysis, and recruitment insights across domestic and international football.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button(" Try Predictor", key="home_try_predictor"):
            go_to_page("  Match Predictor")
            st.rerun()
    with btn_col2:
        if st.button(" View Players", key="home_view_players"):
            go_to_page("  Player Stats")
            st.rerun()

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matches Analysed", f"{len(matches):,}")
    m2.metric("Leagues Covered", matches["league"].nunique())
    m3.metric("Seasons of Data", matches["season"].nunique())
    m4.metric("Player Records", f"{len(players):,}" if players is not None else "Not loaded")

    st.divider()

    st.markdown("## Core Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div style='border-radius: 20px; padding: 26px; background: #f7fafc; min-height: 240px;'>
                <div style='font-size: 34px; margin-bottom: 14px;'></div>
                <h3 style='margin: 0 0 12px 0;'>Match Predictor</h3>
                <p style='color: #52525c; line-height: 1.65; margin: 0;'>Predict Home Win, Draw or Away Win using model probabilities and recent form insights.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div style='border-radius: 20px; padding: 26px; background: #f7fafc; min-height: 240px;'>
                <div style='font-size: 34px; margin-bottom: 14px;'></div>
                <h3 style='margin: 0 0 12px 0;'>Player Comparison</h3>
                <p style='color: #52525c; line-height: 1.65; margin: 0;'>Compare players side-by-side with radar metrics, per 90 performance and head-to-head insights.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div style='border-radius: 20px; padding: 26px; background: #f7fafc; min-height: 240px;'>
                <div style='font-size: 34px; margin-bottom: 14px;'></div>
                <h3 style='margin: 0 0 12px 0;'>Team Analysis</h3>
                <p style='color: #52525c; line-height: 1.65; margin: 0;'>Review squad performance, match trends and lineup metrics for any team, league, and season.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("## How to use this dashboard")
    left_help, right_help = st.columns(2)
    with left_help:
        st.markdown("1. Open **Match Predictor** and select two teams.")
        st.markdown("2. Use **Player Stats** to filter by league, season, position and minutes.")
        st.markdown("3. Visit **Player Comparison** to compare players head-to-head.")
    with right_help:
        st.markdown("4. Open **Team Analysis** to inspect team form, goals, clean sheets and player output.")
        st.markdown("5. Use **Transfer Analysis** to build a shortlist of recruitment candidates.")
        st.markdown("6. Check **Model Performance** to review accuracy, confusion matrix and feature importance.")

    st.divider()
    st.caption("Built by Daniel Olutade · Python · Streamlit · Pandas · Plotly · Scikit Learn · XGBoost")


elif page == "  Overview":
    st.markdown("# Football Analytics")
    st.markdown("Match results and player coverage across domestic and international football.")

    match_league_set = set(MATCH_LEAGUES)
    player_league_values = []
    if players is not None and len(players) and "competition" in players.columns:
        player_league_values = players["competition"].dropna().astype(str).unique().tolist()

    combined_leagues = ["All"]
    seen = {"all"}
    for value in MATCH_LEAGUES + sorted(player_league_values, key=lambda item: display_competition(item)):
        label = display_competition(value).lower()
        if label not in seen:
            combined_leagues.append(value)
            seen.add(label)

    league = st.selectbox("League", combined_leagues, key="overview_league", format_func=display_competition)
    match_frame = filter_matches_by_league(league) if league in LEAGUES else pd.DataFrame()
    player_frame = filter_players(players, league, None) if players is not None and league != "All" else players.copy() if players is not None else pd.DataFrame()

    st.divider()

    if not match_frame.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Matches", f"{len(match_frame):,}")
        c2.metric("Leagues", match_frame["league"].nunique())
        c3.metric("Teams", len(set(match_frame["home_team"]) | set(match_frame["away_team"])))
        c4.metric("Seasons", match_frame["season"].nunique())
        st.divider()

        home_pct = (match_frame["result"] == 1).mean()
        avg_goals = match_frame["total_goals"].mean()
        top_home_team = match_frame.groupby("home_team")["home_goals"].mean().idxmax()

        insight_card("", f"Home teams win <b>{home_pct:.0%}</b> of all matches in this selection.")
        col1, col2 = st.columns(2)
        with col1:
            insight_card("", f"<b>{top_home_team}</b> average the most home goals per match.")
        with col2:
            insight_card("", f"Average of <b>{avg_goals:.2f}</b> goals per match.")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Match Outcomes")
            result_counts = match_frame["result"].map({1: "Home Win", 0: "Draw", -1: "Away Win"}).value_counts()
            fig = go.Figure(
                go.Pie(
                    values=result_counts.values,
                    labels=result_counts.index,
                    hole=0.6,
                    marker=dict(colors=["#0071e3", "#ff9f0a", "#ff3b30"], line=dict(color="#ffffff", width=2)),
                )
            )
            fig.update_layout(**BASE_LAYOUT, showlegend=True, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            st.subheader("Goals by Season")
            season_goals = (
                match_frame.groupby("season")
                .agg(avg_goals=("total_goals", "mean"), matches=("total_goals", "count"))
                .reset_index()
            )
            season_goals["season_order"] = season_goals["season"].apply(season_start)
            season_goals = season_goals.sort_values("season_order").tail(12)
            fig = go.Figure(
                go.Scatter(
                    x=season_goals["season"],
                    y=season_goals["avg_goals"],
                    mode="lines+markers",
                    line=dict(color="#0071e3", width=3),
                    marker=dict(size=8),
                    hovertemplate="%{x}<br>Avg goals: %{y:.2f}<extra></extra>",
                )
            )
            fig.update_layout(**BASE_LAYOUT, height=420, yaxis_title="Average Goals", xaxis_title="Season")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("League Entertainment Profile")
        profile_source = match_frame.copy()
        profile_source["over_2_5_goals"] = profile_source["total_goals"] > 2.5
        profile_source["both_teams_scored"] = (profile_source["home_goals"] > 0) & (profile_source["away_goals"] > 0)
        profile_source["home_win"] = profile_source["result"] == 1
        profile_source["draw"] = profile_source["result"] == 0
        profile_source["away_win"] = profile_source["result"] == -1

        league_profile = (
            profile_source.groupby("league")
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

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[display_competition(value) for value in league_profile["league"]],
                    y=league_profile["avg_goals"],
                    name="Average Goals",
                    marker=dict(color="#0071e3", line=dict(width=0)),
                    text=league_profile["avg_goals"].round(2),
                    textposition="outside",
                )
            )
            fig.update_layout(**BASE_LAYOUT, height=420, yaxis_title="Goals per Match", xaxis_title="League")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            best_goals = league_profile.loc[league_profile["avg_goals"].idxmax()]
            most_draws = league_profile.loc[league_profile["draw_rate"].idxmax()]
            most_btts = league_profile.loc[league_profile["btts_rate"].idxmax()]
            st.metric("Highest Scoring", display_competition(best_goals["league"]), f"{best_goals['avg_goals']:.2f} goals")
            st.metric("Most Draw Heavy", display_competition(most_draws["league"]), f"{most_draws['draw_rate']:.0%} draws")
            st.metric("Most BTTS", display_competition(most_btts["league"]), f"{most_btts['btts_rate']:.0%} BTTS")

        display_profile = league_profile.copy()
        display_profile["league"] = display_profile["league"].apply(display_competition)
        display_profile["avg_goals"] = display_profile["avg_goals"].round(2)
        for column in ["home_win_rate", "draw_rate", "away_win_rate", "over_2_5_rate", "btts_rate"]:
            display_profile[column] = (display_profile[column] * 100).round(1)
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
        st.dataframe(display_profile, use_container_width=True, hide_index=True)



    if lineups is not None and not lineups.empty:
        st.divider()
        st.subheader("European & Domestic Cup Lineup Coverage")
        lineup_profile = (
            lineups.groupby("competition")
            .agg(
                matches=("match_id", "nunique"),
                teams=("team", "nunique"),
                seasons=("season", "nunique"),
                latest_date=("date", "max"),
            )
            .reset_index()
            .sort_values(["latest_date", "matches"], ascending=False)
        )
        st.caption("These competitions come from the confirmed lineup dataset and can be used in Team Analysis lineup visuals.")
        st.dataframe(
            lineup_profile.rename(
                columns={
                    "competition": "Competition",
                    "matches": "Matches",
                    "teams": "Teams",
                    "seasons": "Seasons",
                    "latest_date": "Latest Match",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    if not player_frame.empty:
        st.divider()
        st.subheader("Player League Coverage")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Player Records", f"{len(player_frame):,}")
        p2.metric("Competitions", player_frame["competition"].nunique())
        p3.metric("Clubs", player_frame["team"].nunique())
        p4.metric("Seasons", player_frame["season"].nunique())

        player_profile = (
            player_frame.groupby("competition")
            .agg(
                players=("name", "count"),
                clubs=("team", "nunique"),
                goals=("goals", "sum"),
                assists=("assists", "sum"),
                avg_rating=("performance_score", "mean"),
            )
            .reset_index()
        )
        player_profile["display"] = player_profile["competition"].apply(display_competition)
        player_profile["avg_rating"] = player_profile["avg_rating"].round(2)
        player_profile = player_profile.sort_values(["goals", "assists"], ascending=False)

        fig = go.Figure(
            go.Bar(
                x=player_profile["display"],
                y=player_profile["players"],
                marker=dict(color="#0071e3", line=dict(width=0)),
                text=player_profile["players"],
                textposition="outside",
            )
        )
        fig.update_layout(**BASE_LAYOUT, height=420, yaxis_title="Player Records", xaxis_title="Competition")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.dataframe(
            player_profile[["display", "players", "clubs", "goals", "assists", "avg_rating"]].rename(
                columns={"display": "Competition", "players": "Players", "clubs": "Clubs", "goals": "Goals", "assists": "Assists", "avg_rating": "Avg Performance Score"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    elif match_frame.empty:
        st.warning("No data available for this selection.")


elif page == "  Match Predictor":
    st.markdown("# Match Predictor")
    st.markdown("Select two teams from any available league and season.")
    st.divider()

    if model is None:
        st.warning("Model file not found. Run python scripts/train_model.py first.")
        st.stop()

    match_leagues_available = [league for league in MATCH_LEAGUES if str(league).lower() != "international"]
    player_leagues_available = []
    if players is not None and len(players) and "competition" in players.columns:
        player_leagues_available = players["competition"].dropna().astype(str).unique().tolist()

    all_predictor_leagues = []
    seen = set()
    for value in match_leagues_available + sorted(player_leagues_available, key=lambda item: display_competition(item)):
        label = display_competition(value).lower()
        if label not in seen:
            all_predictor_leagues.append(value)
            seen.add(label)

    if not all_predictor_leagues:
        st.warning("No team data is available for predictions.")
        st.stop()

    def teams_for_selection(league: str, season: str) -> list[str]:
        team_values = []
        match_frame = matches[(matches["league"].astype(str) == str(league)) & (matches["season"].astype(str) == str(season))].copy()
        if not match_frame.empty:
            team_values += available_match_teams(match_frame)
        if players is not None and len(players) and "competition" in players.columns:
            player_frame = players[(players["competition"].astype(str) == str(league)) & (players["season"].astype(str) == str(season))].copy()
            if not player_frame.empty:
                team_values += available_player_teams(player_frame)
        return sorted(list(dict.fromkeys([str(team) for team in team_values if str(team).strip()])))

    def seasons_for_selection(league: str) -> list[str]:
        season_values = []
        match_frame = matches[matches["league"].astype(str) == str(league)].copy()
        if not match_frame.empty:
            season_values += match_seasons_for(match_frame)
        if players is not None and len(players) and "competition" in players.columns:
            player_frame = players[players["competition"].astype(str) == str(league)].copy()
            if not player_frame.empty:
                season_values += player_seasons_for(player_frame)
        return sorted_season_values(season_values)

    def squad_strength(team_name: str, league: str, season: str) -> float:
        if players is None or players.empty or "competition" not in players.columns:
            return 0.0
        squad = players[
            (players["competition"].astype(str) == str(league))
            & (players["season"].astype(str) == str(season))
            & (players["team"].astype(str) == str(team_name))
        ].copy()
        if squad.empty:
            return 0.0
        squad = squad[squad["minutes"] >= 180].copy()
        if squad.empty:
            return 0.0
        top_attack = squad.sort_values("contrib_p90", ascending=False).head(6)
        top_quality = squad.sort_values("performance_score", ascending=False).head(11)
        attack_score = float(top_attack["contrib_p90"].mean()) if len(top_attack) else 0.0
        rating_score = float(top_quality["performance_score"].mean()) / 10 if len(top_quality) else 0.0
        minutes_score = min(float(squad["minutes"].sum()) / 25000, 1.0)
        return rating_score + attack_score * 2.5 + minutes_score

    def squad_prediction(home_team_name: str, away_team_name: str, home_league_name: str, away_league_name: str, home_season_name: str, away_season_name: str) -> dict:
        home_strength = squad_strength(home_team_name, home_league_name, home_season_name)
        away_strength = squad_strength(away_team_name, away_league_name, away_season_name)
        diff = home_strength + 0.25 - away_strength
        scaled = max(min(diff / 3, 2.0), -2.0)
        home_base = 1 / (1 + np.exp(-scaled))
        draw_probability = max(0.18, min(0.30, 0.30 - abs(scaled) * 0.04))
        home_probability = home_base * (1 - draw_probability)
        away_probability = (1 - home_base) * (1 - draw_probability)
        probabilities = pd.DataFrame(
            {
                "Outcome": ["Home Win", "Draw", "Away Win"],
                "Probability": [home_probability, draw_probability, away_probability],
            }
        )
        best = probabilities.sort_values("Probability", ascending=False).iloc[0]
        return {
            "prediction": str(best["Outcome"]),
            "confidence": float(best["Probability"]),
            "probabilities": probabilities,
            "features": pd.DataFrame([{"home_squad_strength": home_strength, "away_squad_strength": away_strength, "strength_difference": diff}]),
            "explanation_features": pd.DataFrame([{"home_squad_strength": home_strength, "away_squad_strength": away_strength, "strength_difference": diff}]),
            "method": "Squad strength estimate",
        }

    premier_label = next((value for value in all_predictor_leagues if display_competition(value) == "Premier League"), all_predictor_leagues[0])
    default_home_index = all_predictor_leagues.index(premier_label)
    default_away_index = min(default_home_index + 1, len(all_predictor_leagues) - 1)

    col1, col2 = st.columns(2)
    with col1:
        home_league = st.selectbox("Home League", all_predictor_leagues, index=default_home_index, key="predictor_home_league", format_func=display_competition)
        home_seasons = seasons_for_selection(home_league)
        if not home_seasons:
            st.warning("No seasons available for the selected home league.")
            st.stop()
        home_season = st.selectbox("Home Season", home_seasons, index=0, key=f"predictor_home_season_{home_league}")
        home_teams = teams_for_selection(home_league, home_season)
        if not home_teams:
            st.warning("No teams available for the selected home league and season.")
            st.stop()
        st.markdown("** Home Team**")
        home_team = st.selectbox("Home Team", home_teams, index=0, label_visibility="collapsed", key=f"predictor_home_team_{home_league}_{home_season}")

        # Show Elo if available
        if "home_elo" in matches.columns:
            try:
                h_elo = latest_team_elo(matches, home_team)
                st.metric("Home Elo", f"{h_elo:.0f}")
            except Exception:
                pass

    with col2:
        away_league = st.selectbox("Away League", all_predictor_leagues, index=default_away_index, key="predictor_away_league", format_func=display_competition)
        away_seasons = seasons_for_selection(away_league)
        if not away_seasons:
            st.warning("No seasons available for the selected away league.")
            st.stop()
        away_season = st.selectbox("Away Season", away_seasons, index=0, key=f"predictor_away_season_{away_league}")
        away_teams = teams_for_selection(away_league, away_season)
        if not away_teams:
            st.warning("No teams available for the selected away league and season.")
            st.stop()
        st.markdown("** Away Team**")
        away_default = min(1, len(away_teams) - 1)
        away_team = st.selectbox("Away Team", away_teams, index=away_default, label_visibility="collapsed", key=f"predictor_away_team_{away_league}_{away_season}")

        # Show Elo if available
        if "away_elo" in matches.columns:
            try:
                a_elo = latest_team_elo(matches, away_team)
                st.metric("Away Elo", f"{a_elo:.0f}")
            except Exception:
                pass

    if home_team == away_team and home_league == away_league and home_season == away_season:
        st.warning("Choose two different teams.")
        st.stop()

    if st.button("Predict Match →", key="predict_match_button"):
        home_history = matches[matches["league"].astype(str) == str(home_league)].copy()
        away_history = matches[matches["league"].astype(str) == str(away_league)].copy()
        latest_dates = []
        if not home_history.empty:
            selected_home_dates = home_history[home_history["season"].astype(str) == str(home_season)]["date"]
            if len(selected_home_dates):
                latest_dates.append(selected_home_dates.max())
        if not away_history.empty:
            selected_away_dates = away_history[away_history["season"].astype(str) == str(away_season)]["date"]
            if len(selected_away_dates):
                latest_dates.append(selected_away_dates.max())

        use_model_prediction = bool(latest_dates) and not home_history.empty and not away_history.empty
        try:
            if use_model_prediction:
                cutoff = max(latest_dates)
                prediction_history = pd.concat([home_history, away_history], ignore_index=True, sort=False)
                prediction_history = prediction_history[prediction_history["date"] <= cutoff].copy()
                prediction = predict_match(model, prediction_history, home_team, away_team, metrics)
                prediction["method"] = "Historical match model"
            else:
                prediction = squad_prediction(home_team, away_team, home_league, away_league, home_season, away_season)
        except Exception:
            prediction = squad_prediction(home_team, away_team, home_league, away_league, home_season, away_season)

        outcome = prediction["prediction"]
        confidence = prediction["confidence"]
        probabilities = prediction["probabilities"]
        feature_row = prediction.get("explanation_features", prediction["features"]).iloc[0]
        method = prediction.get("method", "Historical match model")

        def feature(name, default=0.0):
            return float(feature_row[name]) if name in feature_row.index else default

        if confidence >= 0.60:
            confidence_label = "High"
        elif confidence >= 0.45:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"

        st.divider()
        insight_card("", f"Prediction: <b>{outcome}</b> with <b>{confidence_label}</b> confidence ({confidence:.0%}). Method: <b>{method}</b>.")
        st.subheader("Outcome Probabilities")

        home_probability = probabilities.loc[probabilities["Outcome"] == "Home Win", "Probability"].sum()
        draw_probability = probabilities.loc[probabilities["Outcome"] == "Draw", "Probability"].sum()
        away_probability = probabilities.loc[probabilities["Outcome"] == "Away Win", "Probability"].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric(" Home Win", f"{home_probability:.0%}")
        c2.metric(" Draw", f"{draw_probability:.0%}")
        c3.metric(" Away Win", f"{away_probability:.0%}")

        fig = go.Figure(
            go.Bar(
                x=probabilities["Outcome"],
                y=probabilities["Probability"],
                marker=dict(color=["#0071e3", "#ff9f0a", "#ff3b30"], line=dict(width=0)),
                text=[f"{value:.0%}" for value in probabilities["Probability"]],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            )
        )
        fig.update_layout(**BASE_LAYOUT, height=420, yaxis_title="Probability")
        fig.update_yaxes(tickformat=".0%", range=[0, max(probabilities["Probability"].max() + 0.15, 0.6)])
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("Why this prediction?")
        if method == "Historical match model":
            reasons = []
            elo_diff = feature("elo_diff")
            home_points_form5 = feature("home_points_form5")
            away_points_form5 = feature("away_points_form5")
            home_goals_form = feature("home_goals_form")
            away_goals_form = feature("away_goals_form")
            home_conceded_form = feature("home_conceded_form")
            away_conceded_form = feature("away_conceded_form")

            if elo_diff > 75:
                reasons.append(f" {home_team} have the stronger Elo rating advantage.")
            elif elo_diff < -75:
                reasons.append(f" {away_team} have the stronger Elo rating advantage.")
            else:
                reasons.append(" The Elo ratings are fairly close, so the model sees this as competitive.")

            if home_points_form5 > away_points_form5:
                reasons.append(f" {home_team} have better recent points form.")
            elif away_points_form5 > home_points_form5:
                reasons.append(f" {away_team} have better recent points form.")

            if home_goals_form > away_goals_form:
                reasons.append(f" {home_team} have stronger recent scoring form.")
            elif away_goals_form > home_goals_form:
                reasons.append(f" {away_team} have stronger recent scoring form.")

            if home_conceded_form < away_conceded_form:
                reasons.append(f" {home_team} have conceded fewer goals recently.")
            elif away_conceded_form < home_conceded_form:
                reasons.append(f" {away_team} have conceded fewer goals recently.")

            if confidence < 0.45:
                reasons.append(" The model confidence is low, so this should be treated as a close match.")
        else:
            home_strength = feature("home_squad_strength")
            away_strength = feature("away_squad_strength")
            reasons = [
                f" {home_team} squad strength score: {home_strength:.2f}.",
                f" {away_team} squad strength score: {away_strength:.2f}.",
                " This estimate uses squad-level player data because full match-history data is not available for one or both selections.",
            ]

        for reason in reasons:
            st.markdown(reason)

        st.divider()
        st.subheader("Feature Snapshot")

        if method == "Historical match model":
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
        else:
            feature_display = pd.DataFrame(
                {
                    "Feature": ["Home Squad Strength", "Away Squad Strength", "Strength Difference"],
                    "Value": [feature("home_squad_strength"), feature("away_squad_strength"), feature("strength_difference")],
                }
            )

        st.dataframe(feature_display, use_container_width=True, hide_index=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Recent Form: {home_team}")
            home_context = matches[(matches["league"].astype(str) == str(home_league)) & (matches["season"].astype(str) == str(home_season))].copy()
            recent_home = recent_team_matches(home_context, home_team, 5)
            if recent_home.empty:
                recent_home = recent_team_matches(matches, home_team, 5)
            render_form_badges(home_team, recent_home)
            st.subheader(f"Recent Matches: {home_team}")
            if recent_home.empty:
                st.info("No recent match rows available for this team.")
            else:
                st.dataframe(recent_home[["date", "league", "home_team", "away_team", "home_goals", "away_goals"]], use_container_width=True, hide_index=True)

        with col2:
            st.subheader(f"Recent Form: {away_team}")
            away_context = matches[(matches["league"].astype(str) == str(away_league)) & (matches["season"].astype(str) == str(away_season))].copy()
            recent_away = recent_team_matches(away_context, away_team, 5)
            if recent_away.empty:
                recent_away = recent_team_matches(matches, away_team, 5)
            render_form_badges(away_team, recent_away)
            st.subheader(f"Recent Matches: {away_team}")
            if recent_away.empty:
                st.info("No recent match rows available for this team.")
            else:
                st.dataframe(recent_away[["date", "league", "home_team", "away_team", "home_goals", "away_goals"]], use_container_width=True, hide_index=True)

elif page == "  Player Stats":
    st.markdown("# Player Statistics")
    st.markdown("Filter players by league, season, position, team and minutes.")
    st.divider()

    if players is None:
        st.error("Player data is not available. Add data/processed/player_stats.csv first.")
        st.stop()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        league_filter = st.selectbox("League", PLAYER_LEAGUES, key="players_league", format_func=display_competition)

    league_pool = filter_players(players, league_filter, None)
    season_options = player_seasons_for(league_pool)
    if not season_options:
        st.warning("No seasons available for the selected league.")
        st.stop()

    with col2:
        season_filter = st.selectbox("Season", season_options, index=0, key=f"players_season_{league_filter}")

    filtered = filter_players(players, league_filter, season_filter)

    with col3:
        positions = ["All"] + sorted(filtered["position"].dropna().astype(str).unique().tolist())
        pos_filter = st.selectbox("Position", positions, key="players_position")

    with col4:
        teams_list = ["All"] + sorted(filtered["team"].dropna().astype(str).unique().tolist())
        team_filter = st.selectbox("Team", teams_list, key="players_team")

    with col5:
        max_minutes = int(max(filtered["minutes"].max(), 90)) if len(filtered) else 90
        min_minutes = st.number_input("Min Minutes", min_value=0, max_value=max_minutes, value=0, step=90, key="players_min_minutes")

    with col6:
        stat_view = st.selectbox("Stat View", ["Raw Totals", "Per 90"], key="players_stat_view")

    if pos_filter != "All":
        filtered = filtered[filtered["position"].astype(str) == pos_filter]
    if team_filter != "All":
        filtered = filtered[filtered["team"].astype(str) == team_filter]
    filtered = filtered[filtered["minutes"] >= min_minutes]

    search = st.text_input(" Search player", placeholder="Example: Salah, Haaland, Palmer")
    if search:
        filtered = filtered[filtered["name"].astype(str).str.contains(search, case=False, na=False)]

    st.markdown(f"**{len(filtered):,} players found**")
    if filtered.empty:
        st.warning("No players match these filters.")
        st.stop()

    min_450 = filtered[filtered["minutes"] >= 450]
    top_scorer = filtered.loc[filtered["goals"].idxmax()]
    top_assister = filtered.loc[filtered["assists"].idxmax()]
    top_p90 = min_450.loc[min_450["goals_p90"].idxmax()] if len(min_450) else filtered.loc[filtered["goals_p90"].idxmax()]

    insight_card("", f"<b>{top_scorer['name']}</b> leads this selection with <b>{int(top_scorer['goals'])}</b> goals.")
    insight_card("", f"<b>{top_assister['name']}</b> leads this selection with <b>{int(top_assister['assists'])}</b> assists.")
    insight_card("", f"<b>{top_p90['name']}</b> has the best goals per 90 in this selection: <b>{top_p90['goals_p90']:.2f}</b>.")
    st.divider()

    if search and len(filtered) == 1:
        player = filtered.iloc[0]
        st.subheader(f" {player['name']} · {player['team']}")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Goals", int(player["goals"]))
        p2.metric("Assists", int(player["assists"]))
        p3.metric("Goals/90", f"{player['goals_p90']:.2f}")
        p4.metric("Assists/90", f"{player['assists_p90']:.2f}")
        p5.metric("Minutes", int(player["minutes"]))
        p6.metric("Performance Score", f"{player['performance_score']:.0f}/100")
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
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Assisters")
        st.dataframe(filtered.nlargest(10, assist_col)[["name", "team", "competition", "season", assist_col]], use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Most Booked")
        booked = filtered.copy()
        booked["total_cards"] = booked["yellow_cards"] + booked["red_cards"]
        st.dataframe(booked.nlargest(10, "total_cards")[["name", "team", "competition", "season", "yellow_cards", "red_cards", "total_cards"]], use_container_width=True, hide_index=True)

    st.subheader("Defensive Leaders")
    defensive_leader_cols = [column for column in ["name", "team", "competition", "position", "season", "tackles", "tackles_won", "interceptions", "blocks", "clearances", "defensive_score"] if column in filtered.columns]
    defensive_sort = "defensive_score" if "defensive_score" in filtered.columns else "interceptions"
    st.dataframe(filtered.nlargest(10, defensive_sort)[defensive_leader_cols], use_container_width=True, hide_index=True)

    st.subheader("Full Stats Table")
    if stat_view == "Per 90":
        display_cols = ["name", "team", "competition", "position", "season", "goals_p90", "assists_p90", "shots_p90", "sot_p90", "tackles_p90", "interc_p90", "blocks_p90", "clearances_p90", "contrib_p90", "defensive_score", "performance_score"]
    else:
        display_cols = ["name", "team", "competition", "position", "season", "goals", "assists", "appearances", "starts", "minutes", "shots_on_target", "tackles", "tackles_won", "interceptions", "blocks", "clearances", "defensive_score", "performance_score"]
    st.dataframe(filtered[display_cols].sort_values(goal_col, ascending=False), use_container_width=True, hide_index=True)

elif page == "  Player Comparison":
    st.markdown("# Player Comparison")
    st.markdown("Compare players across leagues, teams and seasons using easier head-to-head categories.")
    st.divider()

    if players is None:
        st.error("Player data is not available. Add data/processed/player_stats.csv first.")
        st.stop()

    def comparison_pool_for(league_value: str, season_value: str, team_value: str) -> pd.DataFrame:
        selected_season = None if season_value == "All Seasons" else season_value
        pool = filter_players(players, league_value, selected_season)
        if team_value != "All":
            pool = pool[pool["team"].astype(str) == str(team_value)]
        return pool[pool["minutes"] > 0].copy()

    def safe_number(row: pd.Series, column: str) -> float:
        if column not in row.index:
            return 0.0
        return float(pd.to_numeric(row[column], errors="coerce") if pd.notna(pd.to_numeric(row[column], errors="coerce")) else 0.0)

    def player_line(row: pd.Series) -> str:
        return f"{row['team']} · {display_competition(row['competition'])} · {row['season']} · {row['position']} · {int(row['minutes'])} mins"

    filter_left, filter_right = st.columns(2)
    with filter_left:
        st.markdown(
            """
            <div style='border-radius: 20px; padding: 24px; background: #f7fafc;'>
                <h3 style='margin: 0 0 12px 0;'>Player 1</h3>
                <p style='color: #52525c; margin: 0 0 18px 0;'>Choose league, season, team, and player for the first comparison profile.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        first_league = st.selectbox("League", PLAYER_LEAGUES, key="compare_league_1", format_func=display_competition)
        first_league_pool = filter_players(players, first_league, None)
        first_seasons = ["All Seasons"] + player_seasons_for(first_league_pool)
        first_season = st.selectbox("Season", first_seasons, index=1 if len(first_seasons) > 1 else 0, key=f"compare_season_1_{first_league}")
        first_team_pool = filter_players(players, first_league, None if first_season == "All Seasons" else first_season)
        first_teams = ["All"] + sorted(first_team_pool["team"].dropna().astype(str).unique().tolist())
        first_team = st.selectbox("Team", first_teams, key=f"compare_team_1_{first_league}_{first_season}")
        first_pool = comparison_pool_for(first_league, first_season, first_team)
        if first_pool.empty:
            st.warning("No players found for Player 1 filters.")
            st.stop()
        first_names = sorted(first_pool["name"].dropna().astype(str).unique().tolist())
        first_name = st.selectbox("Player", first_names, index=0, key=f"compare_player_1_{first_league}_{first_season}_{first_team}")

    with filter_right:
        st.markdown(
            """
            <div style='border-radius: 20px; padding: 24px; background: #f7fafc;'>
                <h3 style='margin: 0 0 12px 0;'>Player 2</h3>
                <p style='color: #52525c; margin: 0 0 18px 0;'>Choose league, season, team, and player for the second comparison profile.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        second_league = st.selectbox("League", PLAYER_LEAGUES, key="compare_league_2", format_func=display_competition)
        second_league_pool = filter_players(players, second_league, None)
        second_seasons = ["All Seasons"] + player_seasons_for(second_league_pool)
        second_season = st.selectbox("Season", second_seasons, index=1 if len(second_seasons) > 1 else 0, key=f"compare_season_2_{second_league}")
        second_team_pool = filter_players(players, second_league, None if second_season == "All Seasons" else second_season)
        second_teams = ["All"] + sorted(second_team_pool["team"].dropna().astype(str).unique().tolist())
        second_team = st.selectbox("Team", second_teams, key=f"compare_team_2_{second_league}_{second_season}")
        second_pool = comparison_pool_for(second_league, second_season, second_team)
        if second_pool.empty:
            st.warning("No players found for Player 2 filters.")
            st.stop()
        second_names = sorted(second_pool["name"].dropna().astype(str).unique().tolist())
        second_default = 1 if len(second_names) > 1 else 0
        second_name = st.selectbox("Player", second_names, index=second_default, key=f"compare_player_2_{second_league}_{second_season}_{second_team}")

    first_player = first_pool[first_pool["name"].astype(str) == first_name].sort_values(["season", "minutes"], ascending=[False, False]).iloc[0]
    second_player = second_pool[second_pool["name"].astype(str) == second_name].sort_values(["season", "minutes"], ascending=[False, False]).iloc[0]

    benchmark_parts = []
    for league_value, season_value in [(first_league, first_season), (second_league, second_season)]:
        season_value = None if season_value == "All Seasons" else season_value
        benchmark_parts.append(filter_players(players, league_value, season_value))
    benchmark_pool = pd.concat(benchmark_parts, ignore_index=True).drop_duplicates(subset=["name", "team", "competition", "season"])
    benchmark_pool = benchmark_pool[benchmark_pool["minutes"] > 0].copy()
    if benchmark_pool.empty:
        benchmark_pool = pd.concat([first_pool, second_pool], ignore_index=True).drop_duplicates(subset=["name", "team", "competition", "season"])

    st.divider()
    st.markdown("## Key player scores")
    score_cols = st.columns(4)
    score_cols[0].metric("Performance", f"{safe_number(first_player, 'performance_score'):.0f}/100", f"{safe_number(second_player, 'performance_score'):.0f}/100")
    score_cols[1].metric("Attack", f"{safe_number(first_player, 'attacking_score'):.0f}/100", f"{safe_number(second_player, 'attacking_score'):.0f}/100")
    score_cols[2].metric("Creativity", f"{safe_number(first_player, 'creative_score'):.0f}/100", f"{safe_number(second_player, 'creative_score'):.0f}/100")
    score_cols[3].metric("Defence", f"{safe_number(first_player, 'defensive_score'):.0f}/100", f"{safe_number(second_player, 'defensive_score'):.0f}/100")

    st.divider()
    st.subheader("Category Radar")
    st.caption("The radar shows the relative category performance for each player.")
    radar_cols = ["performance_score", "attacking_score", "creative_score", "defensive_score", "goals_p90", "assists_p90", "tackles_p90", "interc_p90"]
    radar_labels = ["Overall", "Attack", "Creativity", "Defence", "Goals/90", "Assists/90", "Tackles/90", "Interceptions/90"]
    radar_cols = [column for column in radar_cols if column in benchmark_pool.columns]
    radar_labels = [label for column, label in zip(["performance_score", "attacking_score", "creative_score", "defensive_score", "goals_p90", "assists_p90", "tackles_p90", "interc_p90"], ["Overall", "Attack", "Creativity", "Defence", "Goals/90", "Assists/90", "Tackles/90", "Interceptions/90"]) if column in radar_cols]

    def benchmark_score(player_row: pd.Series, col: str) -> float:
        values = pd.to_numeric(benchmark_pool[col], errors="coerce").fillna(0)
        value = safe_number(player_row, col)
        if values.nunique() <= 1:
            return 50.0
        return float((values <= value).mean() * 100)

    first_values = [benchmark_score(first_player, col) for col in radar_cols]
    second_values = [benchmark_score(second_player, col) for col in radar_cols]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=first_values + [first_values[0]], theta=radar_labels + [radar_labels[0]], fill="toself", fillcolor="rgba(0,113,227,0.15)", line=dict(color="#0071e3", width=2.5), name=f"{first_name} {first_player['season']}"))
    fig.add_trace(go.Scatterpolar(r=second_values + [second_values[0]], theta=radar_labels + [radar_labels[0]], fill="toself", fillcolor="rgba(255,59,48,0.15)", line=dict(color="#ff3b30", width=2.5), name=f"{second_name} {second_player['season']}"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color="#6e6e73"), gridcolor="#e0e0e5"), angularaxis=dict(tickfont=dict(size=12, color="#1d1d1f"), gridcolor="#e0e0e5")),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader("Head-to-Head Breakdown")
    st.caption("This comparison shows the category values, who leads each metric, and provides the full table when needed.")
    categories = [
        ("Overall", "performance_score", "/100"),
        ("Attacking Score", "attacking_score", "/100"),
        ("Creative Score", "creative_score", "/100"),
        ("Defensive Score", "defensive_score", "/100"),
        ("Goals", "goals", ""),
        ("Assists", "assists", ""),
        ("Goals/90", "goals_p90", ""),
        ("Assists/90", "assists_p90", ""),
        ("Shots on Target", "shots_on_target", ""),
        ("Tackles", "tackles", ""),
        ("Tackles Won", "tackles_won", ""),
        ("Interceptions", "interceptions", ""),
        ("Blocks", "blocks", ""),
        ("Clearances", "clearances", ""),
        ("Errors", "errors", "lower"),
    ]
    breakdown_rows = []
    for label, column, suffix in categories:
        if column not in players.columns:
            continue
        first_value = safe_number(first_player, column)
        second_value = safe_number(second_player, column)
        if suffix == "lower":
            if first_value < second_value:
                advantage = first_name
            elif second_value < first_value:
                advantage = second_name
            else:
                advantage = "Even"
        else:
            if first_value > second_value:
                advantage = first_name
            elif second_value > first_value:
                advantage = second_name
            else:
                advantage = "Even"
        breakdown_rows.append({"Category": label, first_name: round(first_value, 2), second_name: round(second_value, 2), "Advantage": advantage})

    breakdown = pd.DataFrame(breakdown_rows)
    st.write(f"**{first_name}** leads in **{(breakdown['Advantage'] == first_name).sum()}** metrics; **{second_name}** leads in **{(breakdown['Advantage'] == second_name).sum()}**.")
    with st.expander("View full detailed comparison table"):
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    first_wins = (breakdown["Advantage"] == first_name).sum()
    second_wins = (breakdown["Advantage"] == second_name).sum()
    if first_wins > second_wins:
        insight_card("", f"<b>{first_name}</b> leads in <b>{first_wins}</b> categories, while <b>{second_name}</b> leads in <b>{second_wins}</b>.")
    elif second_wins > first_wins:
        insight_card("", f"<b>{second_name}</b> leads in <b>{second_wins}</b> categories, while <b>{first_name}</b> leads in <b>{first_wins}</b>.")
    else:
        insight_card("", "These players are evenly matched across the selected categories.")

elif page == "  Team Analysis":
    st.markdown("# Team Analysis")
    st.markdown("Analyse teams by league and season, then generate a predicted XI from the best available players in that squad.")
    st.divider()

    league_options = overview_league_values()
    
    col_league, col_season, col_team, col_formation = st.columns(4)
    
    with col_league:
        league = st.selectbox("League", league_options, key="team_league", format_func=display_competition)
    
    player_frame_all = filter_players(players, league, None) if players is not None and league != "All" else pd.DataFrame()
    match_frame_all = filter_matches_by_league(league) if league in LEAGUES else pd.DataFrame()

    if not player_frame_all.empty:
        season_options = player_seasons_for(player_frame_all)
    elif not match_frame_all.empty:
        season_options = match_seasons_for(match_frame_all)
    else:
        season_options = []

    if not season_options:
        st.warning("No team data available for this selection.")
        st.stop()

    with col_season:
        season = st.selectbox("Season", season_options, index=0, key=f"team_season_{league}")

    player_frame = filter_players(players, league, season) if players is not None and league != "All" else pd.DataFrame()
    match_frame = match_frame_all[match_frame_all["season"].astype(str) == str(season)].copy() if not match_frame_all.empty else pd.DataFrame()

    player_teams = sorted(player_frame["team"].dropna().astype(str).unique().tolist()) if not player_frame.empty and "team" in player_frame.columns else []
    match_teams = available_match_teams(match_frame) if not match_frame.empty else []
    teams = sorted(set(player_teams) | set(match_teams))

    if not teams:
        st.warning("No teams available for this league and season.")
        st.stop()

    with col_team:
        team = st.selectbox("Team", teams, key=f"team_select_{league}_{season}")

    with col_formation:
        formation = st.selectbox("Formation", list(FORMATION_SLOTS.keys()), key=f"predicted_formation_{league}_{season}_{team}")

    squad = player_frame[player_frame["team"].astype(str) == str(team)].copy() if not player_frame.empty else pd.DataFrame()

    st.divider()
    st.subheader(" Predicted Starting XI")
    
    if squad.empty:
        st.info("No squad player data is available for this team and season, so a predicted lineup cannot be generated.")
    else:
        show_predicted_lineup_visual(squad, team, formation)

    st.divider()

    if not squad.empty:
        st.markdown("## Squad Overview")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Squad Size", len(squad))
        kpi_cols[1].metric("Avg Performance", f"{squad['performance_score'].mean():.0f}/100")
        kpi_cols[2].metric("Total Goals", int(squad["goals"].sum()))
        kpi_cols[3].metric("Avg Defence", f"{squad['defensive_score'].mean():.0f}/100")

        st.markdown("### Key Players")
        insight_cols = st.columns(3)
        top_scorer = squad.loc[squad["goals"].idxmax()]
        with insight_cols[0]:
            insight_card("", f"<b>{top_scorer['name']}</b> leads with <b>{int(top_scorer['goals'])}</b> goals.")

        top_creator = squad.loc[squad["assists"].idxmax()]
        with insight_cols[1]:
            insight_card("", f"<b>{top_creator['name']}</b> leads with <b>{int(top_creator['assists'])}</b> assists.")

        if "tackles" in squad.columns and "interceptions" in squad.columns:
            squad["defensive_activity"] = squad["tackles"].fillna(0) + squad["interceptions"].fillna(0)
            top_defender = squad.loc[squad["defensive_activity"].idxmax()]
            with insight_cols[2]:
                insight_card("", f"<b>{top_defender['name']}</b> leads with <b>{int(top_defender['defensive_activity'])}</b> defensive actions.")

        st.divider()

        with st.expander(" View full squad stats"):
            squad_display_cols = [column for column in ["name", "position", "age", "starts", "minutes", "goals", "assists", "tackles", "tackles_won", "interceptions", "blocks", "clearances", "defensive_score", "performance_score"] if column in squad.columns]
            st.dataframe(
                squad.sort_values(["minutes", "performance_score"], ascending=False)[squad_display_cols],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Goals and Assists Distribution")
        top = squad.nlargest(12, "contrib_p90")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top["name"], y=top["goals"], name="Goals", marker=dict(color="#0071e3")))
        fig.add_trace(go.Bar(x=top["name"], y=top["assists"], name="Assists", marker=dict(color="#34c759")))
        fig.update_layout(**BASE_LAYOUT, barmode="group", height=420)
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    if not match_frame.empty:
        team_matches = match_frame[(match_frame["home_team"].astype(str) == str(team)) | (match_frame["away_team"].astype(str) == str(team))].copy()
        if not team_matches.empty:
            home_frame = team_matches[team_matches["home_team"].astype(str) == str(team)].copy()
            away_frame = team_matches[team_matches["away_team"].astype(str) == str(team)].copy()
            home_frame["gf"] = home_frame["home_goals"]
            home_frame["ga"] = home_frame["away_goals"]
            home_frame["venue"] = "Home"
            away_frame["gf"] = away_frame["away_goals"]
            away_frame["ga"] = away_frame["home_goals"]
            away_frame["venue"] = "Away"
            team_results = pd.concat([home_frame, away_frame], ignore_index=True)
            if "date" in team_results.columns:
                team_results = team_results.sort_values("date", na_position="last")
            team_results["win"] = (team_results["gf"] > team_results["ga"]).astype(int)
            team_results["draw"] = (team_results["gf"] == team_results["ga"]).astype(int)
            team_results["loss"] = (team_results["gf"] < team_results["ga"]).astype(int)
            total = len(team_results)
            wins = int(team_results["win"].sum())
            draws = int(team_results["draw"].sum())
            losses = int(team_results["loss"].sum())
            goals_for = int(team_results["gf"].sum())
            goals_against = int(team_results["ga"].sum())

            st.divider()
            st.subheader("Match Results & Performance")
            insight_card("", f"<b>{team}</b>: <b>{wins}W {draws}D {losses}L</b> from {total} matches in {season}. Win rate: <b>{wins / total:.0%}</b>.")
            insight_card("", f"Scored <b>{goals_for}</b>, conceded <b>{goals_against}</b>, goal difference <b>{goals_for - goals_against:+}</b>.")

            metric_cols = st.columns(5)
            metric_cols[0].metric("Games", total)
            metric_cols[1].metric("Wins", wins)
            metric_cols[2].metric("Draws", draws)
            metric_cols[3].metric("Goals For", goals_for)
            metric_cols[4].metric("Goals Against", goals_against)

            st.markdown("### Recent Matches")
            recent = team_results.tail(10)[["date", "venue", "home_team", "away_team", "home_goals", "away_goals"]]
            st.dataframe(recent, use_container_width=True, hide_index=True)

            st.markdown("### Goal Trend")
            trend = team_results.tail(15).copy()
            trend["match"] = np.arange(1, len(trend) + 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["match"], y=trend["gf"], mode="lines+markers", name="Goals For", line=dict(color="#0071e3", width=3)))
            fig.add_trace(go.Scatter(x=trend["match"], y=trend["ga"], mode="lines+markers", name="Goals Against", line=dict(color="#ff3b30", width=3)))
            fig.update_layout(**BASE_LAYOUT, height=420, xaxis_title="Recent Match", yaxis_title="Goals")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

elif page == "  Transfer Analysis":
    st.markdown("# Transfer & Scouting")
    st.markdown("Find hidden gems, analyse team weaknesses, score role fit and build recruitment shortlists.")
    st.divider()

    if players is None:
        st.error("Player data is not available. Add data/processed/player_stats.csv first.")
        st.stop()

    role_profiles = {
        "Advanced Forward": {"goals_p90": 3.0, "shots_p90": 1.7, "sot_p90": 1.7, "contrib_p90": 1.0, "performance_score": 0.8},
        "Poacher": {"goals_p90": 3.5, "sot_p90": 2.0, "shots_p90": 1.5, "performance_score": 0.6},
        "Complete Forward": {"goals_p90": 2.4, "assists_p90": 1.4, "shots_p90": 1.2, "contrib_p90": 1.6, "performance_score": 1.0},
        "False 9": {"assists_p90": 2.0, "key_passes": 1.8, "progressive_passes": 1.4, "contrib_p90": 1.2, "performance_score": 1.0},
        "Target Forward": {"goals_p90": 2.0, "duels_won": 1.5, "assists_p90": 0.8, "performance_score": 1.0},
        "Pressing Forward": {"goals_p90": 1.8, "tackles_p90": 1.2, "contrib_p90": 1.1, "performance_score": 0.9},
        "Deep Lying Forward": {"assists_p90": 1.8, "key_passes": 1.5, "contrib_p90": 1.4, "performance_score": 1.0},
        "Inside Forward": {"goals_p90": 2.4, "shots_p90": 1.7, "dribbles": 1.4, "contrib_p90": 1.2, "performance_score": 0.8},
        "Inverted Winger": {"assists_p90": 1.8, "key_passes": 1.8, "dribbles": 1.5, "progressive_carries": 1.2, "performance_score": 0.8},
        "Winger": {"assists_p90": 1.8, "crosses": 1.8, "dribbles": 1.5, "key_passes": 1.2, "performance_score": 0.8},
        "Wide Playmaker": {"assists_p90": 1.8, "key_passes": 2.0, "progressive_passes": 1.5, "performance_score": 1.0},
        "Advanced Playmaker": {"assists_p90": 1.8, "key_passes": 2.0, "progressive_passes": 1.8, "pass_accuracy": 1.0, "performance_score": 1.0},
        "Deep Lying Playmaker": {"progressive_passes": 2.0, "pass_accuracy": 1.8, "interc_p90": 0.8, "performance_score": 1.0},
        "Box to Box Midfielder": {"tackles_p90": 1.4, "interc_p90": 1.2, "progressive_passes": 1.0, "contrib_p90": 0.8, "performance_score": 1.0},
        "Ball Winning Midfielder": {"tackles_p90": 2.2, "interc_p90": 1.8, "duels_won": 1.2, "performance_score": 0.8},
        "Defensive Midfielder": {"tackles_p90": 1.8, "interc_p90": 1.8, "pass_accuracy": 1.0, "performance_score": 0.8},
        "Anchor": {"interc_p90": 2.0, "tackles_p90": 1.8, "pass_accuracy": 1.0, "performance_score": 0.8},
        "Half Back": {"interc_p90": 1.8, "tackles_p90": 1.4, "pass_accuracy": 1.4, "performance_score": 0.9},
        "Mezzala": {"contrib_p90": 1.1, "progressive_carries": 1.5, "key_passes": 1.2, "performance_score": 1.0},
        "Carrilero": {"tackles_p90": 1.2, "interc_p90": 1.2, "pass_accuracy": 1.2, "performance_score": 0.9},
        "Ball Playing Defender": {"tackles_p90": 1.3, "interc_p90": 1.4, "pass_accuracy": 1.2, "progressive_passes": 1.0, "performance_score": 1.0},
        "Central Defender": {"tackles_p90": 1.8, "interc_p90": 1.8, "duels_won": 1.5, "performance_score": 1.0},
        "No Nonsense Centre Back": {"tackles_p90": 2.0, "interc_p90": 1.8, "duels_won": 1.8, "performance_score": 0.8},
        "Libero": {"progressive_passes": 1.8, "pass_accuracy": 1.4, "interc_p90": 1.0, "performance_score": 1.0},
        "Full Back": {"tackles_p90": 1.5, "interc_p90": 1.2, "crosses": 1.0, "assists_p90": 0.8, "performance_score": 0.8},
        "Wing Back": {"crosses": 1.8, "assists_p90": 1.2, "tackles_p90": 1.0, "progressive_carries": 1.0, "performance_score": 0.8},
        "Inverted Full Back": {"pass_accuracy": 1.6, "progressive_passes": 1.4, "tackles_p90": 1.2, "interc_p90": 1.0, "performance_score": 0.8},
        "Sweeper Keeper": {"saves": 2.0, "pass_accuracy": 1.5, "clean_sheets": 1.2, "performance_score": 1.0},
        "Shot Stopper": {"saves": 2.4, "clean_sheets": 1.2, "goals_against": -1.0, "performance_score": 1.0},
    }

    role_groups = {
        "Goalkeeper": ["Sweeper Keeper", "Shot Stopper"],
        "Centre Back": ["Central Defender", "Ball Playing Defender", "No Nonsense Centre Back", "Libero"],
        "Full Back / Wing Back": ["Full Back", "Wing Back", "Inverted Full Back"],
        "Defensive Midfield": ["Defensive Midfielder", "Anchor", "Half Back", "Ball Winning Midfielder"],
        "Central Midfield": ["Box to Box Midfielder", "Advanced Playmaker", "Deep Lying Playmaker", "Mezzala", "Carrilero"],
        "Wide Forward / Winger": ["Winger", "Inverted Winger", "Inside Forward", "Wide Playmaker"],
        "Striker": ["Advanced Forward", "Poacher", "Complete Forward", "False 9", "Target Forward", "Pressing Forward", "Deep Lying Forward"],
    }

    role_position_patterns = {
        "Goalkeeper": "GK|keeper",
        "Centre Back": "CB|DF|defender|centre|center|back",
        "Full Back / Wing Back": "FB|WB|LB|RB|DF|back",
        "Defensive Midfield": "DM|MF|mid",
        "Central Midfield": "CM|MF|mid",
        "Wide Forward / Winger": "LW|RW|AM|FW|wing|attacker|forward",
        "Striker": "ST|CF|FW|forward|striker|attacker",
    }

    tab1, tab2, tab3, tab4, tab5 = st.tabs([" Player Scouting", " Hidden Gems", " Role Fit", " Team Weaknesses", " Attack vs Defence"])

    with tab1:
        st.subheader("Recruitment Shortlist Builder")
        st.markdown("Filter players by league and recruitment criteria to build a ranked shortlist.")
        st.divider()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            scout_league = st.selectbox("League", PLAYER_LEAGUES, key="scout_league", format_func=display_competition)
            scout_pool = latest_player_pool(scout_league)
            scout_positions = ["All"] + sorted(scout_pool["position"].dropna().astype(str).unique().tolist()) if len(scout_pool) else ["All"]
            scout_position = st.selectbox("Position", scout_positions, key="scout_position")
        with col2:
            min_age = st.number_input("Min Age", min_value=0, max_value=50, value=18, key="scout_min_age")
            max_age = st.number_input("Max Age", min_value=0, max_value=50, value=28, key="scout_max_age")
            min_minutes = st.number_input("Min Minutes Played", min_value=0, max_value=int(max(scout_pool["minutes"].max(), 90)) if len(scout_pool) else 90, value=90, step=90, key="scout_min_minutes")
        with col3:
            min_goals_p90 = st.number_input("Min Goals/90", min_value=0.0, max_value=5.0, value=0.0, step=0.05, key="scout_min_goals")
            min_assists_p90 = st.number_input("Min Assists/90", min_value=0.0, max_value=5.0, value=0.0, step=0.05, key="scout_min_assists")
            min_rating = st.number_input("Min Performance Score", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="scout_min_rating")
        with col4:
            scout_teams = ["All"] + sorted(scout_pool["team"].dropna().astype(str).unique().tolist())
            scout_team = st.selectbox("Team", scout_teams, key="scout_team")
            scout_nations = ["All"] + sorted(scout_pool["nationality"].dropna().astype(str).unique().tolist())
            scout_nation = st.selectbox("Nationality", scout_nations, key="scout_nation")

        if st.button("Build Shortlist →", key="build_shortlist"):
            shortlist = scout_pool.copy()
            if scout_position != "All":
                shortlist = shortlist[shortlist["position"].astype(str) == scout_position]
            if scout_team != "All":
                shortlist = shortlist[shortlist["team"].astype(str) == scout_team]
            if scout_nation != "All":
                shortlist = shortlist[shortlist["nationality"].astype(str) == scout_nation]
            shortlist = shortlist[(shortlist["age"] >= min_age) & (shortlist["age"] <= max_age) & (shortlist["minutes"] >= min_minutes) & (shortlist["goals_p90"] >= min_goals_p90) & (shortlist["assists_p90"] >= min_assists_p90) & (shortlist["performance_score"] >= min_rating)].copy()
            if shortlist.empty:
                st.warning("No players match your criteria. Relax the filters slightly.")
            else:
                score_cols = ["goals_p90", "assists_p90", "sot_p90", "tackles_p90", "interc_p90", "performance_score", "contrib_p90"]
                if MinMaxScaler is not None and len(shortlist) > 1:
                    scaled = MinMaxScaler().fit_transform(shortlist[score_cols].fillna(0))
                    shortlist["score"] = (scaled.mean(axis=1) * 100).round(1)
                else:
                    shortlist["score"] = shortlist[score_cols].fillna(0).mean(axis=1).round(1)
                shortlist = shortlist.sort_values("score", ascending=False)
                display_cols = ["name", "team", "competition", "age", "position", "goals_p90", "assists_p90", "performance_score", "minutes", "score"]
                insight_card("", f"Found <b>{len(shortlist)}</b> players matching your criteria.")
                st.dataframe(shortlist[display_cols], use_container_width=True, hide_index=True)
                csv = shortlist[display_cols].to_csv(index=False)
                st.download_button("⬇ Download Shortlist CSV", data=csv, file_name="recruitment_shortlist.csv", mime="text/csv")
                top = shortlist.iloc[0]
                insight_card("⭐", f"Top recommendation: <b>{top['name']}</b> from <b>{top['team']}</b>. Score: <b>{top['score']}</b>/100.")

    with tab2:
        st.subheader("Hidden Gems Finder")
        st.markdown("Find young, efficient or under-used players using the latest available player dataset.")
        st.divider()
        gems_league = st.selectbox("League", PLAYER_LEAGUES, key="gems_league", format_func=display_competition)
        gems_pool = latest_player_pool(gems_league)
        gems_pool = gems_pool[gems_pool["minutes"] >= 90].copy()
        if gems_pool.empty:
            st.warning("No players found for this league.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Best Under 23s**")
                under_23 = gems_pool[gems_pool["age"] <= 23].copy()
                under_23["score"] = (under_23["goals_p90"] * 2 + under_23["assists_p90"] * 1.5 + under_23["performance_score"] / 10).round(2)
                st.dataframe(under_23.nlargest(10, "score")[["name", "team", "competition", "age", "goals_p90", "assists_p90", "performance_score", "score"]], use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**High Output Low Minutes**")
                hidden = gems_pool[gems_pool["minutes"] <= 900].copy()
                hidden["score"] = (hidden["goals_p90"] * 2 + hidden["assists_p90"] * 1.5 + hidden["contrib_p90"]).round(2)
                st.dataframe(hidden.nlargest(10, "score")[["name", "team", "competition", "minutes", "goals_p90", "assists_p90", "performance_score", "score"]], use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Role Fit Analysis")
        st.markdown("Score players against role profiles using output, technical contribution and defensive activity.")
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            role_league = st.selectbox("League", PLAYER_LEAGUES, key="role_league", format_func=display_competition)
        with col2:
            role_position = st.selectbox("Position", list(role_groups.keys()), key="role_position")
        with col3:
            selected_role = st.selectbox("Role", role_groups[role_position], key=f"role_select_{role_position}")
        with col4:
            role_min_minutes = st.number_input("Min Minutes", min_value=0, max_value=int(max(players["minutes"].max(), 90)), value=450, step=90, key="role_min_minutes")

        role_pool = latest_player_pool(role_league)
        pattern = role_position_patterns.get(role_position, "")
        if pattern:
            role_pool = role_pool[role_pool["position"].astype(str).str.contains(pattern, case=False, na=False)].copy()
        role_pool = role_pool[role_pool["minutes"] >= role_min_minutes].copy()
        if role_pool.empty:
            st.warning("No players found for this role filter.")
        else:
            weights = role_profiles[selected_role]
            role_score = pd.Series(0.0, index=role_pool.index)
            total_weight = 0.0
            for column, weight in weights.items():
                if column not in role_pool.columns:
                    continue
                values = pd.to_numeric(role_pool[column], errors="coerce").fillna(0)
                normalised = (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else pd.Series(0.0, index=role_pool.index)
                role_score += normalised * weight
                total_weight += weight
            role_pool["role_fit"] = ((role_score / total_weight) * 100).round(1) if total_weight else 0
            role_pool["bargain_score"] = (role_pool["role_fit"] * 0.65 + (100 - role_pool["age"].clip(upper=35) * 2) * 0.15 + role_pool["performance_score"] * 0.2).round(1)
            role_pool = role_pool.sort_values(["role_fit", "bargain_score"], ascending=False)
            display_cols = ["name", "team", "competition", "position", "age", "minutes", "performance_score", "role_fit", "bargain_score"]
            st.dataframe(role_pool[display_cols].head(30), use_container_width=True, hide_index=True)
            top = role_pool.iloc[0]
            insight_card("", f"Best role fit for <b>{selected_role}</b>: <b>{top['name']}</b> from <b>{top['team']}</b> with a <b>{top['role_fit']}</b>/100 score.")

    with tab4:
        st.subheader("Team Needs Analysis")
        st.markdown("Select a team to find realistic areas for reinforcement.")
        st.divider()
        needs_leagues = [league for league in MATCH_LEAGUES if str(league).lower() != "international"]
        league_tw = st.selectbox("League", needs_leagues, key="needs_league", format_func=display_competition)
        frame_all = filter_matches_by_league(league_tw)
        season_tw = latest_match_season_for(league_tw)
        frame_tw = frame_all[frame_all["season"].astype(str) == str(season_tw)].copy() if season_tw else pd.DataFrame()
        team_options = available_match_teams(frame_tw)
        if not team_options:
            st.warning("No match data available for this league.")
            st.stop()
        team_tw = st.selectbox("Select Team", team_options, key="needs_team")

        home_tw = frame_tw[frame_tw["home_team"].astype(str) == team_tw]
        away_tw = frame_tw[frame_tw["away_team"].astype(str) == team_tw]
        avg_scored = (safe_mean(home_tw["home_goals"], 0.0) + safe_mean(away_tw["away_goals"], 0.0)) / 2
        avg_conceded = (safe_mean(home_tw["away_goals"], 0.0) + safe_mean(away_tw["home_goals"], 0.0)) / 2
        league_avg_scored = frame_tw["total_goals"].mean() / 2
        league_avg_conceded = frame_tw["total_goals"].mean() / 2
        attack_diff = avg_scored - league_avg_scored
        defence_diff = avg_conceded - league_avg_conceded
        all_team_results = pd.concat([home_tw.assign(gf=home_tw["home_goals"], ga=home_tw["away_goals"]), away_tw.assign(gf=away_tw["away_goals"], ga=away_tw["home_goals"])], ignore_index=True)
        home_wr = (home_tw["result"] == 1).mean() if len(home_tw) else 0
        away_wr = (away_tw["result"] == -1).mean() if len(away_tw) else 0
        clean_sheet_rate = (all_team_results["ga"] == 0).mean() if len(all_team_results) else 0
        fail_score_rate = (all_team_results["gf"] == 0).mean() if len(all_team_results) else 0

        weaknesses = []
        strengths = []
        if attack_diff < -0.2:
            weaknesses.append(f" **Weak attack**: scoring {avg_scored:.2f} goals per game vs league average {league_avg_scored:.2f}.")
        elif attack_diff > 0.2:
            strengths.append(f" **Strong attack**: scoring {avg_scored:.2f} goals per game vs league average {league_avg_scored:.2f}.")
        if defence_diff > 0.2:
            weaknesses.append(f" **Leaky defence**: conceding {avg_conceded:.2f} goals per game vs league average {league_avg_conceded:.2f}.")
        elif defence_diff < -0.2:
            strengths.append(f" **Solid defence**: conceding {avg_conceded:.2f} goals per game vs league average {league_avg_conceded:.2f}.")
        if home_wr < 0.35:
            weaknesses.append(f" **Poor home form**: winning only {home_wr:.0%} of home games.")
        if away_wr < 0.25:
            weaknesses.append(f" **Poor away form**: winning only {away_wr:.0%} of away games.")
        if fail_score_rate > 0.25:
            weaknesses.append(f" **Scoring issue**: fails to score in {fail_score_rate:.0%} of games.")
        if clean_sheet_rate < 0.20:
            weaknesses.append(f" **Defensive frailty**: keeps a clean sheet in only {clean_sheet_rate:.0%} of games.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("** Areas Needing Improvement**")
            for item in weaknesses or ["No major weaknesses identified."]:
                st.markdown(item)
        with col2:
            st.markdown("** Team Strengths**")
            for item in strengths or ["No standout strengths identified."]:
                st.markdown(item)

        st.divider()
        st.subheader("💡 Transfer Suggestions")
        latest_player_season = PLAYER_SEASONS[0] if PLAYER_SEASONS else None
        base_candidates = filter_players(players, "All", latest_player_season) if latest_player_season else players.copy()
        suggestion_pool = realistic_recruitment_pool(league_tw, base_candidates)
        if suggestion_pool.empty:
            st.info("No realistic recruitment suggestions are available for this team level with the current player dataset.")
        else:
            if any("attack" in item.lower() or "score" in item.lower() for item in weaknesses):
                insight_card("", f"<b>{team_tw}</b> need attacking reinforcement. Showing realistic targets for their level.")
                striker_targets = suggestion_pool[(suggestion_pool["position"].astype(str).str.contains("FW|ST|forward|attacker|striker", case=False, na=False)) & (suggestion_pool["minutes"] >= 90)].nlargest(5, "goals_p90")
                st.dataframe(striker_targets[["name", "team", "competition", "age", "minutes", "goals_p90", "assists_p90", "performance_score"]], use_container_width=True, hide_index=True)
            if any("defence" in item.lower() or "defensive" in item.lower() for item in weaknesses):
                insight_card("", f"<b>{team_tw}</b> need defensive reinforcement. Showing realistic targets for their level.")
                defender_targets = suggestion_pool[(suggestion_pool["position"].astype(str).str.contains("DF|defender|back", case=False, na=False)) & (suggestion_pool["minutes"] >= 90)].nlargest(5, "tackles_p90")
                st.dataframe(defender_targets[["name", "team", "competition", "age", "minutes", "tackles_p90", "interc_p90", "performance_score"]], use_container_width=True, hide_index=True)
            if not weaknesses:
                insight_card("", f"<b>{team_tw}</b> appear balanced. Focus on depth signings rather than emergency starters.")

    with tab5:
        st.subheader("Attack vs Defence")
        st.markdown("Team efficiency across the latest available season for the selected league.")
        st.divider()
        scatter_leagues = [league for league in MATCH_LEAGUES if str(league).lower() != "international"]
        league_scatter = st.selectbox("League", scatter_leagues, key="scatter_league", format_func=display_competition)
        scatter_all = filter_matches_by_league(league_scatter)
        scatter_season = latest_match_season_for(league_scatter)
        frame_scatter = scatter_all[scatter_all["season"].astype(str) == str(scatter_season)].copy() if scatter_season else pd.DataFrame()
        home_stats = frame_scatter.groupby("home_team").agg(home_attack=("home_goals", "mean"), home_defence=("away_goals", "mean"))
        away_stats = frame_scatter.groupby("away_team").agg(away_attack=("away_goals", "mean"), away_defence=("home_goals", "mean"))
        team_stats = pd.DataFrame({"attack": (home_stats["home_attack"] + away_stats["away_attack"]) / 2, "defence": (home_stats["home_defence"] + away_stats["away_defence"]) / 2}).dropna().reset_index()
        team_stats.columns = ["team", "attack", "defence"]
        team_stats["score"] = team_stats["attack"] - team_stats["defence"]
        stop_if_empty(team_stats, "Not enough data to build attack vs defence chart.")
        best_attack = team_stats.loc[team_stats["attack"].idxmax()]
        best_defence = team_stats.loc[team_stats["defence"].idxmin()]
        insight_card("", f"<b>{best_attack['team']}</b> have the strongest attack at <b>{best_attack['attack']:.2f}</b> goals per game.")
        insight_card("", f"<b>{best_defence['team']}</b> have the strongest defence, conceding <b>{best_defence['defence']:.2f}</b> per game.")
        highlight = st.text_input(" Highlight a team", placeholder="Example: Chelsea", key="scatter_search")
        team_stats["highlight"] = team_stats["team"].astype(str).str.contains(highlight, case=False, na=False) if highlight else False
        fig = go.Figure()
        normal = team_stats[~team_stats["highlight"]]
        fig.add_trace(go.Scatter(x=normal["defence"], y=normal["attack"], mode="markers", name="Teams", marker=dict(size=9, color=normal["score"], colorscale=[[0, "#ff3b30"], [0.5, "#ff9f0a"], [1, "#34c759"]], line=dict(width=0)), text=normal["team"], hovertemplate="<b>%{text}</b><br>Attack: %{y:.2f}<br>Defence: %{x:.2f}<extra></extra>"))
        highlighted = team_stats[team_stats["highlight"]]
        if not highlighted.empty:
            fig.add_trace(go.Scatter(x=highlighted["defence"], y=highlighted["attack"], mode="markers+text", name="Highlighted", marker=dict(size=16, color="#0071e3", line=dict(color="white", width=2)), text=highlighted["team"], textposition="top center"))
        fig.update_layout(**BASE_LAYOUT, xaxis_title="Avg Goals Conceded", yaxis_title="Avg Goals Scored", height=520)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 Attack")
            st.dataframe(team_stats.nlargest(10, "attack")[["team", "attack", "defence"]], use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Top 10 Defence")
            st.dataframe(team_stats.nsmallest(10, "defence")[["team", "attack", "defence"]], use_container_width=True, hide_index=True)

elif page == "  Model Performance":
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
        insight_card("", f"The model achieved <b>{accuracy:.1%}</b> accuracy, which is strong for a three outcome football prediction task.")
    elif accuracy >= 0.42:
        insight_card("", f"The model achieved <b>{accuracy:.1%}</b> accuracy. This is realistic for football because draws and away wins are difficult to predict.")
    else:
        insight_card("", f"The model achieved <b>{accuracy:.1%}</b> accuracy. This suggests the feature set or training approach needs improvement.")

    if test_seasons:
        insight_card("", f"The model was tested on recent unseen seasons: <b>{', '.join(test_seasons)}</b>.")

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

        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

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
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
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

        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        top_feature = importance_frame.iloc[0]

        insight_card(
            "",
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
