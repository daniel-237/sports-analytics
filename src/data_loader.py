import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MATCH_DATA_PATH,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    PLAYER_DATA_PATH,
)
from src.utils import (
    apply_column_aliases,
    ensure_columns,
    normalise_columns,
    numeric_columns,
    result_code_from_scores,
)


def load_match_data() -> pd.DataFrame:
    if not MATCH_DATA_PATH.exists():
        raise FileNotFoundError(f"Match data not found at: {MATCH_DATA_PATH}")

    frame = pd.read_csv(MATCH_DATA_PATH)
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

    frame = ensure_columns(
        frame,
        {
            "home_team": "Unknown Home",
            "away_team": "Unknown Away",
            "home_goals": 0,
            "away_goals": 0,
            "league": "Unknown League",
            "season": "Unknown Season",
        },
    )

    frame = numeric_columns(frame, ["home_goals", "away_goals"])

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date", na_position="last").reset_index(drop=True)
    else:
        frame["date"] = pd.NaT

    if "result" not in frame.columns:
        frame["result"] = frame.apply(
            lambda row: result_code_from_scores(row["home_goals"], row["away_goals"]),
            axis=1,
        )
    else:
        frame["result"] = pd.to_numeric(frame["result"], errors="coerce").fillna(0).astype(int)

    frame["total_goals"] = frame["home_goals"] + frame["away_goals"]

    frame = ensure_columns(frame, {feature: 0.0 for feature in MODEL_FEATURES})
    frame = numeric_columns(frame, MODEL_FEATURES)

    return frame


def load_player_data() -> Optional[pd.DataFrame]:
    if not PLAYER_DATA_PATH.exists():
        return None

    frame = pd.read_csv(PLAYER_DATA_PATH)
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

    numeric_player_columns = [
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

    frame = numeric_columns(frame, numeric_player_columns)

    minutes_safe = frame["minutes"].replace(0, np.nan)

    frame["goals_p90"] = (frame["goals"] / minutes_safe * 90).round(2)
    frame["assists_p90"] = (frame["assists"] / minutes_safe * 90).round(2)
    frame["shots_p90"] = (frame["shots_total"] / minutes_safe * 90).round(2)
    frame["sot_p90"] = (frame["shots_on_target"] / minutes_safe * 90).round(2)
    frame["tackles_p90"] = (frame["tackles"] / minutes_safe * 90).round(2)
    frame["interc_p90"] = (frame["interceptions"] / minutes_safe * 90).round(2)
    frame["contrib_p90"] = ((frame["goals"] + frame["assists"]) / minutes_safe * 90).round(2)
    frame["cards_p90"] = ((frame["yellow_cards"] + frame["red_cards"]) / minutes_safe * 90).round(2)

    return frame.fillna(0)


def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)