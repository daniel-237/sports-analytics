from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES
from src.utils import result_code_from_scores


def first_existing_column(frame: pd.DataFrame, options: list[str]) -> str | None:
    for column in options:
        if column in frame.columns:
            return column
    return None


def mean_value(values: list[float], fallback: float = 0.0) -> float:
    if not values:
        return fallback

    value = np.mean(values)

    if pd.isna(value):
        return fallback

    return float(value)


def recent(records: list[dict], size: int) -> list[dict]:
    if not records:
        return []

    return records[-size:]


def streak_value(records: list[dict]) -> int:
    if not records:
        return 0

    latest_result = records[-1]["result"]

    if latest_result == 0:
        return 0

    streak = 0

    for record in reversed(records):
        if record["result"] == latest_result:
            streak += latest_result
        else:
            break

    return streak


def summarise_history(records: list[dict]) -> dict:
    last_5 = recent(records, 5)
    last_10 = recent(records, 10)

    return {
        "goals_form": mean_value([record["gf"] for record in last_5]),
        "conceded_form": mean_value([record["ga"] for record in last_5]),
        "shots_form": mean_value([record["shots"] for record in last_5]),
        "sot_form": mean_value([record["sot"] for record in last_5]),
        "gd_form": mean_value([record["gf"] - record["ga"] for record in last_5]),
        "points_form5": mean_value([record["points"] for record in last_5]),
        "points_form10": mean_value([record["points"] for record in last_10]),
        "win_rate": mean_value([record["result"] == 1 for record in last_5]),
        "cs_rate": mean_value([record["ga"] == 0 for record in last_5]),
        "fts_rate": mean_value([record["gf"] == 0 for record in last_5]),
        "streak": streak_value(records),
    }


def points_from_result(result: int, is_home: bool) -> int:
    if result == 0:
        return 1

    if is_home and result == 1:
        return 3

    if not is_home and result == -1:
        return 3

    return 0


def team_result_from_match(result: int, is_home: bool) -> int:
    if result == 0:
        return 0

    if is_home and result == 1:
        return 1

    if not is_home and result == -1:
        return 1

    return -1


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(home_elo: float, away_elo: float, result: int, k: int = 24, home_advantage: int = 60) -> tuple[float, float]:
    expected_home = elo_expected(home_elo + home_advantage, away_elo)

    if result == 1:
        actual_home = 1.0
    elif result == 0:
        actual_home = 0.5
    else:
        actual_home = 0.0

    home_change = k * (actual_home - expected_home)

    return home_elo + home_change, away_elo - home_change


def safe_number(row: pd.Series, column: str | None) -> float:
    if column is None:
        return 0.0

    value = pd.to_numeric(row.get(column, 0), errors="coerce")

    if pd.isna(value):
        return 0.0

    return float(value)


def build_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    frame = matches.copy()

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date", na_position="last").reset_index(drop=True)

    if "result" not in frame.columns:
        frame["result"] = frame.apply(
            lambda row: result_code_from_scores(row["home_goals"], row["away_goals"]),
            axis=1,
        )

    frame["result"] = pd.to_numeric(frame["result"], errors="coerce").fillna(0).astype(int)

    home_shots_col = first_existing_column(frame, ["home_shots", "hs"])
    away_shots_col = first_existing_column(frame, ["away_shots", "as"])
    home_sot_col = first_existing_column(frame, ["home_sot", "home_shots_on_target", "home_shots_target", "hst"])
    away_sot_col = first_existing_column(frame, ["away_sot", "away_shots_on_target", "away_shots_target", "ast"])

    season_counts = frame.groupby("season").cumcount()
    season_totals = frame.groupby("season")["season"].transform("count").replace(0, 1)
    frame["season_stage"] = (season_counts / season_totals).fillna(0)

    histories = defaultdict(list)
    elos = defaultdict(lambda: 1500.0)
    h2h_records = defaultdict(lambda: {"home": 0, "away": 0, "draw": 0, "total": 0})

    feature_rows = []

    for _, row in frame.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        home_history = histories[home_team]
        away_history = histories[away_team]

        home_summary = summarise_history(home_history)
        away_summary = summarise_history(away_history)

        home_elo = float(elos[home_team])
        away_elo = float(elos[away_team])

        pair = (home_team, away_team)
        h2h = h2h_records[pair]
        h2h_total = h2h["total"]
        h2h_home_rate = h2h["home"] / h2h_total if h2h_total else 0.0

        home_attack_strength = home_summary["goals_form"] / 1.35 if home_summary["goals_form"] else 1.0
        away_attack_strength = away_summary["goals_form"] / 1.35 if away_summary["goals_form"] else 1.0
        home_defence_strength = 1.35 / home_summary["conceded_form"] if home_summary["conceded_form"] else 1.0
        away_defence_strength = 1.35 / away_summary["conceded_form"] if away_summary["conceded_form"] else 1.0

        feature_rows.append(
            {
                "home_goals_form": home_summary["goals_form"],
                "away_goals_form": away_summary["goals_form"],
                "home_conceded_form": home_summary["conceded_form"],
                "away_conceded_form": away_summary["conceded_form"],
                "home_shots_form": home_summary["shots_form"],
                "away_shots_form": away_summary["shots_form"],
                "home_sot_form": home_summary["sot_form"],
                "away_sot_form": away_summary["sot_form"],
                "home_gd_form": home_summary["gd_form"],
                "away_gd_form": away_summary["gd_form"],
                "home_points_form5": home_summary["points_form5"],
                "away_points_form5": away_summary["points_form5"],
                "home_points_form10": home_summary["points_form10"],
                "away_points_form10": away_summary["points_form10"],
                "home_win_rate": home_summary["win_rate"],
                "away_win_rate": away_summary["win_rate"],
                "home_cs_rate": home_summary["cs_rate"],
                "away_cs_rate": away_summary["cs_rate"],
                "home_fts_rate": home_summary["fts_rate"],
                "away_fts_rate": away_summary["fts_rate"],
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
                "home_streak": home_summary["streak"],
                "away_streak": away_summary["streak"],
                "home_attack_strength": home_attack_strength,
                "away_attack_strength": away_attack_strength,
                "home_defence_strength": home_defence_strength,
                "away_defence_strength": away_defence_strength,
                "season_stage": row["season_stage"],
                "h2h_home_wins": h2h["home"],
                "h2h_away_wins": h2h["away"],
                "h2h_draws": h2h["draw"],
                "h2h_total": h2h_total,
                "h2h_home_rate": h2h_home_rate,
            }
        )

        result = int(row["result"])

        home_goals = float(row["home_goals"])
        away_goals = float(row["away_goals"])

        home_record = {
            "gf": home_goals,
            "ga": away_goals,
            "shots": safe_number(row, home_shots_col),
            "sot": safe_number(row, home_sot_col),
            "points": points_from_result(result, True),
            "result": team_result_from_match(result, True),
        }

        away_record = {
            "gf": away_goals,
            "ga": home_goals,
            "shots": safe_number(row, away_shots_col),
            "sot": safe_number(row, away_sot_col),
            "points": points_from_result(result, False),
            "result": team_result_from_match(result, False),
        }

        histories[home_team].append(home_record)
        histories[away_team].append(away_record)

        new_home_elo, new_away_elo = update_elo(home_elo, away_elo, result)
        elos[home_team] = new_home_elo
        elos[away_team] = new_away_elo

        if result == 1:
            h2h_records[pair]["home"] += 1
        elif result == -1:
            h2h_records[pair]["away"] += 1
        else:
            h2h_records[pair]["draw"] += 1

        h2h_records[pair]["total"] += 1

    features = pd.DataFrame(feature_rows)

    for column in MODEL_FEATURES:
        if column not in features.columns:
            features[column] = 0.0

        frame[column] = pd.to_numeric(features[column], errors="coerce").fillna(0).values

    return frame