import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES
from src.feature_engineering import build_match_features
from src.utils import safe_mean

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


def model_class_name(value, metrics: dict | None = None) -> str:
    if metrics and "target_mapping" in metrics:
        mapped = metrics["target_mapping"].get(str(value))
        if mapped:
            return mapped

    mapping = {
        0: "Away Win",
        1: "Draw",
        2: "Home Win",
        "0": "Away Win",
        "1": "Draw",
        "2": "Home Win",
    }

    return mapping.get(value, str(value))


def get_expected_model_features(model, metrics: dict | None = None) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if metrics and "model_features" in metrics:
        return list(metrics["model_features"])

    return MODEL_FEATURES


def explain_prediction(model, features_df: pd.DataFrame, metrics: dict | None = None, top_n: int = 6):
    if not SHAP_AVAILABLE:
        return None

    try:
        model_features = get_expected_model_features(model, metrics)

        X = features_df.copy()
        if isinstance(X, pd.Series):
            X = pd.DataFrame([X])
        X = X.reindex(columns=model_features, fill_value=0.0)

        explainer_model = model
        try:
            if hasattr(model, "named_steps") and isinstance(model.named_steps, dict) and model.named_steps:
                explainer_model = list(model.named_steps.values())[-1]
            elif hasattr(model, "steps") and isinstance(model.steps, list) and model.steps:
                explainer_model = model.steps[-1][1]
        except Exception:
            explainer_model = model

        try:
            probs = model.predict_proba(X)[0]
            pred_index = int(np.argmax(probs))
        except Exception:
            pred_index = 0

        shap_vals_obj = None
        try:
            expl = shap.TreeExplainer(explainer_model)
            shap_vals_obj = expl.shap_values(X)
        except Exception:
            try:
                expl = shap.Explainer(explainer_model, X)
                shap_vals_obj = expl(X)
            except Exception:
                return None

        if isinstance(shap_vals_obj, list):
            if pred_index >= len(shap_vals_obj):
                return None
            selected = np.array(shap_vals_obj[pred_index])
            if selected.ndim == 2 and selected.shape[0] == 1:
                shap_values = selected[0]
            elif selected.ndim == 1:
                shap_values = selected
            else:
                return None
        else:
            arr = np.array(getattr(shap_vals_obj, "values", shap_vals_obj))
            if arr.ndim == 3:
                if arr.shape[1] == len(model_features) and pred_index < arr.shape[2]:
                    shap_values = arr[0, :, pred_index]
                elif arr.shape[2] == len(model_features) and pred_index < arr.shape[1]:
                    shap_values = arr[0, pred_index, :]
                else:
                    return None
            elif arr.ndim == 2:
                if arr.shape[0] == 1 and arr.shape[1] == len(model_features):
                    shap_values = arr[0]
                elif arr.shape[1] == len(model_features):
                    shap_values = arr[0]
                else:
                    return None
            elif arr.ndim == 1 and arr.shape[0] == len(model_features):
                shap_values = arr
            else:
                return None

        if len(shap_values) != len(model_features):
            return None

        rows = []
        for i, feature_name in enumerate(model_features):
            rows.append(
                {
                    "feature": feature_name,
                    "feature_value": float(X.iloc[0].get(feature_name, 0.0)),
                    "shap_value": float(shap_values[i]),
                }
            )

        df = pd.DataFrame(rows)
        human_map = {
            "elo_diff": "Team strength difference",
            "home_elo": "Home team Elo",
            "away_elo": "Away team Elo",
            "home_points_form5": "Home recent form",
            "away_points_form5": "Away recent form",
            "home_goals_form": "Home attacking form",
            "away_goals_form": "Away attacking form",
        }
        df["human_name"] = df["feature"].map(human_map).fillna(df["feature"])

        return {"predicted_class_index": pred_index, "feature_shap": df}
    except Exception:
        return None


def prepare_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["home_elo", "away_elo", "elo_diff"]

    needs_building = False

    for column in required:
        if column not in frame.columns:
            needs_building = True
            break

    if not needs_building:
        total = frame[required].abs().sum().sum()
        if total == 0:
            needs_building = True

    if needs_building:
        return build_match_features(frame)

    return frame.copy()


def team_matches(frame: pd.DataFrame, team: str) -> pd.DataFrame:
    team = str(team)

    matches = frame[
        (frame["home_team"].astype(str) == team)
        | (frame["away_team"].astype(str) == team)
    ].copy()

    if "date" in matches.columns:
        matches = matches.sort_values("date", na_position="last")

    return matches


def home_matches(frame: pd.DataFrame, team: str) -> pd.DataFrame:
    team = str(team)

    matches = frame[frame["home_team"].astype(str) == team].copy()

    if "date" in matches.columns:
        matches = matches.sort_values("date", na_position="last")

    return matches


def away_matches(frame: pd.DataFrame, team: str) -> pd.DataFrame:
    team = str(team)

    matches = frame[frame["away_team"].astype(str) == team].copy()

    if "date" in matches.columns:
        matches = matches.sort_values("date", na_position="last")

    return matches


def row_number(row: pd.Series, columns: list[str]) -> float:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(row[column], errors="coerce")
            if not pd.isna(value):
                return float(value)

    return 0.0


def result_points(row: pd.Series, team: str) -> int:
    team = str(team)
    is_home = str(row["home_team"]) == team

    if row["home_goals"] == row["away_goals"]:
        return 1

    if is_home and row["home_goals"] > row["away_goals"]:
        return 3

    if not is_home and row["away_goals"] > row["home_goals"]:
        return 3

    return 0


def goals_for(row: pd.Series, team: str) -> float:
    if str(row["home_team"]) == str(team):
        return float(row["home_goals"])

    return float(row["away_goals"])


def goals_against(row: pd.Series, team: str) -> float:
    if str(row["home_team"]) == str(team):
        return float(row["away_goals"])

    return float(row["home_goals"])


def shots_for(row: pd.Series, team: str) -> float:
    if str(row["home_team"]) == str(team):
        return row_number(row, ["home_shots", "hs"])

    return row_number(row, ["away_shots", "as"])


def sot_for(row: pd.Series, team: str) -> float:
    if str(row["home_team"]) == str(team):
        return row_number(row, ["home_sot", "home_shots_on_target", "home_shots_target", "hst"])

    return row_number(row, ["away_sot", "away_shots_on_target", "away_shots_target", "ast"])


def team_goal_difference(row: pd.Series, team: str) -> float:
    return goals_for(row, team) - goals_against(row, team)


def win_value(row: pd.Series, team: str) -> int:
    return int(result_points(row, team) == 3)


def clean_sheet_value(row: pd.Series, team: str) -> int:
    return int(goals_against(row, team) == 0)


def failed_to_score_value(row: pd.Series, team: str) -> int:
    return int(goals_for(row, team) == 0)


def streak_value(records: pd.DataFrame, team: str) -> int:
    if records.empty:
        return 0

    streak = 0
    last_result = None

    for _, row in records.tail(10).iloc[::-1].iterrows():
        points = result_points(row, team)

        if points == 3:
            current = 1
        elif points == 1:
            current = 0
        else:
            current = -1

        if last_result is None:
            last_result = current

        if current == last_result and current != 0:
            streak += current
        else:
            break

    return streak


def latest_team_elo(frame: pd.DataFrame, team: str) -> float:
    history = team_matches(frame, team)

    if history.empty:
        return 1500.0

    for _, row in history.iloc[::-1].iterrows():
        if str(row["home_team"]) == str(team) and "home_elo" in row.index:
            return float(row["home_elo"])

        if str(row["away_team"]) == str(team) and "away_elo" in row.index:
            return float(row["away_elo"])

    return 1500.0


def head_to_head_features(frame: pd.DataFrame, home_team: str, away_team: str) -> dict:
    h2h = frame[
        (frame["home_team"].astype(str) == str(home_team))
        & (frame["away_team"].astype(str) == str(away_team))
    ].copy()

    if h2h.empty:
        return {
            "h2h_home_wins": 0,
            "h2h_away_wins": 0,
            "h2h_draws": 0,
            "h2h_total": 0,
            "h2h_home_rate": 0.0,
        }

    home_wins = int((h2h["home_goals"] > h2h["away_goals"]).sum())
    away_wins = int((h2h["away_goals"] > h2h["home_goals"]).sum())
    draws = int((h2h["home_goals"] == h2h["away_goals"]).sum())
    total = int(len(h2h))

    return {
        "h2h_home_wins": home_wins,
        "h2h_away_wins": away_wins,
        "h2h_draws": draws,
        "h2h_total": total,
        "h2h_home_rate": home_wins / total if total else 0.0,
    }


def form_summary(records: pd.DataFrame, team: str, size: int = 5) -> dict:
    recent = records.tail(size)

    if recent.empty:
        return {
            "goals": 0.0,
            "conceded": 0.0,
            "shots": 0.0,
            "sot": 0.0,
            "goal_difference": 0.0,
            "points": 0.0,
            "win_rate": 0.0,
            "clean_sheet_rate": 0.0,
            "failed_to_score_rate": 0.0,
        }

    return {
        "goals": safe_mean(recent.apply(lambda row: goals_for(row, team), axis=1), 0.0),
        "conceded": safe_mean(recent.apply(lambda row: goals_against(row, team), axis=1), 0.0),
        "shots": safe_mean(recent.apply(lambda row: shots_for(row, team), axis=1), 0.0),
        "sot": safe_mean(recent.apply(lambda row: sot_for(row, team), axis=1), 0.0),
        "goal_difference": safe_mean(recent.apply(lambda row: team_goal_difference(row, team), axis=1), 0.0),
        "points": safe_mean(recent.apply(lambda row: result_points(row, team), axis=1), 0.0),
        "win_rate": safe_mean(recent.apply(lambda row: win_value(row, team), axis=1), 0.0),
        "clean_sheet_rate": safe_mean(recent.apply(lambda row: clean_sheet_value(row, team), axis=1), 0.0),
        "failed_to_score_rate": safe_mean(recent.apply(lambda row: failed_to_score_value(row, team), axis=1), 0.0),
    }


def build_prediction_features(
    frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    model_features: list[str],
) -> pd.DataFrame:
    frame = prepare_prediction_frame(frame)

    home_all = team_matches(frame, home_team)
    away_all = team_matches(frame, away_team)

    home_home = home_matches(frame, home_team)
    away_away = away_matches(frame, away_team)

    home_form_5 = form_summary(home_home if not home_home.empty else home_all, home_team, 5)
    away_form_5 = form_summary(away_away if not away_away.empty else away_all, away_team, 5)

    home_form_10 = form_summary(home_all, home_team, 10)
    away_form_10 = form_summary(away_all, away_team, 10)

    home_elo = latest_team_elo(frame, home_team)
    away_elo = latest_team_elo(frame, away_team)

    h2h = head_to_head_features(frame, home_team, away_team)

    feature_row = {
        "home_goals_form": home_form_5["goals"],
        "away_goals_form": away_form_5["goals"],
        "home_conceded_form": home_form_5["conceded"],
        "away_conceded_form": away_form_5["conceded"],
        "home_shots_form": home_form_5["shots"],
        "away_shots_form": away_form_5["shots"],
        "home_sot_form": home_form_5["sot"],
        "away_sot_form": away_form_5["sot"],
        "home_gd_form": home_form_5["goal_difference"],
        "away_gd_form": away_form_5["goal_difference"],
        "home_points_form5": home_form_5["points"],
        "away_points_form5": away_form_5["points"],
        "home_points_form10": home_form_10["points"],
        "away_points_form10": away_form_10["points"],
        "home_win_rate": home_form_5["win_rate"],
        "away_win_rate": away_form_5["win_rate"],
        "home_cs_rate": home_form_5["clean_sheet_rate"],
        "away_cs_rate": away_form_5["clean_sheet_rate"],
        "home_fts_rate": home_form_5["failed_to_score_rate"],
        "away_fts_rate": away_form_5["failed_to_score_rate"],
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_streak": streak_value(home_all, home_team),
        "away_streak": streak_value(away_all, away_team),
        "home_attack_strength": home_form_5["goals"] / 1.35 if home_form_5["goals"] else 1.0,
        "away_attack_strength": away_form_5["goals"] / 1.35 if away_form_5["goals"] else 1.0,
        "home_defence_strength": 1.35 / home_form_5["conceded"] if home_form_5["conceded"] else 1.0,
        "away_defence_strength": 1.35 / away_form_5["conceded"] if away_form_5["conceded"] else 1.0,
        "season_stage": 0.5,
        **h2h,
    }

    row = {feature: feature_row.get(feature, 0.0) for feature in model_features}

    return pd.DataFrame([row], columns=model_features)


def predict_match(model, frame: pd.DataFrame, home_team: str, away_team: str, metrics: dict | None = None) -> dict:
    model_features = get_expected_model_features(model, metrics)
    features = build_prediction_features(frame, home_team, away_team, model_features)

    probabilities = model.predict_proba(features)[0]
    classes = list(getattr(model, "classes_", [0, 1, 2]))

    probability_table = pd.DataFrame(
        {
            "class": classes,
            "outcome": [model_class_name(value, metrics) for value in classes],
            "probability": probabilities,
        }
    )

    ordered = pd.DataFrame(
        {
            "Outcome": ["Home Win", "Draw", "Away Win"],
            "Probability": [
                probability_table.loc[probability_table["outcome"] == "Home Win", "probability"].sum(),
                probability_table.loc[probability_table["outcome"] == "Draw", "probability"].sum(),
                probability_table.loc[probability_table["outcome"] == "Away Win", "probability"].sum(),
            ],
        }
    )

    winner_row = ordered.loc[ordered["Probability"].idxmax()]

    full_features = build_prediction_features(frame, home_team, away_team, MODEL_FEATURES)
    
    return {
    "prediction": winner_row["Outcome"],
    "confidence": float(winner_row["Probability"]),
    "probabilities": ordered,
    "features": features,
    "explanation_features": full_features,
}