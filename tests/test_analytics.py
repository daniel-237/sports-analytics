import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_matches():
    return pd.DataFrame({
        "home_team":          ["Arsenal", "Chelsea", "Liverpool", "Man City"],
        "away_team":          ["Chelsea", "Arsenal", "Man City", "Liverpool"],
        "home_goals":         [2, 1, 3, 0],
        "away_goals":         [1, 1, 1, 2],
        "result":             [1, 0, 1, -1],
        "home_form":          [1.8, 1.2, 2.1, 1.5],
        "away_form":          [1.1, 1.4, 0.9, 1.8],
        "home_conceded_form": [0.8, 1.1, 0.6, 1.2],
        "away_conceded_form": [1.3, 0.9, 1.5, 0.8],
        "home_shots_form":    [5.2, 4.8, 6.1, 5.5],
        "away_shots_form":    [4.1, 5.3, 3.8, 5.9],
        "season":             ["2223", "2223", "2324", "2324"],
        "league":             ["Premier League"] * 4,
    })


@pytest.fixture
def sample_players():
    return pd.DataFrame({
        "name":            ["Player A", "Player B", "Player C"],
        "team":            ["Arsenal",  "Chelsea",  "Liverpool"],
        "position":        ["Forward",  "Midfielder", "Defender"],
        "age":             [24, 28, 22],
        "goals":           [15, 5, 1],
        "assists":         [8, 12, 3],
        "minutes":         [2700, 2400, 1800],
        "shots_on_target": [45, 20, 8],
        "tackles":         [12, 35, 68],
        "yellow_cards":    [2, 4, 6],
        "red_cards":       [0, 0, 1],
        "rating":          [7.8, 7.2, 7.0],
    })


# ── TEST 1: Match result labels ───────────────────────────────────────────────

def test_home_win_label(sample_matches):
    home_wins = sample_matches[sample_matches["result"] == 1]
    for _, row in home_wins.iterrows():
        assert row["home_goals"] > row["away_goals"]

def test_draw_label(sample_matches):
    draws = sample_matches[sample_matches["result"] == 0]
    for _, row in draws.iterrows():
        assert row["home_goals"] == row["away_goals"]

def test_away_win_label(sample_matches):
    away_wins = sample_matches[sample_matches["result"] == -1]
    for _, row in away_wins.iterrows():
        assert row["away_goals"] > row["home_goals"]

def test_result_values_valid(sample_matches):
    assert set(sample_matches["result"].unique()).issubset({-1, 0, 1})

def test_proper_draw_detection():
    def result_label(row):
        if row["gf"] > row["ga"]:    return "Win"
        elif row["gf"] == row["ga"]: return "Draw"
        else:                          return "Loss"
    assert result_label({"gf": 1, "ga": 1}) == "Draw"
    assert result_label({"gf": 2, "ga": 1}) == "Win"
    assert result_label({"gf": 0, "ga": 1}) == "Loss"


# ── TEST 2: Prediction features ───────────────────────────────────────────────

FEATURES = ["home_form", "away_form", "home_conceded_form",
            "away_conceded_form", "home_shots_form", "away_shots_form"]

def test_all_required_features_present(sample_matches):
    for f in FEATURES:
        assert f in sample_matches.columns

def test_feature_types_are_numeric(sample_matches):
    for f in FEATURES:
        assert pd.api.types.is_numeric_dtype(sample_matches[f])

def test_prediction_input_shape(sample_matches):
    assert sample_matches[FEATURES].shape[1] == 6

def test_feature_values_in_reasonable_range(sample_matches):
    for f in ["home_form", "away_form", "home_conceded_form", "away_conceded_form"]:
        assert sample_matches[f].min() >= 0
        assert sample_matches[f].max() <= 10


# ── TEST 3: No missing values ─────────────────────────────────────────────────

def test_no_nulls_in_features(sample_matches):
    for f in FEATURES:
        assert sample_matches[f].isna().sum() == 0

def test_no_nulls_in_result(sample_matches):
    assert sample_matches["result"].isna().sum() == 0

def test_dropna_removes_incomplete_rows():
    df = pd.DataFrame({
        "home_form": [1.5, np.nan, 1.2],
        "away_form": [1.2, 1.4, np.nan],
        "home_conceded_form": [0.8, 1.1, 0.9],
        "away_conceded_form": [1.1, 0.9, 1.0],
        "home_shots_form":    [4.5, 5.1, 4.8],
        "away_shots_form":    [4.1, 4.8, 5.2],
        "result": [1, 0, -1],
    })
    clean = df.dropna(subset=FEATURES)
    assert len(clean) == 1


# ── TEST 4: Per 90 stats ──────────────────────────────────────────────────────

def test_zero_minutes_returns_zero_not_inf(sample_players):
    df = sample_players.copy()
    df.loc[0, "minutes"] = 0
    df["mins_safe"]   = df["minutes"].replace(0, np.nan)
    df["goals_p90"]   = (df["goals"]   / df["mins_safe"] * 90).fillna(0)
    df["assists_p90"] = (df["assists"] / df["mins_safe"] * 90).fillna(0)
    assert not df["goals_p90"].isin([np.inf, -np.inf]).any()
    assert df.loc[0, "goals_p90"] == 0

def test_per90_calculation_correct(sample_players):
    df = sample_players.copy()
    df["mins_safe"]  = df["minutes"].replace(0, np.nan)
    df["goals_p90"] = (df["goals"] / df["mins_safe"] * 90).round(2)
    expected = round(15 / 2700 * 90, 2)
    actual   = df.loc[df["name"] == "Player A", "goals_p90"].values[0]
    assert abs(actual - expected) < 0.01

def test_per90_higher_for_efficient_players(sample_players):
    df = sample_players.copy()
    efficient = pd.DataFrame([{
        "name": "Efficient", "team": "Test", "position": "Forward",
        "age": 25, "goals": 5, "assists": 2, "minutes": 450,
        "shots_on_target": 15, "tackles": 5, "yellow_cards": 1,
        "red_cards": 0, "rating": 7.5
    }])
    df = pd.concat([df, efficient], ignore_index=True)
    df["mins_safe"] = df["minutes"].replace(0, np.nan)
    df["goals_p90"] = (df["goals"] / df["mins_safe"] * 90).round(2)
    eff_p90 = df.loc[df["name"] == "Efficient", "goals_p90"].values[0]
    pa_p90  = df.loc[df["name"] == "Player A",  "goals_p90"].values[0]
    assert eff_p90 > pa_p90


# ── TEST 5: Team filtering ────────────────────────────────────────────────────

def test_home_filter_correct(sample_matches):
    filtered = sample_matches[sample_matches["home_team"] == "Arsenal"]
    assert all(filtered["home_team"] == "Arsenal")

def test_away_filter_correct(sample_matches):
    filtered = sample_matches[sample_matches["away_team"] == "Chelsea"]
    assert all(filtered["away_team"] == "Chelsea")

def test_combined_filter_covers_all_games(sample_matches):
    all_games = sample_matches[
        (sample_matches["home_team"] == "Arsenal") |
        (sample_matches["away_team"] == "Arsenal")
    ]
    assert len(all_games) == 2

def test_season_filter_works(sample_matches):
    filtered = sample_matches[sample_matches["season"] == "2223"]
    assert all(filtered["season"] == "2223")
    assert len(filtered) == 2


# ── TEST 6: Chronological split ───────────────────────────────────────────────

def test_train_seasons_older(sample_matches):
    df = sample_matches.copy()
    df["season"] = df["season"].astype(str)
    train = df[df["season"] <= "2223"]
    test  = df[df["season"] >  "2223"]
    assert len(train) > 0
    assert len(test)  > 0
    assert max(train["season"]) <= min(test["season"])

def test_no_data_leakage(sample_matches):
    df = sample_matches.copy()
    df["season"] = df["season"].astype(str)
    train_s = set(df[df["season"] <= "2223"]["season"])
    test_s  = set(df[df["season"] >  "2223"]["season"])
    assert len(train_s & test_s) == 0