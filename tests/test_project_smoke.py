"""Smoke tests that exercise the real src/ code paths end to end.

Unlike test_analytics.py (which tests inline pandas snippets), these import
and run the actual project modules so a regression in feature engineering,
prediction, or league simulation fails CI.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_FEATURES
from src.utils import (
    format_season,
    result_code_from_scores,
    season_start_year,
    sorted_seasons,
)
from src.feature_engineering import build_match_features
from src.prediction import build_prediction_features, model_class_name
from src.league_simulation import (
    _current_table_summary,
    _normalise_probs,
    simulate_league_table,
)


def _toy_matches(n_rounds: int = 6) -> pd.DataFrame:
    """Two small synthetic seasons so form/Elo/H2H features have history."""
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City"]
    rng = np.random.default_rng(0)
    rows = []
    for season, year in [("2021", 2021), ("2122", 2022)]:
        for r in range(n_rounds):
            for i in range(0, len(teams), 2):
                rows.append(
                    {
                        "date": f"{year}-{r + 1:02d}-01",
                        "season": season,
                        "league": "Premier League",
                        "home_team": teams[i],
                        "away_team": teams[i + 1],
                        "home_goals": int(rng.integers(0, 4)),
                        "away_goals": int(rng.integers(0, 4)),
                    }
                )
            teams = teams[1:] + teams[:1]  # rotate to vary fixtures
    return pd.DataFrame(rows)


def _toy_standings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_name": ["Arsenal", "Chelsea", "Liverpool", "Man City"],
            "rank": [1, 2, 3, 4],
            "points": [70, 65, 60, 55],
            "goal_diff": [30, 20, 15, 10],
            "goals_for": [70, 60, 55, 50],
        }
    )


# ── core label / scoring helpers ──────────────────────────────────────────────

def test_result_code_from_scores():
    assert result_code_from_scores(2, 1) == 1
    assert result_code_from_scores(1, 1) == 0
    assert result_code_from_scores(0, 2) == -1


def test_model_class_name_uses_encoded_mapping():
    # The model encodes 0=Away, 1=Draw, 2=Home. Regression guard against the
    # old conflated maps where 0 meant "Draw".
    assert model_class_name(0) == "Away Win"
    assert model_class_name(1) == "Draw"
    assert model_class_name(2) == "Home Win"


def test_season_helpers():
    assert format_season("2122") == "2021/22"
    assert sorted_seasons(["2122", "1920", "2021"]) == ["1920", "2021", "2122"]


def test_season_start_year_parses_historic_seasons():
    # Regression: "1962/63" must parse to 1962, not 2062. The old parser took
    # the trailing four digits, flinging 1930s-1960s seasons into the future
    # and corrupting the chronological train/test split (test set became 1960s
    # data instead of recent seasons).
    assert season_start_year("1962/63") == 1962
    assert season_start_year("1937/38") == 1937
    assert season_start_year("2015/16") == 2015
    assert season_start_year("2024/25") == 2024
    assert format_season("1962/63") == "1962/63"
    # modern seasons must sort after historic ones
    assert sorted_seasons(["2024/25", "1962/63", "1990/91"]) == [
        "1962/63",
        "1990/91",
        "2024/25",
    ]


# ── feature engineering ───────────────────────────────────────────────────────

def test_build_match_features_emits_all_model_features():
    feats = build_match_features(_toy_matches())
    for col in MODEL_FEATURES:
        assert col in feats.columns, f"missing engineered feature: {col}"
    assert feats[MODEL_FEATURES].isna().sum().sum() == 0
    assert set(feats["result"].unique()).issubset({-1, 0, 1})


def test_build_prediction_features_one_row_exact_columns():
    matches = build_match_features(_toy_matches())
    X = build_prediction_features(matches, "Arsenal", "Chelsea", MODEL_FEATURES)
    assert list(X.columns) == MODEL_FEATURES
    assert len(X) == 1
    assert X.isna().sum().sum() == 0


# ── league simulation ─────────────────────────────────────────────────────────

def test_normalise_probs_always_sums_to_one():
    assert abs(sum(_normalise_probs(0.45, 0.25, 0.30)) - 1.0) < 1e-9
    # degenerate input must fall back to a valid distribution, not zeros
    assert abs(sum(_normalise_probs(0.0, 0.0, 0.0)) - 1.0) < 1e-9


def test_current_table_summary_columns_and_champion():
    summary = _current_table_summary(_toy_standings())
    for col in [
        "team_name", "current_rank", "expected_points",
        "champion_prob", "top_4_prob", "top_6_prob", "bottom_3_prob",
    ]:
        assert col in summary.columns
    arsenal = summary.loc[summary["team_name"] == "Arsenal"].iloc[0]
    assert arsenal["champion_prob"] == 1.0


def test_simulate_league_table_probabilities_valid():
    fixtures_probs = pd.DataFrame(
        {
            "home_team": ["Arsenal", "Liverpool"],
            "away_team": ["Chelsea", "Man City"],
            "home_win_prob": [0.5, 0.4],
            "draw_prob": [0.3, 0.3],
            "away_win_prob": [0.2, 0.3],
        }
    )
    table = simulate_league_table(fixtures_probs, _toy_standings(), simulations=50)
    assert not table.empty
    for col in ["champion_prob", "top_4_prob", "bottom_3_prob"]:
        assert (table[col] >= 0).all() and (table[col] <= 1).all()
    # exactly one champion per simulation -> probabilities sum to 1
    assert abs(table["champion_prob"].sum() - 1.0) < 1e-6


def test_simulate_league_table_is_reproducible_with_seed():
    fixtures_probs = pd.DataFrame(
        {
            "home_team": ["Arsenal", "Liverpool"],
            "away_team": ["Chelsea", "Man City"],
            "home_win_prob": [0.5, 0.4],
            "draw_prob": [0.3, 0.3],
            "away_win_prob": [0.2, 0.3],
        }
    )
    a = simulate_league_table(fixtures_probs, _toy_standings(), simulations=50, seed=7)
    b = simulate_league_table(fixtures_probs, _toy_standings(), simulations=50, seed=7)
    pd.testing.assert_frame_equal(a, b)


# ── trained-model integration (skips if artifacts absent) ─────────────────────

def test_trained_model_predicts_end_to_end():
    from src.data_loader import load_model
    from src.prediction import predict_match

    model = load_model()
    if model is None:
        pytest.skip("trained model artifact not present")

    matches = build_match_features(_toy_matches())
    result = predict_match(model, matches, "Arsenal", "Chelsea")
    assert result["prediction"] in {"Home Win", "Draw", "Away Win"}
    assert 0.0 <= result["confidence"] <= 1.0
