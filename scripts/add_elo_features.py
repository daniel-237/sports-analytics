#!/usr/bin/env python3
"""Compute Elo ratings for matches and persist them into processed files.

Usage:
    python scripts/add_elo_features.py

This script will:
 - backup data/processed/matches_clean.csv -> data/processed/matches_clean_before_elo.csv
 - compute pre-match Elo (`home_elo`, `away_elo`, `elo_diff`) and post-match Elo (`home_elo_after`, `away_elo_after`) chronologically
 - overwrite data/processed/matches_clean.csv with the new columns
 - write latest team ratings to data/processed/team_elo_ratings.csv

Elo parameters (per project requirements):
 - starting Elo: 1500
 - home advantage: 60
 - K-factor: 30
"""
from pathlib import Path
import shutil
from collections import defaultdict
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import result_code_from_scores
from src.feature_engineering import update_elo


MATCHES_PATH = Path("data/processed/matches_clean.csv")
BACKUP_PATH = Path("data/processed/matches_clean_before_elo.csv")
TEAM_RATINGS_PATH = Path("data/processed/team_elo_ratings.csv")


def main():
    if not MATCHES_PATH.exists():
        print(f"No matches file found at {MATCHES_PATH}. Nothing to do.")
        return

    # Backup
    shutil.copy(MATCHES_PATH, BACKUP_PATH)
    print(f"Backed up {MATCHES_PATH} -> {BACKUP_PATH}")

    df = pd.read_csv(MATCHES_PATH)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Sort chronologically
    if "date" in df.columns:
        df = df.sort_values("date", na_position="last").reset_index(drop=True)

    elos = defaultdict(lambda: 1500.0)

    home_elos = []
    away_elos = []
    elo_diffs = []
    home_elos_after = []
    away_elos_after = []

    for _, row in df.iterrows():
        home = str(row.get("home_team", "")).strip()
        away = str(row.get("away_team", "")).strip()

        home_elo = float(elos[home])
        away_elo = float(elos[away])

        # record pre-match elos
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        elo_diffs.append(home_elo - away_elo)

        # compute result from scores if possible
        try:
            home_goals = float(row.get("home_goals", 0))
            away_goals = float(row.get("away_goals", 0))
            result = result_code_from_scores(home_goals, away_goals)
        except Exception:
            result = 0

        # update using K=30 and home advantage 60 (per requirements)
        new_home, new_away = update_elo(home_elo, away_elo, result, k=30, home_advantage=60)

        home_elos_after.append(new_home)
        away_elos_after.append(new_away)

        elos[home] = new_home
        elos[away] = new_away

    # Attach columns
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = elo_diffs
    df["home_elo_after"] = home_elos_after
    df["away_elo_after"] = away_elos_after

    # Persist updated matches file (overwrite)
    df.to_csv(MATCHES_PATH, index=False)
    print(f"Wrote updated matches to {MATCHES_PATH}")

    # Build latest team ratings
    rows = []
    teams = set(list(df["home_team"].dropna().astype(str).unique()) + list(df["away_team"].dropna().astype(str).unique()))
    for team in teams:
        team_mask = (df["home_team"].astype(str) == str(team)) | (df["away_team"].astype(str) == str(team))
        played = df[team_mask].copy()
        matches_played = len(played)
        last_date = None
        league = None
        if not played.empty and "date" in played.columns:
            last_date = played["date"].max()
            last_row = played.sort_values("date", na_position="last").iloc[-1]
            league = last_row.get("league") if "league" in last_row.index else None

        latest = float(elos.get(str(team), 1500.0))

        rows.append({
            "team": team,
            "league": league,
            "latest_elo": latest,
            "matches_played": matches_played,
            "last_match_date": last_date,
        })

    ratings = pd.DataFrame(rows)
    ratings.to_csv(TEAM_RATINGS_PATH, index=False)
    print(f"Wrote team ratings to {TEAM_RATINGS_PATH}")


if __name__ == "__main__":
    main()
