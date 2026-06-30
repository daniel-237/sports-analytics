#!/usr/bin/env python3
"""Fetch current football data from football-data.org API v4.

Usage:
    python scripts/update_football_data_api.py

This script will:
 - Read FOOTBALL_DATA_API_KEY from .env (via python-dotenv)
 - Fetch standings for major competitions
 - Fetch upcoming fixtures
 - Fetch latest finished results
 - Save to CSV files in data/processed/

API Documentation:
    https://www.football-data.org/client/register (API v4)
"""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv


# Competition codes for major leagues and Cup
COMPETITIONS = ["PL", "PD", "BL1", "SA", "FL1", "CL"]

API_BASE_URL = "https://api.football-data.org/v4"

# Project root and output paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_PATH = PROJECT_ROOT / "data/processed/current_tables.csv"
UPCOMING_PATH = PROJECT_ROOT / "data/processed/upcoming_fixtures.csv"
RESULTS_PATH = PROJECT_ROOT / "data/processed/latest_results.csv"


def load_api_key() -> Optional[str]:
    """Load API key from .env file safely.
    
    Returns:
        API key string if found, None otherwise.
    """
    load_dotenv()
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("Error: FOOTBALL_DATA_API_KEY not found in .env file.")
        return None
    return api_key


def get_headers(api_key: str) -> dict:
    """Return headers for football-data.org API requests."""
    return {"X-Auth-Token": api_key}


def handle_api_response(response: requests.Response, competition_code: str, endpoint: str) -> Optional[dict]:
    """Handle API response and rate limit errors gracefully.
    
    Args:
        response: requests.Response object
        competition_code: Competition code (e.g., 'PL')
        endpoint: API endpoint name (e.g., 'standings')
    
    Returns:
        Parsed JSON data if successful, None on error.
    """
    if response.status_code == 429:
        print(f"Rate limit reached for {competition_code} ({endpoint}). API quota exhausted.")
        return None
    
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error fetching {endpoint} for {competition_code}: {e.response.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching {endpoint} for {competition_code}: {str(e)[:100]}")
        return None


def season_label(season: dict) -> str:
    """Build a readable season label (e.g. '2025/26') from the API season object,
    derived from the start/end dates rather than the opaque numeric id."""
    start = str(season.get("startDate") or "")[:4]
    end = str(season.get("endDate") or "")[:4]
    if start.isdigit():
        if end.isdigit() and end != start:
            return f"{start}/{end[-2:]}"
        return start
    season_id = season.get("id")
    return str(season_id) if season_id is not None else ""


def fetch_standings(api_key: str) -> pd.DataFrame:
    """Fetch standings for all competitions."""
    headers = get_headers(api_key)
    rows = []

    for code in COMPETITIONS:
        try:
            url = f"{API_BASE_URL}/competitions/{code}/standings"
            response = requests.get(url, headers=headers, timeout=10)
            
            data = handle_api_response(response, code, "standings")
            if not data:
                continue

            competition_name = data.get("competition", {}).get("name", code)
            season = data.get("season", {})

            for standings_item in data.get("standings", []):
                for table in standings_item.get("table", []):
                    rows.append(
                        {
                            "competition": competition_name,
                            "competition_code": code,
                            "season": season_label(season),
                            "position": table.get("position"),
                            "team": table.get("team", {}).get("name"),
                            "team_id": table.get("team", {}).get("id"),
                            "played": table.get("playedGames"),
                            "won": table.get("won"),
                            "draw": table.get("draw"),
                            "lost": table.get("lost"),
                            "goals_for": table.get("goalsFor"),
                            "goals_against": table.get("goalsAgainst"),
                            "goal_difference": table.get("goalDifference"),
                            "points": table.get("points"),
                            "last_updated": datetime.now(timezone.utc).isoformat(),
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

        except Exception as e:
            print(f"Unexpected error processing standings for {code}: {str(e)[:100]}")
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_upcoming_fixtures(api_key: str) -> pd.DataFrame:
    """Fetch upcoming fixtures for all competitions."""
    headers = get_headers(api_key)
    rows = []

    for code in COMPETITIONS:
        try:
            url = f"{API_BASE_URL}/competitions/{code}/matches?status=SCHEDULED"
            response = requests.get(url, headers=headers, timeout=10)
            
            data = handle_api_response(response, code, "upcoming fixtures")
            if not data:
                continue

            competition_name = data.get("competition", {}).get("name", code)

            for match in data.get("matches", []):
                rows.append(
                    {
                        "competition": competition_name,
                        "competition_code": code,
                        "season": season_label(data.get("season", {})),
                        "match_id": match.get("id"),
                        "utc_date": match.get("utcDate"),
                        "status": match.get("status"),
                        "home_team": match.get("homeTeam", {}).get("name"),
                        "away_team": match.get("awayTeam", {}).get("name"),
                        "home_team_id": match.get("homeTeam", {}).get("id"),
                        "away_team_id": match.get("awayTeam", {}).get("id"),
                        "stage": match.get("stage"),
                        "group": match.get("group"),
                        "matchday": match.get("matchday"),
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

        except Exception as e:
            print(f"Unexpected error processing upcoming fixtures for {code}: {str(e)[:100]}")
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_latest_results(api_key: str) -> pd.DataFrame:
    """Fetch latest finished results for all competitions."""
    headers = get_headers(api_key)
    rows = []

    for code in COMPETITIONS:
        try:
            url = f"{API_BASE_URL}/competitions/{code}/matches?status=FINISHED"
            response = requests.get(url, headers=headers, timeout=10)
            
            data = handle_api_response(response, code, "finished results")
            if not data:
                continue

            competition_name = data.get("competition", {}).get("name", code)

            for match in data.get("matches", []):
                rows.append(
                    {
                        "competition": competition_name,
                        "competition_code": code,
                        "season": season_label(data.get("season", {})),
                        "match_id": match.get("id"),
                        "utc_date": match.get("utcDate"),
                        "status": match.get("status"),
                        "home_team": match.get("homeTeam", {}).get("name"),
                        "away_team": match.get("awayTeam", {}).get("name"),
                        "home_team_id": match.get("homeTeam", {}).get("id"),
                        "away_team_id": match.get("awayTeam", {}).get("id"),
                        "home_goals": match.get("score", {}).get("fullTime", {}).get("home"),
                        "away_goals": match.get("score", {}).get("fullTime", {}).get("away"),
                        "stage": match.get("stage"),
                        "group": match.get("group"),
                        "matchday": match.get("matchday"),
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

        except Exception as e:
            print(f"Unexpected error processing finished results for {code}: {str(e)[:100]}")
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    """Main entry point."""
    api_key = load_api_key()
    if not api_key:
        return

    # Ensure output directory exists
    TABLES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Define column schemas for empty CSVs
    standings_columns = [
        "competition", "competition_code", "season", "position", "team", "team_id",
        "played", "won", "draw", "lost", "goals_for", "goals_against",
        "goal_difference", "points", "last_updated", "fetched_at"
    ]
    upcoming_columns = [
        "competition", "competition_code", "season", "match_id", "utc_date", "status",
        "home_team", "away_team", "home_team_id", "away_team_id", "stage",
        "group", "matchday", "last_updated", "fetched_at"
    ]
    results_columns = [
        "competition", "competition_code", "season", "match_id", "utc_date", "status",
        "home_team", "away_team", "home_team_id", "away_team_id", "home_goals",
        "away_goals", "stage", "group", "matchday", "last_updated", "fetched_at"
    ]

    print("Fetching standings...")
    standings = fetch_standings(api_key)
    if standings.empty:
        standings = pd.DataFrame(columns=standings_columns)
    standings.to_csv(TABLES_PATH, index=False)
    print(f"  Saved {len(standings)} rows to {TABLES_PATH}")

    print("Fetching upcoming fixtures...")
    upcoming = fetch_upcoming_fixtures(api_key)
    if upcoming.empty:
        upcoming = pd.DataFrame(columns=upcoming_columns)
    upcoming.to_csv(UPCOMING_PATH, index=False)
    print(f"  Saved {len(upcoming)} rows to {UPCOMING_PATH}")

    print("Fetching latest results...")
    results = fetch_latest_results(api_key)
    if results.empty:
        results = pd.DataFrame(columns=results_columns)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"  Saved {len(results)} rows to {RESULTS_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()
