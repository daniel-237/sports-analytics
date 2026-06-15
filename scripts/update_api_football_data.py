#!/usr/bin/env python3
"""Fetch football data from API-FOOTBALL v3 (API-SPORTS).

Usage:
    python scripts/update_api_football_data.py

This script will:
 - Read API_FOOTBALL_KEY from .env (via python-dotenv)
 - Fetch fixtures, standings, injuries, and lineups
 - Save outputs to CSV files in data/processed/

API Documentation:
    https://www.api-football.com/documentation-v3
"""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

API_BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_IDS = [39, 140, 78, 135, 61]  # PL, La Liga, Bundesliga, Serie A, Ligue 1
MAX_LINEUPS = 25
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIXTURES_PATH = PROCESSED_DIR / "api_football_fixtures.csv"
STANDINGS_PATH = PROCESSED_DIR / "api_football_standings.csv"
INJURIES_PATH = PROCESSED_DIR / "api_football_injuries.csv"
LINEUPS_PATH = PROCESSED_DIR / "api_football_lineups.csv"


def load_api_key() -> Optional[str]:
    load_dotenv()
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        print("Error: API_FOOTBALL_KEY not found in .env file.")
        return None
    return api_key


def get_headers(api_key: str) -> Dict[str, str]:
    return {"x-apisports-key": api_key, "Accept": "application/json"}


def handle_api_response(response: requests.Response, endpoint: str, context: str) -> Optional[Dict]:
    if response.status_code == 429:
        print(f"Rate limit reached for {endpoint} ({context}).")
        return None

    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response else "unknown"
        print(f"HTTP error for {endpoint} ({context}): {status_code}")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"Request error for {endpoint} ({context}): {str(exc)[:100]}")
        return None


def current_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def fetch_fixtures(api_key: str) -> pd.DataFrame:
    headers = get_headers(api_key)
    season = current_season()
    rows: List[Dict] = []

    for league_id in LEAGUE_IDS:
        response = requests.get(
            f"{API_BASE_URL}/fixtures",
            headers=headers,
            params={"league": league_id, "season": season},
            timeout=10,
        )
        data = handle_api_response(response, "fixtures", f"league {league_id}")
        if not data:
            continue

        for item in data.get("response", []):
            fixture = item.get("fixture", {})
            league = item.get("league", {})
            teams = item.get("teams", {})
            score = item.get("score", {})
            rows.append(
                {
                    "competition": league.get("name"),
                    "competition_code": league.get("id"),
                    "season": season,
                    "fixture_id": fixture.get("id"),
                    "fixture_date": fixture.get("date"),
                    "status": fixture.get("status", {}).get("long"),
                    "status_short": fixture.get("status", {}).get("short"),
                    "round": fixture.get("round"),
                    "venue": fixture.get("venue", {}).get("name"),
                    "referee": fixture.get("referee"),
                    "home_team_id": teams.get("home", {}).get("id"),
                    "home_team": teams.get("home", {}).get("name"),
                    "away_team_id": teams.get("away", {}).get("id"),
                    "away_team": teams.get("away", {}).get("name"),
                    "home_goals": score.get("fulltime", {}).get("home"),
                    "away_goals": score.get("fulltime", {}).get("away"),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    return pd.DataFrame(rows)


def fetch_standings(api_key: str) -> pd.DataFrame:
    headers = get_headers(api_key)
    season = current_season()
    rows: List[Dict] = []

    for league_id in LEAGUE_IDS:
        response = requests.get(
            f"{API_BASE_URL}/standings",
            headers=headers,
            params={"league": league_id, "season": season},
            timeout=10,
        )
        data = handle_api_response(response, "standings", f"league {league_id}")
        if not data:
            continue

        for item in data.get("response", []):
            league = item.get("league", {})
            for group in league.get("standings", []):
                for table in group:
                    rows.append(
                        {
                            "competition": league.get("name"),
                            "competition_code": league.get("id"),
                            "season": season,
                            "group": table.get("group"),
                            "rank": table.get("rank"),
                            "team_id": table.get("team", {}).get("id"),
                            "team_name": table.get("team", {}).get("name"),
                            "points": table.get("points"),
                            "played": table.get("all", {}).get("played"),
                            "win": table.get("all", {}).get("win"),
                            "draw": table.get("all", {}).get("draw"),
                            "lose": table.get("all", {}).get("lose"),
                            "goals_for": table.get("all", {}).get("goals", {}).get("for"),
                            "goals_against": table.get("all", {}).get("goals", {}).get("against"),
                            "goal_diff": table.get("goalsDiff"),
                            "form": table.get("form"),
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

    return pd.DataFrame(rows)


def fetch_injuries(api_key: str) -> pd.DataFrame:
    headers = get_headers(api_key)
    season = current_season()
    rows: List[Dict] = []

    for league_id in LEAGUE_IDS:
        response = requests.get(
            f"{API_BASE_URL}/injuries",
            headers=headers,
            params={"league": league_id, "season": season},
            timeout=10,
        )
        data = handle_api_response(response, "injuries", f"league {league_id}")
        if not data:
            continue

        for item in data.get("response", []):
            team = item.get("team", {})
            player = item.get("player", {})
            fixture = item.get("fixture", {})
            injury = item.get("injury", {})
            rows.append(
                {
                    "competition_code": league_id,
                    "season": season,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "position": player.get("position"),
                    "injury": injury.get("type"),
                    "venue": injury.get("venue"),
                    "status": injury.get("status"),
                    "fixture_id": fixture.get("id"),
                    "fixture_date": fixture.get("date"),
                    "injury_date": injury.get("date"),
                    "expected_return": injury.get("end"),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    return pd.DataFrame(rows)


def fetch_lineups(api_key: str, fixture_ids: List[int]) -> pd.DataFrame:
    headers = get_headers(api_key)
    rows: List[Dict] = []

    for fixture_id in fixture_ids[:MAX_LINEUPS]:
        response = requests.get(
            f"{API_BASE_URL}/fixtures/lineups",
            headers=headers,
            params={"fixture": fixture_id},
            timeout=10,
        )
        data = handle_api_response(response, "lineups", f"fixture {fixture_id}")
        if not data:
            continue

        for item in data.get("response", []):
            team = item.get("team", {})
            coach = item.get("coach", {})
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "coach_id": coach.get("id"),
                    "coach_name": coach.get("name"),
                    "formation": item.get("formation"),
                    "startXI": [
                        player.get("player", {}).get("name")
                        for player in item.get("startXI", [])
                    ],
                    "substitutes": [
                        player.get("player", {}).get("name")
                        for player in item.get("substitutes", [])
                    ],
                    "bench": [
                        player.get("player", {}).get("name")
                        for player in item.get("bench", [])
                    ],
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    return pd.DataFrame(rows)


def ensure_output_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def write_csv_with_headers(path: Path, df: pd.DataFrame, columns: List[str]) -> None:
    if df.empty:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def main() -> None:
    api_key = load_api_key()
    if not api_key:
        return

    ensure_output_dir()

    fixtures = fetch_fixtures(api_key)
    fixture_ids = (
        fixtures["fixture_id"].dropna().astype(int).tolist() if not fixtures.empty else []
    )

    standings = fetch_standings(api_key)
    injuries = fetch_injuries(api_key)
    lineups = fetch_lineups(api_key, fixture_ids)

    write_csv_with_headers(
        FIXTURES_PATH,
        fixtures,
        [
            "competition",
            "competition_code",
            "season",
            "fixture_id",
            "fixture_date",
            "status",
            "status_short",
            "round",
            "venue",
            "referee",
            "home_team_id",
            "home_team",
            "away_team_id",
            "away_team",
            "home_goals",
            "away_goals",
            "fetched_at",
        ],
    )
    write_csv_with_headers(
        STANDINGS_PATH,
        standings,
        [
            "competition",
            "competition_code",
            "season",
            "group",
            "rank",
            "team_id",
            "team_name",
            "points",
            "played",
            "win",
            "draw",
            "lose",
            "goals_for",
            "goals_against",
            "goal_diff",
            "form",
            "fetched_at",
        ],
    )
    write_csv_with_headers(
        INJURIES_PATH,
        injuries,
        [
            "competition_code",
            "season",
            "team_id",
            "team_name",
            "player_id",
            "player_name",
            "position",
            "injury",
            "venue",
            "status",
            "fixture_id",
            "fixture_date",
            "injury_date",
            "expected_return",
            "fetched_at",
        ],
    )
    write_csv_with_headers(
        LINEUPS_PATH,
        lineups,
        [
            "fixture_id",
            "team_id",
            "team_name",
            "coach_id",
            "coach_name",
            "formation",
            "startXI",
            "substitutes",
            "bench",
            "fetched_at",
        ],
    )

    print(f"Saved fixtures to {FIXTURES_PATH}")
    print(f"Saved standings to {STANDINGS_PATH}")
    print(f"Saved injuries to {INJURIES_PATH}")
    print(f"Saved lineups to {LINEUPS_PATH}")


if __name__ == "__main__":
    main()
