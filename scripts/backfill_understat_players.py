#!/usr/bin/env python3
"""Backfill historical player-season stats from Understat into player_stats.csv.

Understat (https://understat.com) exposes per-player season aggregates for the
big-five European leagues back to 2014/15 via a POST endpoint. This pulls those
seasons and merges them into data/processed/player_stats.csv, mapping the fields
onto the existing FBref-style schema (the dashboard recomputes per-90 and
performance scores on load, so only raw stats are needed). Understat adds xG/xA
which the FBref export lacks.

The current 2025/26 rows already in the file (from the FBref merge) are kept as
is; the historical seasons handled here are dropped-and-rebuilt, so re-running is
idempotent. Respects the site with a short delay between requests.

Usage:
    python scripts/backfill_understat_players.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "data" / "processed" / "player_stats.csv"

URL = "https://understat.com/main/getPlayersStats/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}
# Understat league key -> the competition label used in player_stats.csv
LEAGUES = {
    "EPL": "eng Premier League",
    "La_liga": "es La Liga",
    "Bundesliga": "de Bundesliga",
    "Serie_A": "it Serie A",
    "Ligue_1": "fr Ligue 1",
}
SEASONS = list(range(2014, 2025))  # 2014/15 .. 2024/25 (2025/26 stays from FBref)


def map_position(raw: str) -> tuple[str, str]:
    """Map an Understat position token to (code, group)."""
    text = str(raw or "").strip().upper()
    token = text.split()[0] if text else ""
    if token.startswith("GK"):
        return "GK", "Goalkeeper"
    if token.startswith(("AM", "DM")):
        return "MF", "Midfielder"
    return {"D": ("DF", "Defender"), "M": ("MF", "Midfielder"), "F": ("FW", "Forward")}.get(
        token[:1], ("", "Unknown Position")
    )


def number(value, integer: bool = True):
    try:
        result = float(value)
        return int(result) if integer else round(result, 3)
    except (TypeError, ValueError):
        return 0


def fetch_players(league: str, year: int) -> list[dict]:
    """Fetch one league-season of player aggregates, with a couple of retries."""
    for _ in range(3):
        try:
            resp = requests.post(
                URL,
                headers={**HEADERS, "Referer": f"https://understat.com/league/{league}/{year}"},
                data={"league": league, "season": str(year)},
                timeout=30,
            )
            return resp.json().get("players", [])
        except Exception:  # noqa: BLE001 - transient network/parse issues, retry
            time.sleep(2)
    return []


def main() -> None:
    rows = []
    for league, competition in LEAGUES.items():
        for year in SEASONS:
            players = fetch_players(league, year)
            if not players:
                print(f"  WARN: no data for {league} {year}")
                continue
            for player in players:
                position, group = map_position(player.get("position"))
                rows.append(
                    {
                        "season": f"{year}-{year + 1}",
                        "name": player.get("player_name"),
                        "player": player.get("player_name"),
                        "team": player.get("team_title"),
                        "squad": player.get("team_title"),
                        "competition": competition,
                        "position": position,
                        "position_group": group,
                        "appearances": number(player.get("games")),
                        "minutes": number(player.get("time")),
                        "goals": number(player.get("goals")),
                        "assists": number(player.get("assists")),
                        "non_penalty_goals": number(player.get("npg")),
                        "shots_total": number(player.get("shots")),
                        "key_passes": number(player.get("key_passes")),
                        "yellow_cards": number(player.get("yellow_cards")),
                        "red_cards": number(player.get("red_cards")),
                        "expected_goals": number(player.get("xG"), integer=False),
                        "expected_assists": number(player.get("xA"), integer=False),
                    }
                )
            print(f"  {league} {year}: {len(players)} players")
            sys.stdout.flush()
            time.sleep(0.8)

    historical = pd.DataFrame(rows)
    print(f"Fetched {len(historical):,} historical player-season rows.")

    backfill_seasons = {f"{year}-{year + 1}" for year in SEASONS}
    if TARGET.exists():
        existing = pd.read_csv(TARGET, low_memory=False)
        existing = existing[~existing["season"].astype(str).isin(backfill_seasons)]
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, historical], ignore_index=True)
    combined.to_csv(TARGET, index=False)
    print(f"Saved {len(combined):,} rows to {TARGET}")
    print("Seasons:", sorted(combined["season"].dropna().astype(str).unique().tolist()))


if __name__ == "__main__":
    main()
