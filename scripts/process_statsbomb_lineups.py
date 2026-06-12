import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_BOMB_ROOT = PROJECT_ROOT / "data" / "raw" / "statsbomb"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "lineups_clean.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "lineups_summary.json"


def find_statsbomb_data_root() -> Path:
    candidates = [
        STATS_BOMB_ROOT / "open-data-master" / "data",
        STATS_BOMB_ROOT / "data",
    ]

    for candidate in candidates:
        if (candidate / "lineups").exists() and (candidate / "matches").exists():
            return candidate

    for candidate in STATS_BOMB_ROOT.rglob("lineups"):
        data_root = candidate.parent
        if (data_root / "matches").exists():
            return data_root

    raise FileNotFoundError("Could not find a StatsBomb data folder containing lineups and matches.")


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def nested_name(value, keys, fallback="Unknown"):
    if isinstance(value, dict):
        for key in keys:
            if value.get(key):
                return str(value.get(key))
    if value is None:
        return fallback
    return str(value)


def build_match_lookup(data_root: Path) -> dict:
    lookup = {}

    for path in (data_root / "matches").rglob("*.json"):
        try:
            matches = read_json(path)
        except Exception:
            continue

        for match in matches:
            match_id = str(match.get("match_id"))

            home_team = nested_name(
                match.get("home_team"),
                ["home_team_name", "team_name", "name"],
            )
            away_team = nested_name(
                match.get("away_team"),
                ["away_team_name", "team_name", "name"],
            )
            competition = nested_name(
                match.get("competition"),
                ["competition_name", "name"],
            )
            season = nested_name(
                match.get("season"),
                ["season_name", "name"],
            )
            stage = nested_name(
                match.get("competition_stage"),
                ["name", "competition_stage_name"],
            )

            lookup[match_id] = {
                "match_id": match_id,
                "date": match.get("match_date"),
                "kick_off": match.get("kick_off"),
                "competition": competition,
                "season": season,
                "stage": stage,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": match.get("home_score"),
                "away_score": match.get("away_score"),
            }

    return lookup


def parse_player_positions(player: dict) -> list[dict]:
    positions = player.get("positions") or []

    if not positions:
        return [
            {
                "position": "Unknown",
                "position_id": None,
                "from_time": None,
                "to_time": None,
                "from_period": None,
                "to_period": None,
                "starter": False,
            }
        ]

    parsed = []

    for position in positions:
        from_time = position.get("from")
        from_period = position.get("from_period")
        start_reason = str(position.get("start_reason") or "").lower()

        starter = False

        if start_reason == "starting xi":
            starter = True

        if from_period == 1 and str(from_time) in {"00:00", "0:00", "00:00:00"}:
            starter = True

        parsed.append(
            {
                "position": position.get("position") or "Unknown",
                "position_id": position.get("position_id"),
                "from_time": from_time,
                "to_time": position.get("to"),
                "from_period": from_period,
                "to_period": position.get("to_period"),
                "starter": starter,
            }
        )

    return parsed


def process_lineups(data_root: Path, match_lookup: dict) -> pd.DataFrame:
    rows = []

    for path in (data_root / "lineups").glob("*.json"):
        match_id = path.stem
        match_info = match_lookup.get(match_id, {})

        try:
            lineups = read_json(path)
        except Exception:
            continue

        home_team = match_info.get("home_team", "Unknown")
        away_team = match_info.get("away_team", "Unknown")

        for team in lineups:
            team_name = team.get("team_name") or nested_name(team.get("team"), ["team_name", "name"])

            if team_name == home_team:
                opponent = away_team
                venue = "Home"
            elif team_name == away_team:
                opponent = home_team
                venue = "Away"
            else:
                opponent = away_team if home_team == "Unknown" else home_team
                venue = "Unknown"

            for player in team.get("lineup", []):
                player_name = player.get("player_name") or player.get("player_nickname") or "Unknown Player"
                player_display_name = player.get("player_nickname") or player_name
                country = nested_name(player.get("country"), ["name", "country_name"])

                for position in parse_player_positions(player):
                    rows.append(
                        {
                            "match_id": match_id,
                            "date": match_info.get("date"),
                            "kick_off": match_info.get("kick_off"),
                            "competition": match_info.get("competition", "Unknown"),
                            "season": match_info.get("season", "Unknown"),
                            "stage": match_info.get("stage", "Unknown"),
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_score": match_info.get("home_score"),
                            "away_score": match_info.get("away_score"),
                            "team": team_name,
                            "opponent": opponent,
                            "venue": venue,
                            "player_id": player.get("player_id"),
                            "player": player_name,
                            "player_display_name": player_display_name,
                            "jersey_number": player.get("jersey_number"),
                            "country": country,
                            "position": position["position"],
                            "position_id": position["position_id"],
                            "from_time": position["from_time"],
                            "to_time": position["to_time"],
                            "from_period": position["from_period"],
                            "to_period": position["to_period"],
                            "starter": position["starter"],
                        }
                    )

    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["starter"] = frame["starter"].fillna(False).astype(bool)
    frame = frame.sort_values(
        ["date", "match_id", "team", "starter", "player"],
        ascending=[False, True, True, False, True],
    )
    frame = frame.drop_duplicates(
        subset=["match_id", "team", "player_id", "position", "from_time", "to_time"],
        keep="first",
    )

    return frame


def main():
    data_root = find_statsbomb_data_root()
    match_lookup = build_match_lookup(data_root)
    lineups = process_lineups(data_root, match_lookup)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lineups.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "statsbomb_data_root": str(data_root),
        "lineup_rows": int(len(lineups)),
        "matches": int(lineups["match_id"].nunique()) if not lineups.empty else 0,
        "competitions": sorted(lineups["competition"].dropna().astype(str).unique().tolist()) if not lineups.empty else [],
        "seasons": sorted(lineups["season"].dropna().astype(str).unique().tolist()) if not lineups.empty else [],
        "teams": int(lineups["team"].nunique()) if not lineups.empty else 0,
        "output": str(OUTPUT_PATH),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"StatsBomb data root: {data_root}")
    print(f"Saved {len(lineups):,} lineup rows to {OUTPUT_PATH}")
    print(f"Matches: {summary['matches']:,}")
    print(f"Teams: {summary['teams']:,}")
    print(f"Competitions: {len(summary['competitions']):,}")
    print(f"Seasons: {len(summary['seasons']):,}")


if __name__ == "__main__":
    main()
