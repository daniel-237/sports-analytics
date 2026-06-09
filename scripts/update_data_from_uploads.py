from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "raw" / "update_uploads",
]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MATCH_OUTPUT = PROCESSED_DIR / "matches_clean.csv"
PLAYER_OUTPUT = PROCESSED_DIR / "player_stats.csv"
SUMMARY_OUTPUT = ROOT / "data_update_summary.json"

LEAGUE_MAP = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
}

PLAYER_SEASON_PATTERN = re.compile(r"(20\d{2})[_-](20\d{2})")


def read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "cp1252"]

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                on_bad_lines="skip",
                engine="python",
            )

    return pd.read_csv(
        path,
        encoding="latin1",
        low_memory=False,
        on_bad_lines="skip",
        engine="python",
    )


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = (
        frame.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return frame


def pick(frame: pd.DataFrame, names: list[str], default=0):
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def numeric(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def parse_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce")
    return parsed


def season_from_date(value) -> str:
    if pd.isna(value):
        return "Unknown Season"
    year = int(value.year)
    start = year if int(value.month) >= 7 else year - 1
    return f"{start}/{str(start + 1)[-2:]}"


def result_code(home_goals, away_goals) -> int:
    if home_goals > away_goals:
        return 1
    if home_goals == away_goals:
        return 0
    return -1


def source_files() -> list[Path]:
    files = []
    seen = set()
    for raw_dir in RAW_DIRS:
        if raw_dir.exists():
            for file in raw_dir.rglob("*.csv"):
                resolved = file.resolve()
                if resolved not in seen:
                    files.append(file)
                    seen.add(resolved)
    return sorted(files)


def clean_existing_matches(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    frame = read_csv(path)
    frame = normalise_columns(frame)

    aliases = {
        "home_team": ["hometeam", "home", "home_side"],
        "away_team": ["awayteam", "away", "away_side"],
        "home_goals": ["fthg", "home_score", "hg"],
        "away_goals": ["ftag", "away_score", "ag"],
        "league": ["division", "div", "competition"],
        "season": ["year", "season_name"],
        "date": ["match_date"],
    }

    for target, options in aliases.items():
        if target not in frame.columns:
            for option in options:
                if option in frame.columns:
                    frame = frame.rename(columns={option: target})
                    break

    required = ["home_team", "away_team", "home_goals", "away_goals"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()

    if "date" in frame.columns:
        frame["date"] = parse_date(frame["date"])
    else:
        frame["date"] = pd.NaT

    if "season" not in frame.columns:
        frame["season"] = frame["date"].apply(season_from_date)

    if "league" not in frame.columns:
        frame["league"] = "Unknown League"

    frame["home_goals"] = numeric(frame["home_goals"]).astype(int)
    frame["away_goals"] = numeric(frame["away_goals"]).astype(int)
    frame["result"] = [result_code(h, a) for h, a in zip(frame["home_goals"], frame["away_goals"])]
    frame["total_goals"] = frame["home_goals"] + frame["away_goals"]

    for column in ["home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target"]:
        if column not in frame.columns:
            frame[column] = np.nan

    frame["source_file"] = frame.get("source_file", "existing_matches_clean.csv")
    frame["data_source"] = "existing_processed"
    return frame


def clean_football_data_file(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    if "HomeTeam" not in frame.columns or "AwayTeam" not in frame.columns or "FTHG" not in frame.columns or "FTAG" not in frame.columns:
        return pd.DataFrame()

    frame = frame.copy()
    frame["date"] = parse_date(frame["Date"] if "Date" in frame.columns else pd.Series([pd.NaT] * len(frame)))
    frame = frame[frame["date"].notna()]
    frame = frame[frame["FTHG"].notna() & frame["FTAG"].notna()]

    cleaned = pd.DataFrame()
    division = frame["Div"].astype(str).str.strip() if "Div" in frame.columns else path.stem.split()[0]
    cleaned["date"] = frame["date"]
    cleaned["season"] = cleaned["date"].apply(season_from_date)
    cleaned["league"] = division.map(LEAGUE_MAP).fillna(division)
    cleaned["division_code"] = division
    cleaned["home_team"] = frame["HomeTeam"].astype(str).str.strip()
    cleaned["away_team"] = frame["AwayTeam"].astype(str).str.strip()
    cleaned["home_goals"] = numeric(frame["FTHG"]).astype(int)
    cleaned["away_goals"] = numeric(frame["FTAG"]).astype(int)
    cleaned["result"] = [result_code(h, a) for h, a in zip(cleaned["home_goals"], cleaned["away_goals"])]
    cleaned["total_goals"] = cleaned["home_goals"] + cleaned["away_goals"]

    optional = {
        "HTHG": "half_time_home_goals",
        "HTAG": "half_time_away_goals",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_on_target",
        "AST": "away_shots_on_target",
        "HC": "home_corners",
        "AC": "away_corners",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HY": "home_yellow_cards",
        "AY": "away_yellow_cards",
        "HR": "home_red_cards",
        "AR": "away_red_cards",
        "B365H": "home_odds",
        "B365D": "draw_odds",
        "B365A": "away_odds",
        "AvgH": "avg_home_odds",
        "AvgD": "avg_draw_odds",
        "AvgA": "avg_away_odds",
    }

    for source, target in optional.items():
        cleaned[target] = numeric(frame[source]) if source in frame.columns else np.nan

    cleaned["referee"] = frame["Referee"].astype(str).str.strip() if "Referee" in frame.columns else "Unknown"
    cleaned["source_file"] = path.name
    cleaned["data_source"] = "uploaded_football_data"
    return cleaned


def build_matches() -> pd.DataFrame:
    frames = []
    existing = clean_existing_matches(MATCH_OUTPUT)
    if not existing.empty:
        frames.append(existing)

    for file in source_files():
        cleaned = clean_football_data_file(file)
        if not cleaned.empty:
            frames.append(cleaned)

    if not frames:
        raise FileNotFoundError("No match data found. Keep your existing data/processed/matches_clean.csv or add Football-Data CSVs to data/raw/update_uploads.")

    matches = pd.concat(frames, ignore_index=True, sort=False)
    matches["home_team"] = matches["home_team"].astype(str).str.strip()
    matches["away_team"] = matches["away_team"].astype(str).str.strip()
    matches["league"] = matches["league"].astype(str).str.strip()
    matches["season"] = matches["season"].astype(str).str.strip()
    matches["date"] = parse_date(matches["date"])
    matches["home_goals"] = numeric(matches["home_goals"]).astype(int)
    matches["away_goals"] = numeric(matches["away_goals"]).astype(int)
    matches["result"] = [result_code(h, a) for h, a in zip(matches["home_goals"], matches["away_goals"])]
    matches["total_goals"] = matches["home_goals"] + matches["away_goals"]
    matches = matches.dropna(subset=["date"])
    matches = matches.drop_duplicates(subset=["date", "league", "home_team", "away_team"], keep="last")
    matches = matches.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)
    return matches


def season_from_player_filename(path: Path) -> str | None:
    match = PLAYER_SEASON_PATTERN.search(path.name)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    return f"{start}/{str(end)[-2:]}"


def build_rating(frame: pd.DataFrame) -> pd.Series:
    mins = numeric(frame["minutes"])
    metrics = pd.DataFrame(index=frame.index)
    metrics["attack"] = numeric(frame["goals_p90"]) * 2.5 + numeric(frame["assists_p90"]) * 2.0 + numeric(frame["shots_p90"]) * 0.25
    metrics["passing"] = numeric(frame["pass_accuracy"]) / 100 + numeric(frame["key_passes"]) / 90
    metrics["defence"] = numeric(frame["tackles_p90"]) * 0.6 + numeric(frame["interc_p90"]) * 0.7
    metrics["availability"] = np.minimum(mins / 2500, 1)
    score = metrics.sum(axis=1)
    if score.max() == score.min():
        return pd.Series([6.5] * len(frame), index=frame.index)
    return (5.5 + ((score - score.min()) / (score.max() - score.min())) * 4.5).round(2)


def clean_player_file(path: Path) -> pd.DataFrame:
    season = season_from_player_filename(path)
    if season is None:
        return pd.DataFrame()

    frame = read_csv(path)
    if "Player" not in frame.columns:
        return pd.DataFrame()

    cleaned = pd.DataFrame()
    cleaned["name"] = pick(frame, ["Player"], "Unknown Player").astype(str).str.strip()
    cleaned["team"] = pick(frame, ["Squad"], "Unknown Team").astype(str).str.strip()
    cleaned["position"] = pick(frame, ["Pos"], "Unknown Position").astype(str).str.strip()
    cleaned["season"] = season
    cleaned["nationality"] = pick(frame, ["Nation"], "Unknown").astype(str).str.strip()
    cleaned["competition"] = pick(frame, ["Comp"], "Unknown").astype(str).str.strip()
    cleaned["age"] = numeric(pick(frame, ["Age"], 0))
    cleaned["goals"] = numeric(pick(frame, ["Gls"], 0))
    cleaned["assists"] = numeric(pick(frame, ["Ast"], 0))
    cleaned["minutes"] = numeric(pick(frame, ["Min", "Min_stats_playing_time", "Min_stats_keeper"], 0))
    cleaned["appearances"] = numeric(pick(frame, ["MP", "MP_stats_playing_time", "MP_stats_keeper"], 0))
    cleaned["shots_total"] = numeric(pick(frame, ["Sh"], 0))
    cleaned["shots_on_target"] = numeric(pick(frame, ["SoT"], 0))
    cleaned["pass_accuracy"] = numeric(pick(frame, ["Cmp%", "Cmp%_stats_keeper_adv"], 0))
    cleaned["dribbles"] = numeric(pick(frame, ["Succ"], 0))
    cleaned["tackles"] = numeric(pick(frame, ["Tkl", "TklW", "TklW_stats_misc"], 0))
    cleaned["interceptions"] = numeric(pick(frame, ["Int", "Int_stats_misc"], 0))
    cleaned["yellow_cards"] = numeric(pick(frame, ["CrdY", "CrdY_stats_misc"], 0))
    cleaned["red_cards"] = numeric(pick(frame, ["CrdR", "CrdR_stats_misc"], 0))
    cleaned["duels_won"] = numeric(pick(frame, ["Won"], 0))
    cleaned["crosses"] = numeric(pick(frame, ["Crs", "Crs_stats_misc"], 0))
    cleaned["key_passes"] = numeric(pick(frame, ["KP"], 0))
    cleaned["progressive_carries"] = numeric(pick(frame, ["PrgC", "PrgC_stats_possession"], 0))
    cleaned["progressive_passes"] = numeric(pick(frame, ["PrgP", "PrgP_stats_passing"], 0))
    cleaned["expected_goals"] = numeric(pick(frame, ["xG", "xG_stats_shooting"], 0))
    cleaned["expected_assists"] = numeric(pick(frame, ["xAG", "xAG_stats_passing", "xA"], 0))
    cleaned["clean_sheets"] = numeric(pick(frame, ["CS"], 0))
    cleaned["saves"] = numeric(pick(frame, ["Saves"], 0))
    cleaned["goals_against"] = numeric(pick(frame, ["GA"], 0))

    minutes_safe = cleaned["minutes"].replace(0, np.nan)
    cleaned["goals_p90"] = (cleaned["goals"] / minutes_safe * 90).fillna(0).round(2)
    cleaned["assists_p90"] = (cleaned["assists"] / minutes_safe * 90).fillna(0).round(2)
    cleaned["shots_p90"] = (cleaned["shots_total"] / minutes_safe * 90).fillna(0).round(2)
    cleaned["sot_p90"] = (cleaned["shots_on_target"] / minutes_safe * 90).fillna(0).round(2)
    cleaned["tackles_p90"] = (cleaned["tackles"] / minutes_safe * 90).fillna(0).round(2)
    cleaned["interc_p90"] = (cleaned["interceptions"] / minutes_safe * 90).fillna(0).round(2)
    cleaned["contrib_p90"] = ((cleaned["goals"] + cleaned["assists"]) / minutes_safe * 90).fillna(0).round(2)
    cleaned["cards_p90"] = ((cleaned["yellow_cards"] + cleaned["red_cards"]) / minutes_safe * 90).fillna(0).round(2)
    cleaned["rating"] = build_rating(cleaned)
    cleaned["source_file"] = path.name
    cleaned = cleaned[cleaned["name"].ne("Unknown Player")]
    cleaned = cleaned[cleaned["name"].str.lower().ne("player")]
    return cleaned


def build_players() -> pd.DataFrame:
    frames = []
    for file in source_files():
        cleaned = clean_player_file(file)
        if not cleaned.empty:
            frames.append(cleaned)

    if not frames:
        if PLAYER_OUTPUT.exists():
            return read_csv(PLAYER_OUTPUT)
        raise FileNotFoundError("No player data found.")

    players = pd.concat(frames, ignore_index=True)
    players = players.sort_values(["season", "competition", "team", "name", "minutes"], ascending=[False, True, True, True, False])
    players = players.drop_duplicates(subset=["name", "team", "season", "competition"], keep="first")
    return players.reset_index(drop=True)


def summary(matches: pd.DataFrame, players: pd.DataFrame) -> dict:
    return {
        "matches": int(len(matches)),
        "match_date_min": str(matches["date"].min().date()) if len(matches) else None,
        "match_date_max": str(matches["date"].max().date()) if len(matches) else None,
        "match_seasons": sorted(matches["season"].dropna().astype(str).unique().tolist()),
        "match_leagues": sorted(matches["league"].dropna().astype(str).unique().tolist()),
        "players": int(len(players)),
        "player_seasons": sorted(players["season"].dropna().astype(str).unique().tolist()) if len(players) else [],
        "player_competitions": sorted(players["competition"].dropna().astype(str).unique().tolist()) if "competition" in players.columns and len(players) else [],
    }


def main():
    matches = build_matches()
    players = build_players()
    matches.to_csv(MATCH_OUTPUT, index=False)
    players.to_csv(PLAYER_OUTPUT, index=False)
    info = summary(matches, players)
    SUMMARY_OUTPUT.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved {len(matches):,} matches to {MATCH_OUTPUT}")
    print(f"Saved {len(players):,} players to {PLAYER_OUTPUT}")
    print(f"Match date range: {info['match_date_min']} to {info['match_date_max']}")
    print(f"Match seasons: {len(info['match_seasons'])}")
    print(f"Player seasons: {', '.join(info['player_seasons'])}")


if __name__ == "__main__":
    main()
