from pathlib import Path
import json
import re
import shutil
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "raw" / "uploads",
    ROOT / "data" / "raw" / "update_uploads",
]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MATCH_OUTPUT = PROCESSED_DIR / "matches_clean.csv"
PLAYER_OUTPUT = PROCESSED_DIR / "player_stats.csv"
SUMMARY_OUTPUT = ROOT / "data_update_summary.json"

LEAGUE_MAP = {
    "B1": "Belgian Pro League",
    "D1": "Bundesliga",
    "D2": "Bundesliga 2",
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "G1": "Greek Super League",
    "I1": "Serie A",
    "I2": "Serie B",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "SP1": "La Liga",
    "SP2": "Segunda Division",
    "T1": "Turkish Super Lig",
    "CL": "Champions League",
    "UCL": "Champions League",
    "EL": "Europa League",
    "UEL": "Europa League",
    "ECL": "Europa Conference League",
    "UECL": "Europa Conference League",
    "FAC": "FA Cup",
    "FA": "FA Cup",
    "EFLC": "EFL Cup",
    "LC": "EFL Cup",
    "CDR": "Copa del Rey",
    "CIT": "Coppa Italia",
    "DFB": "DFB Pokal",
    "CDF": "Coupe de France",
}

PLAYER_SEASON_PATTERN = re.compile(r"(20\d{2})[_-](20\d{2})")



def normalise_league_value(value, path: Path | None = None) -> str:
    raw = str(value).strip()
    compact = raw.upper().replace(" ", "").replace("-", "").replace("_", "")
    if compact in LEAGUE_MAP:
        return LEAGUE_MAP[compact]

    context = f"{raw} {path.stem if path is not None else ''}".lower()
    keyword_map = {
        "champions": "Champions League",
        "europa league": "Europa League",
        "conference league": "Europa Conference League",
        "fa cup": "FA Cup",
        "efl cup": "EFL Cup",
        "league cup": "EFL Cup",
        "copa del rey": "Copa del Rey",
        "coppa italia": "Coppa Italia",
        "dfb pokal": "DFB Pokal",
        "coupe de france": "Coupe de France",
    }
    for keyword, label in keyword_map.items():
        if keyword in context:
            return label
    return raw


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(path, encoding=encoding, low_memory=False, on_bad_lines="skip", engine="python")
    return pd.read_csv(path, encoding="latin1", low_memory=False, on_bad_lines="skip", engine="python")


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


def safe_extract_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def extract_zip_sources() -> Path:
    extract_dir = ROOT / ".data_update_zip_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            continue
        for zip_path in raw_dir.rglob("*.zip"):
            resolved = zip_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            target_dir = extract_dir / safe_extract_name(zip_path.stem)
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    for member in archive.namelist():
                        if member.lower().endswith(".csv"):
                            output_path = target_dir / Path(member).name
                            with archive.open(member) as source, open(output_path, "wb") as target:
                                shutil.copyfileobj(source, target)
            except zipfile.BadZipFile:
                continue

    return extract_dir


def source_files() -> list[Path]:
    files = []
    seen = set()
    extract_dir = extract_zip_sources()
    scan_dirs = RAW_DIRS + [extract_dir]

    for raw_dir in scan_dirs:
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
    division = frame["Div"].astype(str).str.strip() if "Div" in frame.columns else pd.Series([path.stem.split()[0]] * len(frame), index=frame.index)
    cleaned["date"] = frame["date"]
    cleaned["season"] = cleaned["date"].apply(season_from_date)
    cleaned["league"] = division.apply(lambda value: normalise_league_value(value, path))
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




def clean_international_results_file(path: Path) -> pd.DataFrame:
    if path.name.lower() != "results.csv":
        return pd.DataFrame()

    frame = read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame()

    frame = frame.copy()
    frame["date"] = parse_date(frame["date"])
    frame = frame[frame["date"].notna()]
    frame = frame[frame["home_score"].notna() & frame["away_score"].notna()]

    cleaned = pd.DataFrame()
    cleaned["date"] = frame["date"]
    cleaned["season"] = cleaned["date"].apply(season_from_date)
    cleaned["league"] = "International"
    cleaned["tournament"] = frame["tournament"].astype(str).str.strip() if "tournament" in frame.columns else "Unknown"
    cleaned["home_team"] = frame["home_team"].astype(str).str.strip()
    cleaned["away_team"] = frame["away_team"].astype(str).str.strip()
    cleaned["home_goals"] = numeric(frame["home_score"]).astype(int)
    cleaned["away_goals"] = numeric(frame["away_score"]).astype(int)
    cleaned["result"] = [result_code(h, a) for h, a in zip(cleaned["home_goals"], cleaned["away_goals"])]
    cleaned["total_goals"] = cleaned["home_goals"] + cleaned["away_goals"]
    cleaned["city"] = frame["city"].astype(str).str.strip() if "city" in frame.columns else "Unknown"
    cleaned["country"] = frame["country"].astype(str).str.strip() if "country" in frame.columns else "Unknown"
    cleaned["neutral"] = frame["neutral"] if "neutral" in frame.columns else False
    cleaned["source_file"] = path.name
    cleaned["data_source"] = "uploaded_international_results"
    return cleaned


def clean_generic_match_file(path: Path) -> pd.DataFrame:
    if path.name.lower() in {"results.csv", "goalscorers.csv", "shootouts.csv", "players.csv", "former_names.csv"}:
        return pd.DataFrame()

    raw = read_csv(path)
    frame = normalise_columns(raw)
    home_col = next((column for column in ["home_team", "hometeam", "home", "home_side", "home_name"] if column in frame.columns), None)
    away_col = next((column for column in ["away_team", "awayteam", "away", "away_side", "away_name"] if column in frame.columns), None)
    home_goals_col = next((column for column in ["home_goals", "home_score", "homegoal", "home_goals_ft", "fthg", "hg"] if column in frame.columns), None)
    away_goals_col = next((column for column in ["away_goals", "away_score", "awaygoal", "away_goals_ft", "ftag", "ag"] if column in frame.columns), None)
    date_col = next((column for column in ["date", "match_date", "utc_date", "fixture_date"] if column in frame.columns), None)

    if home_col is None or away_col is None or home_goals_col is None or away_goals_col is None:
        return pd.DataFrame()

    cleaned = pd.DataFrame()
    cleaned["date"] = parse_date(frame[date_col]) if date_col is not None else pd.NaT
    cleaned["home_team"] = frame[home_col].astype(str).str.strip()
    cleaned["away_team"] = frame[away_col].astype(str).str.strip()
    cleaned["home_goals"] = numeric(frame[home_goals_col]).astype(int)
    cleaned["away_goals"] = numeric(frame[away_goals_col]).astype(int)
    league_col = next((column for column in ["league", "competition", "comp", "division", "div", "league_name", "competition_name", "tournament"] if column in frame.columns), None)
    country_col = next((column for column in ["country", "country_name"] if column in frame.columns), None)
    if league_col is not None:
        cleaned["league"] = frame[league_col].astype(str).str.strip()
    elif country_col is not None:
        cleaned["league"] = frame[country_col].astype(str).str.strip()
    else:
        cleaned["league"] = path.stem
    cleaned["league"] = cleaned["league"].apply(lambda value: normalise_league_value(value, path))
    cleaned["season"] = frame["season"].astype(str).str.strip() if "season" in frame.columns else cleaned["date"].apply(season_from_date)
    cleaned["result"] = [result_code(h, a) for h, a in zip(cleaned["home_goals"], cleaned["away_goals"])]
    cleaned["total_goals"] = cleaned["home_goals"] + cleaned["away_goals"]
    cleaned["source_file"] = path.name
    cleaned["data_source"] = "uploaded_generic_match_data"
    cleaned = cleaned[cleaned["home_team"].ne("") & cleaned["away_team"].ne("")]
    cleaned = cleaned[cleaned["home_team"].str.lower().ne("nan") & cleaned["away_team"].str.lower().ne("nan")]
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
            continue
        cleaned = clean_international_results_file(file)
        if not cleaned.empty:
            frames.append(cleaned)
            continue
        cleaned = clean_generic_match_file(file)
        if not cleaned.empty:
            frames.append(cleaned)

    if not frames:
        raise FileNotFoundError("No match data found. Keep your existing data/processed/matches_clean.csv or add match CSVs to data/raw/uploads.")

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
