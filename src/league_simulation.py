from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction import predict_match

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_fixtures.csv"
FIXTURES_FALLBACK_PATH = PROJECT_ROOT / "data" / "processed" / "upcoming_fixtures.csv"
STANDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_standings.csv"
STANDINGS_FALLBACK_PATH = PROJECT_ROOT / "data" / "processed" / "current_tables.csv"

UPCOMING_STATUS = {
    "NS",
    "TBD",
    "TBA",
    "TIMED",
    "SCHEDULED",
    "POSTPONED",
    "PST",
    "POSTP",
}

FINISHED_STATUS = {
    "FT",
    "AET",
    "PEN",
    "FT_PEN",
    "ABD",
    "AWD",
    "WO",
    "CANC",
    "CANCELED",
    "CANCELLED",
    "FINISHED",
    "MATCH FINISHED",
}


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = (
        frame.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )
    return frame


def apply_column_aliases(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    frame = frame.copy()
    for target, names in aliases.items():
        if target in frame.columns:
            continue
        for name in names:
            if name in frame.columns:
                frame = frame.rename(columns={name: target})
                break
    return frame


def ensure_columns(frame: pd.DataFrame, defaults: dict[str, object]) -> pd.DataFrame:
    frame = frame.copy()
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def _load_csv(
    path: Path,
    aliases: dict[str, list[str]],
    defaults: dict[str, object],
    date_columns: list[str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()

    frame = normalise_columns(frame)
    frame = apply_column_aliases(frame, aliases)
    frame = ensure_columns(frame, defaults)

    if date_columns:
        for column in date_columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")

    return frame


def _has_real_rows(frame: pd.DataFrame, required_columns: list[str]) -> bool:
    if frame is None or frame.empty:
        return False
    if not required_columns:
        return True
    for column in required_columns:
        if column not in frame.columns:
            return False
        values = frame[column].dropna().astype(str).str.strip()
        values = values[~values.str.lower().isin(["", "nan", "none", "unknown", "unknown team", "unknown competition"])]
        if len(values) == 0:
            return False
    return True


def _load_csv_with_fallback(
    primary_path: Path,
    fallback_path: Path,
    aliases: dict[str, list[str]],
    defaults: dict[str, object],
    date_columns: list[str] | None = None,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    required_columns = required_columns or []
    primary = _load_csv(primary_path, aliases, defaults, date_columns)
    if _has_real_rows(primary, required_columns):
        return primary
    fallback = _load_csv(fallback_path, aliases, defaults, date_columns)
    if _has_real_rows(fallback, required_columns):
        return fallback
    return primary if primary is not None and not primary.empty else fallback


def _as_text(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
    return frame


def _as_numeric(frame: pd.DataFrame, columns: list[str], default: float = 0.0) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return frame


def load_api_football_fixtures() -> pd.DataFrame:
    aliases = {
        "competition": ["competition", "league", "competition_name"],
        "competition_code": ["competition_code", "league_code", "division", "div"],
        "season": ["season", "season_id", "year"],
        "fixture_id": ["fixture_id", "id", "match_id"],
        "fixture_date": ["fixture_date", "date", "utc_date", "kickoff", "kickoff_time"],
        "status": ["status", "status_long"],
        "status_short": ["status_short", "short_status", "status_code"],
        "round": ["round", "matchday", "stage"],
        "home_team_id": ["home_team_id", "home_id"],
        "away_team_id": ["away_team_id", "away_id"],
        "home_team": ["home_team", "home", "home_team_name"],
        "away_team": ["away_team", "away", "away_team_name"],
        "home_goals": ["home_goals", "home_score", "fthg", "hg"],
        "away_goals": ["away_goals", "away_score", "ftag", "ag"],
    }
    defaults = {
        "competition": "Unknown Competition",
        "competition_code": "",
        "season": "Unknown Season",
        "fixture_id": pd.NA,
        "fixture_date": pd.NaT,
        "status": "",
        "status_short": "",
        "round": "",
        "home_team": "Unknown Home",
        "away_team": "Unknown Away",
        "home_goals": pd.NA,
        "away_goals": pd.NA,
    }
    frame = _load_csv_with_fallback(
        FIXTURES_PATH,
        FIXTURES_FALLBACK_PATH,
        aliases,
        defaults,
        date_columns=["fixture_date"],
        required_columns=["competition", "season", "home_team", "away_team"],
    )
    if frame.empty:
        return frame
    frame = _as_text(frame, ["competition", "competition_code", "season", "status", "status_short", "round", "home_team", "away_team"])
    frame = _as_numeric(frame, ["home_goals", "away_goals"], default=np.nan)
    return frame.sort_values("fixture_date", na_position="last") if "fixture_date" in frame.columns else frame


def load_api_football_standings() -> pd.DataFrame:
    aliases = {
        "competition": ["competition", "league", "competition_name"],
        "competition_code": ["competition_code", "league_code", "division", "div"],
        "season": ["season", "season_id", "year"],
        "group": ["group"],
        "rank": ["rank", "position", "pos"],
        "team_id": ["team_id", "teamid"],
        "team_name": ["team_name", "team", "name"],
        "points": ["points", "pts"],
        "played": ["played", "played_games", "matches_played", "mp"],
        "win": ["win", "won", "wins", "w"],
        "draw": ["draw", "drawn", "draws", "d"],
        "lose": ["lose", "loss", "lost", "losses", "l"],
        "goals_for": ["goals_for", "gf"],
        "goals_against": ["goals_against", "ga"],
        "goal_diff": ["goal_diff", "gd", "goal_difference"],
        "form": ["form"],
    }
    defaults = {
        "competition": "Unknown Competition",
        "competition_code": "",
        "season": "Unknown Season",
        "group": "",
        "rank": 0,
        "team_name": "Unknown Team",
        "points": 0,
        "played": 0,
        "win": 0,
        "draw": 0,
        "lose": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
        "form": "",
    }
    frame = _load_csv_with_fallback(
        STANDINGS_PATH,
        STANDINGS_FALLBACK_PATH,
        aliases,
        defaults,
        required_columns=["competition", "season", "team_name"],
    )
    if frame.empty:
        return frame

    frame = _as_text(frame, ["competition", "competition_code", "season", "group", "team_name", "form"])
    frame = _as_numeric(frame, ["rank", "points", "played", "win", "draw", "lose", "goals_for", "goals_against", "goal_diff"])
    frame = frame[~frame["team_name"].astype(str).str.strip().str.lower().isin(["", "unknown team", "nan", "none"])]
    frame = frame[~frame["competition"].astype(str).str.strip().str.lower().isin(["", "unknown competition", "nan", "none"])]

    if frame.empty:
        return frame

    if (frame["rank"] <= 0).all():
        frame = frame.sort_values(["competition", "season", "points", "goal_diff", "goals_for"], ascending=[True, True, False, False, False])
        frame["rank"] = frame.groupby(["competition", "season"]).cumcount() + 1

    frame["rank"] = frame["rank"].astype(int)
    return frame.sort_values(["competition", "season", "rank"], na_position="last")


def _clean_status(value) -> str:
    return str(value).strip().upper()


def _fixture_has_score(row: pd.Series) -> bool:
    for column in ["home_goals", "away_goals"]:
        value = row.get(column)
        if pd.notna(value):
            text = str(value).strip().lower()
            if text not in {"", "nan", "none", "<na>"}:
                return True
    return False


def _fixture_is_upcoming(row: pd.Series) -> bool:
    if _fixture_has_score(row):
        return False

    status_short = _clean_status(row.get("status_short", ""))
    status = _clean_status(row.get("status", ""))
    combined = f"{status_short} {status}".strip()

    if any(value and value in FINISHED_STATUS for value in [status_short, status]):
        return False
    if "FINISHED" in combined or "CANCEL" in combined or "ABANDON" in combined or "AWARDED" in combined:
        return False
    if any(value and value in UPCOMING_STATUS for value in [status_short, status]):
        return True
    if any(token in combined for token in ["NOT STARTED", "SCHEDULED", "TIME TO BE DEFINED", "TIMED", "POSTPONED"]):
        return True

    fixture_date = row.get("fixture_date")
    if pd.notna(fixture_date):
        now = pd.Timestamp.now(tz="UTC")
        date = pd.to_datetime(fixture_date, errors="coerce", utc=True)
        if pd.notna(date) and date >= now:
            return True

    return not combined


def upcoming_fixtures(fixtures: pd.DataFrame, competition: str, season: str) -> pd.DataFrame:
    if fixtures is None or fixtures.empty:
        return pd.DataFrame()

    filtered = fixtures.copy()
    filtered = filtered[filtered["competition"].astype(str).str.strip() == str(competition).strip()]
    filtered = filtered[filtered["season"].astype(str).str.strip() == str(season).strip()]
    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered[filtered.apply(_fixture_is_upcoming, axis=1)].copy()
    if "fixture_date" in filtered.columns:
        filtered = filtered.sort_values("fixture_date", na_position="last")
    return filtered


def _fallback_probs() -> tuple[float, float, float]:
    return 0.45, 0.25, 0.30


def _normalise_probs(home_win: float, draw: float, away_win: float) -> tuple[float, float, float]:
    values = np.array([home_win, draw, away_win], dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    total = values.sum()
    if total <= 0:
        return _fallback_probs()
    values = values / total
    return float(values[0]), float(values[1]), float(values[2])


def match_probabilities(model, metrics: dict | None, matches: pd.DataFrame, home_team: str, away_team: str) -> tuple[float, float, float]:
    if model is None or matches is None or matches.empty:
        return _fallback_probs()
    try:
        prediction = predict_match(model, matches, home_team, away_team, metrics)
        probabilities = prediction.get("probabilities")
        if probabilities is None or probabilities.empty:
            return _fallback_probs()
        home_win = float(probabilities.loc[probabilities["Outcome"] == "Home Win", "Probability"].sum())
        draw = float(probabilities.loc[probabilities["Outcome"] == "Draw", "Probability"].sum())
        away_win = float(probabilities.loc[probabilities["Outcome"] == "Away Win", "Probability"].sum())
        return _normalise_probs(home_win, draw, away_win)
    except Exception:
        return _fallback_probs()


def build_fixture_probabilities(model, metrics: dict | None, matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures is None or fixtures.empty:
        return pd.DataFrame()

    records = []
    for _, fixture in fixtures.iterrows():
        home_team = str(fixture.get("home_team", "")).strip()
        away_team = str(fixture.get("away_team", "")).strip()
        if not home_team or not away_team:
            continue

        home_win_prob, draw_prob, away_win_prob = match_probabilities(model, metrics, matches, home_team, away_team)
        records.append(
            {
                "fixture_id": fixture.get("fixture_id"),
                "fixture_date": fixture.get("fixture_date"),
                "home_team": home_team,
                "away_team": away_team,
                "home_win_prob": home_win_prob,
                "draw_prob": draw_prob,
                "away_win_prob": away_win_prob,
            }
        )

    return pd.DataFrame(records)


def _current_table_summary(standings: pd.DataFrame) -> pd.DataFrame:
    if standings is None or standings.empty:
        return pd.DataFrame()
    rows = []
    total_teams = len(standings)
    for _, row in standings.iterrows():
        rank = int(row.get("rank", 0)) if pd.notna(row.get("rank", 0)) else 0
        rows.append(
            {
                "team_name": str(row.get("team_name", "")),
                "current_rank": rank,
                "current_points": float(row.get("points", 0)),
                "expected_points": float(row.get("points", 0)),
                "expected_rank": float(rank if rank else total_teams),
                "champion_prob": 1.0 if rank == 1 else 0.0,
                "top_4_prob": 1.0 if 1 <= rank <= 4 else 0.0,
                "top_6_prob": 1.0 if 1 <= rank <= 6 else 0.0,
                "bottom_3_prob": 1.0 if rank > total_teams - 3 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["expected_rank", "expected_points"], ascending=[True, False]).reset_index(drop=True)


def simulate_league_table(fixtures_probs: pd.DataFrame, standings: pd.DataFrame, simulations: int = 400) -> pd.DataFrame:
    if standings is None or standings.empty:
        return pd.DataFrame()
    if fixtures_probs is None or fixtures_probs.empty:
        return _current_table_summary(standings)

    standings = standings.copy()
    team_names = standings["team_name"].dropna().astype(str).tolist()
    if not team_names:
        return pd.DataFrame()

    base_points = standings.set_index("team_name")["points"].to_dict()
    base_goal_diff = standings.set_index("team_name")["goal_diff"].to_dict()
    base_goals_for = standings.set_index("team_name")["goals_for"].to_dict()
    base_rank = standings.set_index("team_name")["rank"].to_dict()
    valid_teams = set(team_names)

    fixtures_probs = fixtures_probs[
        fixtures_probs["home_team"].isin(valid_teams) & fixtures_probs["away_team"].isin(valid_teams)
    ].copy()
    if fixtures_probs.empty:
        return _current_table_summary(standings)

    sim_total = max(1, int(simulations))
    team_rank_counts = {team: np.zeros(len(team_names) + 1, dtype=int) for team in valid_teams}
    fixture_records = fixtures_probs.to_dict("records")

    for _ in range(sim_total):
        simulated_points = {team: float(base_points.get(team, 0.0)) for team in valid_teams}

        for fixture in fixture_records:
            home = fixture["home_team"]
            away = fixture["away_team"]
            home_win, draw, away_win = _normalise_probs(
                fixture.get("home_win_prob", 0.45),
                fixture.get("draw_prob", 0.25),
                fixture.get("away_win_prob", 0.30),
            )
            roll = np.random.random()
            if roll < home_win:
                simulated_points[home] += 3
            elif roll < home_win + draw:
                simulated_points[home] += 1
                simulated_points[away] += 1
            else:
                simulated_points[away] += 3

        table = pd.DataFrame(
            {
                "team_name": list(valid_teams),
                "sim_points": [simulated_points[team] for team in valid_teams],
                "goal_diff": [base_goal_diff.get(team, 0) for team in valid_teams],
                "goals_for": [base_goals_for.get(team, 0) for team in valid_teams],
            }
        )
        table = table.sort_values(["sim_points", "goal_diff", "goals_for"], ascending=False).reset_index(drop=True)
        for rank, team_name in enumerate(table["team_name"], start=1):
            team_rank_counts[team_name][rank] += 1

    rows = []
    for team in sorted(valid_teams, key=lambda value: base_rank.get(value, 999)):
        counts = team_rank_counts[team].astype(float)
        rank_index = np.arange(len(counts))
        expected_rank = float((rank_index * counts).sum() / sim_total)
        expected_points = float(base_points.get(team, 0.0))

        for fixture in fixture_records:
            if fixture["home_team"] == team:
                expected_points += float(fixture["home_win_prob"]) * 3.0 + float(fixture["draw_prob"]) * 1.0
            elif fixture["away_team"] == team:
                expected_points += float(fixture["away_win_prob"]) * 3.0 + float(fixture["draw_prob"]) * 1.0

        rows.append(
            {
                "team_name": team,
                "current_rank": int(base_rank.get(team, 0)),
                "current_points": float(base_points.get(team, 0.0)),
                "expected_points": expected_points,
                "expected_rank": expected_rank,
                "champion_prob": float(counts[1] / sim_total),
                "top_4_prob": float(counts[1:5].sum() / sim_total),
                "top_6_prob": float(counts[1:7].sum() / sim_total),
                "bottom_3_prob": float(counts[-3:].sum() / sim_total) if len(team_names) >= 3 else float(counts[1:].sum() / sim_total),
            }
        )

    return pd.DataFrame(rows).sort_values(["expected_rank", "expected_points"], ascending=[True, False]).reset_index(drop=True)


def load_standings() -> pd.DataFrame:
    return load_api_football_standings()


def load_fixtures() -> pd.DataFrame:
    return load_api_football_fixtures()
