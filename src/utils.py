import pandas as pd


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


def apply_column_aliases(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    frame = frame.copy()

    for target, possible_names in aliases.items():
        if target in frame.columns:
            continue

        for name in possible_names:
            if name in frame.columns:
                frame = frame.rename(columns={name: target})
                break

    return frame


def ensure_columns(frame: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    frame = frame.copy()

    for column, default_value in defaults.items():
        if column not in frame.columns:
            frame[column] = default_value

    return frame


def numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()

    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    return frame


def safe_mean(series: pd.Series, fallback: float = 0.0) -> float:
    value = pd.to_numeric(series, errors="coerce").mean()

    if pd.isna(value):
        return float(fallback)

    return float(value)


def result_code_from_scores(home_goals: float, away_goals: float) -> int:
    if home_goals > away_goals:
        return 1

    if home_goals == away_goals:
        return 0

    return -1


def class_name(value) -> str:
    mapping = {
        -1: "Away Win",
        0: "Draw",
        1: "Home Win",
        2: "Home Win",
        "-1": "Away Win",
        "0": "Draw",
        "1": "Home Win",
        "2": "Home Win",
        "A": "Away Win",
        "D": "Draw",
        "H": "Home Win",
        "away": "Away Win",
        "draw": "Draw",
        "home": "Home Win",
    }

    return mapping.get(value, str(value))