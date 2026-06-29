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


def season_start_year(value) -> int:
    text = str(value).strip()

    if text.lower() in {"", "nan", "none", "unknown season"}:
        return -1

    # When the season has an explicit separator ("2015/16", "1962-63"), the
    # start year is the part before it. Parsing the whole digit run and taking
    # the trailing four digits is wrong: "1962/63" -> "196263" -> "6263" would
    # be read as 2062, which silently corrupts chronological ordering/splits.
    for separator in ("/", "-", "_", " "):
        if separator in text:
            text = text.split(separator)[0].strip()
            break

    digits = "".join(char for char in text if char.isdigit())

    if not digits:
        return -1

    if len(digits) >= 4:
        year = int(digits[:4])
        if 1800 <= year <= 2100:
            return year

    start_yy = int(digits[:2]) if len(digits) >= 2 else int(digits)

    return 1900 + start_yy if start_yy >= 30 else 2000 + start_yy


def format_season(value) -> str:
    start_year = season_start_year(value)

    if start_year < 0:
        return str(value)

    end_year = (start_year + 1) % 100

    return f"{start_year}/{end_year:02d}"


def sorted_seasons(values) -> list:
    unique_values = list(dict.fromkeys(values))

    return sorted(
        unique_values,
        key=lambda value: (season_start_year(value), str(value)),
    )