from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FBREF_PATH = PROJECT_ROOT / "data" / "raw" / "uploads" / "player_stats_fbref_dashboard_ready.csv"
PLAYER_PATH = PROJECT_ROOT / "data" / "processed" / "player_stats.csv"
BACKUP_PATH = PROJECT_ROOT / "data" / "processed" / "player_stats_before_fbref_merge.csv"

def norm(value):
    return str(value).strip().lower()

def main():
    if not FBREF_PATH.exists():
        raise FileNotFoundError(f"Missing {FBREF_PATH}")

    fbref = pd.read_csv(FBREF_PATH)

    if PLAYER_PATH.exists():
        current = pd.read_csv(PLAYER_PATH)
        current.to_csv(BACKUP_PATH, index=False)
        combined = pd.concat([current, fbref], ignore_index=True, sort=False)
    else:
        combined = fbref.copy()

    for column in ["name", "team", "competition", "season", "born"]:
        if column not in combined.columns:
            combined[column] = ""

    combined["_name"] = combined["name"].map(norm)
    combined["_team"] = combined["team"].map(norm)
    combined["_competition"] = combined["competition"].map(norm)
    combined["_season"] = combined["season"].map(norm)
    combined["_born"] = combined["born"].map(norm)

    sort_columns = []
    for column in ["minutes", "performance_score"]:
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0)
            sort_columns.append(column)

    combined = combined.sort_values(sort_columns, ascending=[False] * len(sort_columns)) if sort_columns else combined
    combined = combined.drop_duplicates(subset=["_name", "_team", "_competition", "_season", "_born"], keep="first")
    combined = combined.drop(columns=["_name", "_team", "_competition", "_season", "_born"])
    combined.to_csv(PLAYER_PATH, index=False)

    print(f"FBref rows: {len(fbref):,}")
    print(f"Final player_stats rows: {len(combined):,}")
    print(f"Saved: {PLAYER_PATH}")
    if BACKUP_PATH.exists():
        print(f"Backup: {BACKUP_PATH}")

if __name__ == "__main__":
    main()
