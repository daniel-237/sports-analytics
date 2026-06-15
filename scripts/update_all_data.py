import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BACKUP_ROOT = PROJECT_ROOT / "data" / "backups" / "api_updates"

UPDATE_SCRIPTS = [
    SCRIPTS_DIR / "update_football_data_api.py",
    SCRIPTS_DIR / "update_api_football_data.py",
]

PROTECTED_FILES = [
    PROCESSED_DIR / "current_tables.csv",
    PROCESSED_DIR / "upcoming_fixtures.csv",
    PROCESSED_DIR / "latest_results.csv",
    PROCESSED_DIR / "api_football_fixtures.csv",
    PROCESSED_DIR / "api_football_standings.csv",
    PROCESSED_DIR / "api_football_injuries.csv",
    PROCESSED_DIR / "api_football_lineups.csv",
]


def csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(pd.read_csv(path, low_memory=False))
    except Exception:
        return 0


def snapshot_files() -> tuple[Path, dict[str, dict]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {}
    for path in PROTECTED_FILES:
        key = str(path.relative_to(PROJECT_ROOT))
        rows = csv_row_count(path)
        backup_path = backup_dir / path.name

        snapshot[key] = {
            "path": path,
            "backup_path": backup_path,
            "existed": path.exists(),
            "rows_before": rows,
            "restored": False,
        }

        if path.exists():
            shutil.copy2(path, backup_path)

    return backup_dir, snapshot


def run_script(path: Path) -> dict:
    if not path.exists():
        return {
            "script": path.name,
            "returncode": 1,
            "status": "missing",
            "stdout": "",
            "stderr": f"{path} does not exist",
        }

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    return {
        "script": path.name,
        "returncode": result.returncode,
        "status": "success" if result.returncode == 0 else "failed",
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def restore_empty_overwrites(snapshot: dict[str, dict]) -> None:
    for info in snapshot.values():
        path = info["path"]
        backup_path = info["backup_path"]
        rows_before = int(info["rows_before"])
        rows_after = csv_row_count(path)

        if rows_before > 0 and rows_after == 0 and backup_path.exists():
            shutil.copy2(backup_path, path)
            info["restored"] = True


def file_summary(snapshot: dict[str, dict]) -> list[dict]:
    rows = []
    for key, info in snapshot.items():
        path = info["path"]
        rows.append(
            {
                "file": key,
                "rows_before": int(info["rows_before"]),
                "rows_after": csv_row_count(path),
                "restored": bool(info["restored"]),
                "exists": path.exists(),
            }
        )
    return rows


def print_script_result(result: dict) -> None:
    print(f"\n[{result['status'].upper()}] {result['script']}")
    if result["stdout"]:
        print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"])


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir, snapshot = snapshot_files()

    print("Starting safe data update...")
    print(f"Backup folder: {backup_dir}")

    results = []
    for script in UPDATE_SCRIPTS:
        result = run_script(script)
        results.append(result)
        print_script_result(result)

    restore_empty_overwrites(snapshot)

    print("\nProtected file summary:")
    for row in file_summary(snapshot):
        restored = "yes" if row["restored"] else "no"
        print(
            f"- {row['file']}: before={row['rows_before']} rows, "
            f"after={row['rows_after']} rows, restored={restored}"
        )

    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        print("\nCompleted with script errors. Existing non-empty CSVs were protected where possible.")
        return 1

    print("\nSafe data update completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
