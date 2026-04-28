#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs_cubi"
OUTPUT_CSV = ROOT / "training.csv"


def read_existing_rows(csv_path: Path):
    if not csv_path.exists():
        return [], set(), []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        fieldnames = reader.fieldnames or []

    existing_folders = {
        row.get("folder_name", "")
        for row in existing_rows
        if row.get("folder_name")
    }
    return existing_rows, existing_folders, fieldnames


def load_run_rows(runs_dir: Path, existing_folders):
    new_rows = []
    new_keys = set()

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        folder_name = run_dir.name
        if folder_name in existing_folders:
            continue

        args_path = run_dir / "args.json"
        if not args_path.exists():
            continue

        with args_path.open("r", encoding="utf-8") as f:
            args_data = json.load(f)

        if not isinstance(args_data, dict):
            continue

        row = {"folder_name": folder_name, "name": ""}
        row.update(args_data)
        row["folder_name"] = folder_name
        row["name"] = row.get("name", "") or ""

        new_rows.append(row)
        new_keys.update(row.keys())

    return new_rows, new_keys


def build_fieldnames(existing_fieldnames, existing_rows, new_rows, new_keys):
    required_first = ["folder_name", "name"]

    ordered_keys = []
    for key in existing_fieldnames:
        if key not in ordered_keys:
            ordered_keys.append(key)

    if not ordered_keys:
        seen = set(required_first)
        for row in existing_rows + new_rows:
            for key in row.keys():
                if key in seen:
                    continue
                seen.add(key)
                ordered_keys.append(key)
    else:
        for key in new_keys:
            if key not in ordered_keys and key not in required_first:
                ordered_keys.append(key)

    for key in reversed(required_first):
        if key in ordered_keys:
            ordered_keys.remove(key)
        ordered_keys.insert(0, key)

    return ordered_keys


def normalize_rows(rows, fieldnames):
    normalized = []
    for row in rows:
        normalized.append({key: row.get(key, "") for key in fieldnames})
    return normalized


def main():
    if not RUNS_DIR.exists() or not RUNS_DIR.is_dir():
        raise FileNotFoundError(f"Missing runs directory: {RUNS_DIR}")

    existing_rows, existing_folders, existing_fieldnames = read_existing_rows(OUTPUT_CSV)
    new_rows, new_keys = load_run_rows(RUNS_DIR, existing_folders)

    all_rows = existing_rows + new_rows
    fieldnames = build_fieldnames(existing_fieldnames, existing_rows, new_rows, new_keys)
    all_rows = normalize_rows(all_rows, fieldnames)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(new_rows)} new rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
