#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def find_event_file(folder):
    return next(folder.glob("events.out.tfevents.*"))


def load_scalars(event_file):
    acc = event_accumulator.EventAccumulator(str(event_file))
    acc.Reload()

    tags = sorted(acc.Tags().get("scalars", []))

    rows_by_step = {}
    for tag in tags:
        for event in acc.Scalars(tag):
            step = int(event.step)
            rows_by_step.setdefault(step, {})[tag] = float(event.value)

    return tags, rows_by_step


def write_csv(output_csv, tags, rows_by_step):
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *tags])
        for step in sorted(rows_by_step):
            values = [rows_by_step[step].get(tag, "") for tag in tags]
            writer.writerow([step, *values])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all scalar tags from a TensorBoard event file to CSV."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing a single events.out.tfevents.* file.",
    )
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    event_file = find_event_file(folder)
    output_csv = folder / "scalars.csv"

    tags, rows_by_step = load_scalars(event_file)
    write_csv(output_csv, tags, rows_by_step)
    print(f"Wrote CSV: {output_csv}")
    print(f"Event file: {event_file}")
    print(f"Scalar tags: {len(tags)}")
    print(f"Steps: {len(rows_by_step)}")


if __name__ == "__main__":
    main()
