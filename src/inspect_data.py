"""Print a memory-safe summary of the entity-resolution TSV files.

This uses Python's standard-library CSV reader, so it never loads a whole
1+ GB TSV file into memory.  Run it before modelling to confirm file paths,
columns, row counts, and countries.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


EXPECTED = {
    "train": (
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    ),
    "test": (
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ),
}


def inspect_tsv(path: Path) -> tuple[list[str], int, Counter[str]]:
    """Return header, number of data rows, and country frequencies."""
    countries: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames or []
        row_count = 0
        for row in reader:
            row_count += 1
            if "country" in row:
                countries[(row["country"] or "<empty>").strip()] += 1
    return header, row_count, countries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder that directly contains train/ and test/.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    print(f"Data directory: {data_dir}")
    for split, names in EXPECTED.items():
        print(f"\n{split.upper()} FILES")
        for name in names:
            path = data_dir / split / name
            if not path.is_file():
                print(f"  MISSING  {split}/{name}")
                continue
            header, rows, countries = inspect_tsv(path)
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {split}/{name}")
            print(f"    size={size_mb:,.1f} MB  rows={rows:,}")
            print(f"    columns={header}")
            if countries:
                print(f"    countries={dict(sorted(countries.items()))}")


if __name__ == "__main__":
    main()
