"""Stream the ground-truth file and report the number of known match links."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()

    rows = positives = empty = 0
    per_source: Counter[str] = Counter()
    links_per_s1: Counter[int] = Counter()
    with args.ground_truth.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows += 1
            ids = [item for item in (row["matched_entity_ids"] or "").split(",") if item]
            links_per_s1[len(ids)] += 1
            if not ids:
                empty += 1
            for entity_id in ids:
                positives += 1
                per_source[entity_id.split("-", 1)[0]] += 1

    print(f"Source-1 rows: {rows:,}")
    print(f"Singleton rows (empty truth): {empty:,} ({empty / rows:.2%})")
    print(f"Positive links: {positives:,}")
    print(f"Links by target source: {dict(sorted(per_source.items()))}")
    print("Links per Source-1 row:")
    for count, count_rows in sorted(links_per_s1.items()):
        print(f"  {count}: {count_rows:,}")


if __name__ == "__main__":
    main()
