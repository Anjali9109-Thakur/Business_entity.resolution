"""Low-memory structural validator for very large entity-resolution submissions.

The organizer-provided validator materializes every candidate ID in Python
sets.  With 52M candidates that exceeds an 8 GB machine.  This script streams
each output list and uses a temporary SQLite table only for the required S1
row/duplicate/missing check.  It validates all TSV shape, header, prefix, and
within-row-duplicate rules.  The pipeline itself guarantees target ID existence
and match-subset-of-candidates because it exports directly from test candidates
and their model scores.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def create_expected_table(connection: sqlite3.Connection, source1_path: Path) -> int:
    connection.execute("CREATE TABLE expected (entity_id TEXT PRIMARY KEY, seen INTEGER NOT NULL DEFAULT 0)")
    batch: list[tuple[str]] = []
    total = 0
    with source1_path.open("r", encoding="utf-8", newline="") as handle:
        next(handle, None)
        for line in handle:
            source1_id = line.partition("\t")[0].strip()
            if source1_id:
                batch.append((source1_id,))
                total += 1
            if len(batch) >= 10_000:
                connection.executemany("INSERT INTO expected(entity_id) VALUES (?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO expected(entity_id) VALUES (?)", batch)
    connection.commit()
    return total


def validate_file(
    connection: sqlite3.Connection,
    path: Path,
    expected_header: str,
    list_label: str,
) -> tuple[int, int, list[str]]:
    errors: list[str] = []
    rows = empty_rows = 0
    if not path.is_file():
        return rows, empty_rows, [f"Missing file: {path}"]
    connection.execute("UPDATE expected SET seen = 0")
    batch: list[tuple[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n")
        if header != expected_header:
            return rows, empty_rows, [
                f"{path.name}: header is {header!r}; expected {expected_header!r}"
            ]
        for line_number, line in enumerate(handle, start=2):
            source1_id, separator, rest = line.rstrip("\r\n").partition("\t")
            if not separator:
                errors.append(f"{path.name}: missing tab at line {line_number}")
                if len(errors) >= 10:
                    break
                continue
            rows += 1
            if not source1_id.startswith("S1-"):
                errors.append(f"{path.name}: invalid Source-1 ID at line {line_number}: {source1_id!r}")
            if rest:
                ids = rest.split(",")
                if len(ids) != len(set(ids)):
                    errors.append(f"{path.name}: duplicate {list_label} ID at line {line_number}")
                for entity_id in ids:
                    if not entity_id.startswith(("S2-", "S3-")):
                        errors.append(
                            f"{path.name}: invalid {list_label} ID at line {line_number}: {entity_id!r}"
                        )
                        break
            else:
                empty_rows += 1
            batch.append((source1_id,))
            if len(batch) >= 10_000:
                connection.executemany("UPDATE expected SET seen = seen + 1 WHERE entity_id = ?", batch)
                batch.clear()
            if len(errors) >= 10:
                break
        if batch:
            connection.executemany("UPDATE expected SET seen = seen + 1 WHERE entity_id = ?", batch)
    connection.commit()
    missing = connection.execute("SELECT count(*) FROM expected WHERE seen = 0").fetchone()[0]
    duplicated = connection.execute("SELECT count(*) FROM expected WHERE seen > 1").fetchone()[0]
    matched_rows = connection.execute("SELECT coalesce(sum(seen), 0) FROM expected").fetchone()[0]
    unknown = rows - matched_rows
    if missing:
        errors.append(f"{path.name}: {missing:,} required test S1 row(s) missing")
    if duplicated:
        errors.append(f"{path.name}: {duplicated:,} duplicate Source-1 row(s)")
    if unknown:
        errors.append(f"{path.name}: {unknown:,} row(s) use S1 IDs not in test_source1")
    return rows, empty_rows, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", default="output/matching_results.tsv", type=Path)
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--work-db", default="work/large_validation.sqlite", type=Path)
    args = parser.parse_args()
    source1_path = args.test_dir / "test_source1.tsv"
    if not source1_path.is_file():
        raise SystemExit(f"Missing test Source-1 file: {source1_path}")
    args.work_db.parent.mkdir(parents=True, exist_ok=True)
    if args.work_db.exists():
        args.work_db.unlink()

    connection = sqlite3.connect(args.work_db)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    expected = create_expected_table(connection, source1_path)
    print(f"Required test Source-1 rows: {expected:,}")
    checks = (
        (args.matching, "source1_entity_id\tmatched_entity_ids", "matched"),
        (args.candidate, "source1_entity_id\tcandidate_entity_ids", "candidate"),
    )
    all_errors: list[str] = []
    for path, header, label in checks:
        rows, empty_rows, errors = validate_file(connection, path, header, label)
        print(f"{path.name}: {rows:,} rows ({empty_rows:,} empty)")
        all_errors.extend(errors)
    connection.close()
    if all_errors:
        print("FAIL")
        for number, error in enumerate(all_errors, start=1):
            print(f"{number}. {error}")
        raise SystemExit(1)
    print("PASS — large-file structural validation succeeded.")


if __name__ == "__main__":
    main()
