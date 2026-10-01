"""Convert train_ground_truth.tsv into a pair-level DuckDB label table.

The provided truth file has one Source-1 ID per row and a comma-separated list
of targets.  Entity-resolution modelling needs one row for every true pair.
This step creates that compact pair table without loading all labels into RAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def quoted_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def truth_relation(path: Path, row_limit: int) -> str:
    relation = (
        f"read_csv({quoted_path(path)}, delim = '\\t', header = true, "
        "all_varchar = true, strict_mode = true)"
    )
    if row_limit:
        return f"(SELECT * FROM {relation} LIMIT {row_limit})"
    return relation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument(
        "--row-limit",
        default=0,
        type=int,
        help="Only for a smoke test; 0 reads every ground-truth row.",
    )
    args = parser.parse_args()

    database = args.database.resolve()
    ground_truth = args.ground_truth.resolve()
    if not database.is_file():
        raise SystemExit(f"Database not found: {database}. Run build_database.py first.")
    if not ground_truth.is_file():
        raise SystemExit(f"Ground truth not found: {ground_truth}")

    connection = duckdb.connect(str(database))
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET threads = {args.threads}")
    relation = truth_relation(ground_truth, args.row_limit)

    connection.execute("DROP TABLE IF EXISTS train_truth")
    connection.execute(
        f"""
        CREATE TABLE train_truth AS
        SELECT
            trim(source1_entity_id) AS source1_entity_id,
            trim(item.candidate_entity_id) AS candidate_entity_id,
            split_part(trim(item.candidate_entity_id), '-', 1) AS candidate_source
        FROM {relation} AS truth,
             UNNEST(string_split(coalesce(truth.matched_entity_ids, ''), ','))
                 AS item(candidate_entity_id)
        WHERE trim(coalesce(truth.matched_entity_ids, '')) <> ''
              AND trim(item.candidate_entity_id) <> ''
        """
    )
    connection.execute(
        "CREATE OR REPLACE VIEW train_targets AS "
        "SELECT 'S2' AS candidate_source, * FROM train_s2 "
        "UNION ALL SELECT 'S3' AS candidate_source, * FROM train_s3"
    )
    connection.execute(
        "CREATE OR REPLACE VIEW test_targets AS "
        "SELECT 'S2' AS candidate_source, * FROM test_s2 "
        "UNION ALL SELECT 'S3' AS candidate_source, * FROM test_s3"
    )

    pair_count = connection.execute("SELECT count(*) FROM train_truth").fetchone()[0]
    by_source = connection.execute(
        "SELECT candidate_source, count(*) FROM train_truth "
        "GROUP BY candidate_source ORDER BY candidate_source"
    ).fetchall()
    print(f"Created train_truth with {pair_count:,} positive pairs.")
    print("Positive pairs by target source: " + ", ".join(f"{k}={v:,}" for k, v in by_source))
    print("Created train_targets and test_targets views.")
    connection.close()


if __name__ == "__main__":
    main()
