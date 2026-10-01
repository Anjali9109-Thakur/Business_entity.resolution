"""Check whether candidate-table scan order matches the streamed score Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--rows", default=500_000, type=int)
    args = parser.parse_args()
    if not args.candidate_table.replace("_", "").isalnum():
        raise SystemExit("Invalid table name.")

    connection = duckdb.connect(str(args.database.resolve()), read_only=True)
    candidates = connection.execute(
        f"SELECT source1_entity_id, candidate_entity_id "
        f"FROM {args.candidate_table} LIMIT {args.rows}"
    ).fetchall()
    connection.close()
    runs = 0
    seen: set[str] = set()
    previous = None
    for source1_id, _ in candidates:
        if source1_id != previous:
            runs += 1
            seen.add(source1_id)
            previous = source1_id
    print(f"Candidate sample rows: {len(candidates):,}")
    print(f"Source-1 runs: {runs:,}; distinct Source-1 IDs: {len(seen):,}")
    print(f"Contiguous groups: {'YES' if runs == len(seen) else 'NO'}")

    parquet = pq.ParquetFile(args.scores)
    score_rows = []
    for batch in parquet.iter_batches(
        batch_size=args.rows,
        columns=["source1_entity_id", "candidate_entity_id"],
    ):
        ids1 = batch.column(0).to_pylist()
        ids2 = batch.column(1).to_pylist()
        score_rows.extend(zip(ids1, ids2))
        if len(score_rows) >= len(candidates):
            break
    score_rows = score_rows[: len(candidates)]
    aligned = candidates == score_rows
    print(f"Score sample rows: {len(score_rows):,}")
    score_runs = 0
    score_seen: set[str] = set()
    previous = None
    for source1_id, _ in score_rows:
        if source1_id != previous:
            score_runs += 1
            score_seen.add(source1_id)
            previous = source1_id
    print(f"Score Source-1 runs: {score_runs:,}; distinct Source-1 IDs: {len(score_seen):,}")
    print(f"Score contiguous groups: {'YES' if score_runs == len(score_seen) else 'NO'}")
    print(f"Candidate/score Source-1 group order aligned: "
          f"{'YES' if [row[0] for row in candidates] == [row[0] for row in score_rows] else 'NO'}")
    print(f"Candidate/score order aligned: {'YES' if aligned else 'NO'}")
    if not aligned:
        for index, (candidate, score) in enumerate(zip(candidates, score_rows)):
            if candidate != score:
                print(f"First mismatch at row {index}: candidate={candidate}, score={score}")
                break


if __name__ == "__main__":
    main()
