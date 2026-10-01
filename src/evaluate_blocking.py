"""Measure candidate recall and candidate-set size on a held-out S1 fold."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--validation-fold", default=0, type=int)
    args = parser.parse_args()
    if not args.candidate_table.replace("_", "").isalnum():
        raise SystemExit("--candidate-table may contain only letters, digits, and underscores.")

    connection = duckdb.connect(str(args.database.resolve()), read_only=True)
    result = connection.execute(
        f"""
        WITH validation_truth AS (
            SELECT truth.source1_entity_id, truth.candidate_entity_id
            FROM train_truth AS truth
            WHERE hash(truth.source1_entity_id) % 5 = {args.validation_fold}
        ), recovered_truth AS (
            SELECT count(*) AS recovered
            FROM validation_truth AS truth
            INNER JOIN {args.candidate_table} AS candidate
                USING (source1_entity_id, candidate_entity_id)
        ), candidate_counts AS (
            SELECT source1_entity_id, count(*) AS candidate_count
            FROM {args.candidate_table}
            GROUP BY source1_entity_id
        )
        SELECT
            (SELECT count(*) FROM validation_truth) AS true_links,
            (SELECT recovered FROM recovered_truth) AS recovered_links,
            (SELECT count(*) FROM candidate_counts) AS s1_with_candidates,
            (SELECT round(avg(candidate_count), 3) FROM candidate_counts) AS mean_candidates,
            (SELECT quantile_cont(candidate_count, 0.95) FROM candidate_counts) AS p95_candidates,
            (SELECT max(candidate_count) FROM candidate_counts) AS max_candidates
        """
    ).fetchone()
    true_links, recovered, s1s, mean_count, p95, max_count = result
    recall = recovered / true_links if true_links else 0.0
    print(f"Validation true links: {true_links:,}")
    print(f"Recovered by final candidate set: {recovered:,}")
    print(f"Candidate recall ceiling: {recall:.4%}")
    print(
        "Candidate volume: "
        f"S1 with candidates={s1s:,}; mean={mean_count}; p95={p95}; max={max_count}"
    )
    connection.close()


if __name__ == "__main__":
    main()
