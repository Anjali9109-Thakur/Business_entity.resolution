"""Export validated-format candidate and prediction TSVs from scored candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def valid_table_name(name: str) -> bool:
    return name.replace("_", "").isalnum()


def sql_literal(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output-dir", default="output", type=Path)
    parser.add_argument("--temp-dir", default="work/duckdb_export_tmp", type=Path)
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--threads", default=1, type=int)
    args = parser.parse_args()
    if not valid_table_name(args.candidate_table):
        raise SystemExit("--candidate-table may contain only letters, digits, and underscores.")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1.")
    if not args.scores.is_file():
        raise SystemExit(f"Scores file not found: {args.scores}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    # A prior interrupted COPY can leave an empty partial output.  These are
    # the two explicit artifacts this command is responsible for regenerating.
    for output_path in (candidate_path, matching_path):
        if output_path.exists():
            output_path.unlink()
    # A 52M-pair string aggregation needs external spill on an 8 GB laptop.
    # This connection does not change the model data; its writable mode only
    # permits DuckDB to use the explicit temporary directory for the query.
    connection = duckdb.connect(str(args.database.resolve()))
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET threads = {args.threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = {sql_literal(args.temp_dir)}")
    scores_path = sql_literal(args.scores)

    connection.execute(
        f"""
        COPY (
            SELECT
                s1.source1_entity_id,
                coalesce(candidate_lists.candidate_entity_ids, '') AS candidate_entity_ids
            FROM (
                SELECT entity_id AS source1_entity_id
                FROM test_s1
            ) AS s1
            LEFT JOIN (
                SELECT
                    source1_entity_id,
                    string_agg(candidate_entity_id, ',' ORDER BY candidate_rank,
                               candidate_entity_id) AS candidate_entity_ids
                FROM {args.candidate_table}
                GROUP BY source1_entity_id
            ) AS candidate_lists USING (source1_entity_id)
        ) TO {sql_literal(candidate_path)} (HEADER, DELIMITER '\\t')
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                s1.source1_entity_id,
                coalesce(match_lists.matched_entity_ids, '') AS matched_entity_ids
            FROM (
                SELECT entity_id AS source1_entity_id
                FROM test_s1
            ) AS s1
            LEFT JOIN (
                SELECT
                    candidate.source1_entity_id,
                    string_agg(candidate.candidate_entity_id, ',' ORDER BY candidate.candidate_rank,
                               candidate.candidate_entity_id) AS matched_entity_ids
                FROM {args.candidate_table} AS candidate
                INNER JOIN read_parquet({scores_path}) AS score
                    USING (source1_entity_id, candidate_entity_id)
                WHERE score.match_probability >= {args.threshold}
                GROUP BY candidate.source1_entity_id
            ) AS match_lists USING (source1_entity_id)
        ) TO {sql_literal(matching_path)} (HEADER, DELIMITER '\\t')
        """
    )
    predicted_pairs = connection.execute(
        f"""
        SELECT count(*) FROM {args.candidate_table} AS candidate
        INNER JOIN read_parquet({scores_path}) AS score
            USING (source1_entity_id, candidate_entity_id)
        WHERE score.match_probability >= {args.threshold}
        """
    ).fetchone()[0]
    candidate_pairs = connection.execute(
        f"SELECT count(*) FROM {args.candidate_table}"
    ).fetchone()[0]
    connection.close()
    print(f"Wrote {candidate_path} ({candidate_pairs:,} candidate pairs).")
    print(f"Wrote {matching_path} ({predicted_pairs:,} predicted pairs at {args.threshold:.3f}).")


if __name__ == "__main__":
    main()
