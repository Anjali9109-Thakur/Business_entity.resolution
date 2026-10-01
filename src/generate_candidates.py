"""Generate the final, capped candidate pairs that the matching model scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from blocking import emitted_address_number_keys_sql, emitted_block_keys_sql


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--source-split", choices=("train", "test"), required=True)
    parser.add_argument("--target-split", choices=("train", "test"), required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument(
        "--validation-fold",
        type=int,
        default=None,
        help="For training only: retain IDs where hash(entity_id) %% 5 equals this value.",
    )
    parser.add_argument(
        "--exclude-validation-fold",
        type=int,
        default=None,
        help="For training only: retain IDs where hash(entity_id) %% 5 is not this value.",
    )
    parser.add_argument("--max-block-size", default=30, type=int)
    parser.add_argument("--max-candidates", default=25, type=int)
    parser.add_argument(
        "--candidate-ranking",
        choices=("weight", "idf"),
        default="weight",
        help=(
            "How to prioritize capped candidates. 'weight' preserves the original "
            "rule-weight sum; 'idf' favors keys that occur in fewer target records."
        ),
    )
    parser.add_argument(
        "--include-address-number-blocks",
        action="store_true",
        help=(
            "Also use the optional <split>_address_number_blocks index. Build it first "
            "with build_address_number_blocks.py."
        ),
    )
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument(
        "--temp-directory",
        default=Path("work/duckdb_temp"),
        type=Path,
        help="Disk directory DuckDB may use for low-memory sort/hash spilling.",
    )
    args = parser.parse_args()

    if not args.output_table.replace("_", "").isalnum():
        raise SystemExit("--output-table may contain only letters, digits, and underscores.")
    if args.validation_fold is not None and args.exclude_validation_fold is not None:
        raise SystemExit("Use either --validation-fold or --exclude-validation-fold, not both.")
    for argument_name, fold in (
        ("--validation-fold", args.validation_fold),
        ("--exclude-validation-fold", args.exclude_validation_fold),
    ):
        if fold is not None and not 0 <= fold <= 4:
            raise SystemExit(f"{argument_name} must be an integer from 0 through 4.")

    database = args.database.resolve()
    if not database.is_file():
        raise SystemExit(f"Database not found: {database}")
    source_table = f"{args.source_split}_s1"
    target_blocks = f"{args.target_split}_target_blocks"
    address_number_blocks = f"{args.target_split}_address_number_blocks"
    fold_filter = ""
    if args.validation_fold is not None:
        fold_filter = f"WHERE hash(entity_id) % 5 = {args.validation_fold}"
    elif args.exclude_validation_fold is not None:
        fold_filter = f"WHERE hash(entity_id) % 5 <> {args.exclude_validation_fold}"

    connection = duckdb.connect(str(database))
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET threads = {args.threads}")
    # Candidate ranking can involve tens of millions of intermediate hits.
    # The output order is explicitly defined by candidate_rank, so preserving
    # insertion order offers no benefit and wastes RAM.  A disk temp directory
    # lets DuckDB complete the external sort on an 8 GB laptop.
    temp_directory = args.temp_directory.resolve()
    temp_directory.mkdir(parents=True, exist_ok=True)
    temp_directory_sql = temp_directory.as_posix().replace("'", "''")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = '{temp_directory_sql}'")
    if args.include_address_number_blocks:
        exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [address_number_blocks],
        ).fetchone()[0]
        if not exists:
            raise SystemExit(
                f"Table {address_number_blocks} does not exist. "
                f"Run build_address_number_blocks.py --split {args.target_split} first."
            )
    connection.execute(f"DROP TABLE IF EXISTS {args.output_table}")
    ranking_expression = (
        "blocking_score"
        if args.candidate_ranking == "weight"
        # Ten million is deliberately a stable scale close to the combined
        # target-source size.  Only relative rank matters here.  Keeping the
        # original blocking_score in the output avoids changing model features.
        else "idf_blocking_score"
    )
    connection.execute(
        f"""
        CREATE TABLE {args.output_table} AS
        WITH source_records AS (
            SELECT * FROM {source_table} {fold_filter}
        ), source_keys AS (
            {emitted_block_keys_sql('source_records', 'source', 'source1_entity_id')}
            {"UNION ALL" if args.include_address_number_blocks else ""}
            {emitted_address_number_keys_sql('source_records', 'source', 'source1_entity_id') if args.include_address_number_blocks else ""}
        ), all_target_blocks AS (
            SELECT candidate_entity_id, block_weight, block_key, block_size
            FROM {target_blocks}
            {"UNION ALL SELECT candidate_entity_id, block_weight, block_key, block_size FROM " + address_number_blocks if args.include_address_number_blocks else ""}
        ), raw_hits AS (
            SELECT
                source_keys.source1_entity_id,
                blocks.candidate_entity_id,
                blocks.block_weight,
                blocks.block_size
            FROM source_keys
            INNER JOIN all_target_blocks AS blocks USING (block_key)
            WHERE blocks.block_size <= {args.max_block_size}
        ), aggregated_hits AS (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                sum(block_weight) AS blocking_score,
                sum(block_weight * ln(1.0 + 10000000.0 / block_size))
                    AS idf_blocking_score,
                max(block_weight) AS strongest_rule_weight
            FROM raw_hits
            GROUP BY source1_entity_id, candidate_entity_id
        ), ranked_hits AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY {ranking_expression} DESC,
                             strongest_rule_weight DESC,
                             candidate_entity_id
                ) AS candidate_rank
            FROM aggregated_hits
        )
        SELECT
            source1_entity_id,
            candidate_entity_id,
            blocking_score,
            strongest_rule_weight,
            candidate_rank
        FROM ranked_hits
        WHERE candidate_rank <= {args.max_candidates}
        """
    )
    pair_count = connection.execute(f"SELECT count(*) FROM {args.output_table}").fetchone()[0]
    s1_with_candidates = connection.execute(
        f"SELECT count(DISTINCT source1_entity_id) FROM {args.output_table}"
    ).fetchone()[0]
    total_s1 = connection.execute(f"SELECT count(*) FROM {source_table} {fold_filter}").fetchone()[0]
    distribution = connection.execute(
        f"""
        SELECT
            round(avg(candidate_count), 3),
            quantile_cont(candidate_count, 0.95),
            max(candidate_count)
        FROM (
            SELECT source1_entity_id, count(*) AS candidate_count
            FROM {args.output_table}
            GROUP BY source1_entity_id
        )
        """
    ).fetchone()
    print(f"Created {args.output_table}: {pair_count:,} candidate pairs.")
    print(f"Source-1 records with candidates: {s1_with_candidates:,} / {total_s1:,}")
    if distribution[0] is not None:
        print(
            "Candidates per non-empty Source-1 record: "
            f"mean={distribution[0]}, p95={distribution[1]}, max={distribution[2]}"
        )
    connection.close()


if __name__ == "__main__":
    main()
