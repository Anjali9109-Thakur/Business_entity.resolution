"""Show why sampled labelled links were absent from a candidate table.

This is a small, local-only diagnostic.  It expands the current blocking keys
only for a requested sample of missed validation links, then reports their
shared keys and target-side block sizes.  It helps distinguish a missing key
from a key that was dropped by the block-size/candidate-rank limits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from blocking import emitted_block_keys_sql


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--validation-fold", default=0, type=int)
    parser.add_argument("--limit", default=50, type=int)
    args = parser.parse_args()

    if not args.candidate_table.replace("_", "").isalnum():
        raise SystemExit("--candidate-table may contain only letters, digits, and underscores.")

    database = args.database.resolve()
    connection = duckdb.connect(str(database), read_only=True)
    connection.execute("SET memory_limit = '1GB'")
    connection.execute("SET threads = 2")
    connection.execute(
        f"""
        CREATE TEMP TABLE missed AS
        SELECT truth.source1_entity_id, truth.candidate_entity_id
        FROM train_truth AS truth
        LEFT JOIN {args.candidate_table} AS candidates
          USING (source1_entity_id, candidate_entity_id)
        WHERE hash(truth.source1_entity_id) % 5 = {args.validation_fold}
          AND candidates.source1_entity_id IS NULL
        LIMIT {args.limit}
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW sample_sources AS
        SELECT source.*
        FROM train_s1 AS source
        INNER JOIN (SELECT DISTINCT source1_entity_id FROM missed) AS ids
          ON source.entity_id = ids.source1_entity_id
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW sample_targets AS
        SELECT target.*
        FROM train_targets AS target
        INNER JOIN (SELECT DISTINCT candidate_entity_id FROM missed) AS ids
          ON target.entity_id = ids.candidate_entity_id
        """
    )
    result = connection.execute(
        f"""
        WITH source_keys AS (
            {emitted_block_keys_sql('sample_sources', 'source', 'source1_entity_id')}
        ), target_keys AS (
            {emitted_block_keys_sql('sample_targets', 'target', 'candidate_entity_id')}
        ), shared AS (
            SELECT
                missed.source1_entity_id,
                missed.candidate_entity_id,
                source_keys.block_key,
                source_keys.block_weight,
                blocks.block_size
            FROM missed
            INNER JOIN source_keys USING (source1_entity_id)
            INNER JOIN target_keys
              ON target_keys.candidate_entity_id = missed.candidate_entity_id
             AND target_keys.block_key = source_keys.block_key
            INNER JOIN train_target_blocks AS blocks
              ON blocks.candidate_entity_id = missed.candidate_entity_id
             AND blocks.block_key = source_keys.block_key
        )
        SELECT
            missed.source1_entity_id,
            missed.candidate_entity_id,
            count(shared.block_key) AS shared_key_count,
            min(shared.block_size) AS smallest_shared_block,
            string_agg(
                concat(shared.block_key, ' (size=', shared.block_size, ')'),
                '; ' ORDER BY shared.block_size, shared.block_key
            ) AS shared_keys
        FROM missed
        LEFT JOIN shared USING (source1_entity_id, candidate_entity_id)
        GROUP BY missed.source1_entity_id, missed.candidate_entity_id
        ORDER BY smallest_shared_block NULLS LAST, shared_key_count DESC
        """
    ).fetchall()
    for source_id, candidate_id, count, smallest, keys in result:
        print(f"{source_id} -> {candidate_id}: shared={count}, min_block={smallest}")
        if keys:
            print(f"  {keys}")
    connection.close()


if __name__ == "__main__":
    main()
