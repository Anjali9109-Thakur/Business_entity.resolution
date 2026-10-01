"""Build an optional, compact index of country-scoped address-number keys."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from blocking import emitted_address_number_keys_sql


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()

    database = args.database.resolve()
    if not database.is_file():
        raise SystemExit(f"Database not found: {database}. Run build_database.py first.")
    target_view = f"{args.split}_targets"
    output_table = f"{args.split}_address_number_blocks"

    connection = duckdb.connect(str(database))
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET threads = {args.threads}")
    connection.execute(f"DROP TABLE IF EXISTS {output_table}")
    connection.execute(
        f"""
        CREATE TABLE {output_table} AS
        WITH emitted_keys AS (
            {emitted_address_number_keys_sql(target_view, 'target', 'candidate_entity_id')}
        )
        SELECT
            candidate_entity_id,
            block_weight,
            block_key,
            count(*) OVER (PARTITION BY block_key) AS block_size
        FROM emitted_keys
        """
    )
    rows, keys = connection.execute(
        f"SELECT count(*), count(DISTINCT block_key) FROM {output_table}"
    ).fetchone()
    print(f"Created {output_table}: {rows:,} rows across {keys:,} address-number keys.")
    connection.close()


if __name__ == "__main__":
    main()
