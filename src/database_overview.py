"""Print the relevant DuckDB table names and row counts for experiment planning."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    args = parser.parse_args()
    connection = duckdb.connect(str(args.database.resolve()), read_only=True)
    tables = connection.execute(
        """
        SELECT table_name
        FROM duckdb_tables()
        WHERE table_name LIKE '%candidates%'
           OR table_name LIKE '%target_blocks%'
           OR table_name LIKE '%address_number_blocks%'
           OR table_name IN ('train_truth', 'model_training_features')
        ORDER BY table_name
        """
    ).fetchall()
    for (name,) in tables:
        rows = connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        print(f"{name}: {rows:,} rows")
    connection.close()


if __name__ == "__main__":
    main()
