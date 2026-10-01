"""Materialize memory-safe, normalized source tables in DuckDB.

The original TSVs are too large to hold in 8 GB of RAM.  DuckDB reads them
directly, uses a bounded amount of memory, and spills intermediate work to its
temporary directory on disk.  Later blocking and scoring stages read these
compact tables instead of repeatedly parsing 2+ GB of TSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


LEGAL_SUFFIX_PATTERN = (
    r"(^| )(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|"
    r"corporation|company|co|plc|gmbh|sa|sarl)( |$)"
)


def quoted_path(path: Path) -> str:
    """Return a SQL string literal for a filesystem path."""
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def source_relation(path: Path, row_limit: int) -> str:
    """Build the DuckDB relation that reads one input TSV."""
    relation = (
        f"read_csv({quoted_path(path)}, delim = '\\t', header = true, "
        "all_varchar = true, strict_mode = true)"
    )
    if row_limit:
        return f"(SELECT * FROM {relation} LIMIT {row_limit})"
    return relation


def create_clean_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    input_path: Path,
    row_limit: int,
) -> None:
    """Create a compact source table with deterministic normalized fields."""
    relation = source_relation(input_path, row_limit)
    connection.execute(f"DROP TABLE IF EXISTS {table_name}")
    connection.execute(
        f"""
        CREATE TABLE {table_name} AS
        WITH base AS (
            SELECT
                trim(entity_id) AS entity_id,
                lower(trim(coalesce(country, ''))) AS country_norm,
                trim(regexp_replace(
                    regexp_replace(lower(coalesce(business_name, '')), '&', ' and ', 'g'),
                    '[^[:alnum:]]+', ' ', 'g'
                )) AS name_norm,
                trim(regexp_replace(
                    regexp_replace(lower(coalesce(business_address, '')), '&', ' and ', 'g'),
                    '[^[:alnum:]]+', ' ', 'g'
                )) AS address_norm
            FROM {relation}
        ), cleaned AS (
            SELECT
                *,
                trim(regexp_replace(name_norm, '{LEGAL_SUFFIX_PATTERN}', ' ', 'g'))
                    AS name_without_legal
            FROM base
        )
        SELECT
            entity_id,
            country_norm,
            name_norm,
            regexp_replace(name_norm, ' ', '', 'g') AS name_compact,
            trim(regexp_replace(name_without_legal, ' +', ' ', 'g')) AS name_core,
            address_norm,
            regexp_replace(address_norm, ' ', '', 'g') AS address_compact,
            regexp_extract(name_norm, '^([^ ]+)', 1) AS name_first,
            regexp_extract(address_norm, '(^| )([0-9]{{5,6}})( |$)', 2) AS postal_code,
            regexp_extract(address_norm, '^([0-9]+)', 1) AS leading_address_number
        FROM cleaned
        """
    )
    count = connection.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    print(f"  created {table_name}: {count:,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create normalized DuckDB tables from the provided TSV files."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--temp-dir", default="work/duckdb_tmp", type=Path)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument(
        "--row-limit",
        default=0,
        type=int,
        help="Only for a quick smoke test; 0 means process every input row.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    database = args.database.resolve()
    temp_dir = args.temp_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory does not exist: {data_dir}")
    database.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(database))
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET threads = {args.threads}")
    connection.execute(f"SET temp_directory = {quoted_path(temp_dir)}")

    print(f"Database: {database}")
    print(f"Memory limit: {args.memory_limit}; worker threads: {args.threads}")
    if args.row_limit:
        print(f"SMOKE TEST ONLY: limiting every input file to {args.row_limit:,} rows")

    for split in ("train", "test"):
        for source in (1, 2, 3):
            filename = f"{split}_source{source}.tsv"
            input_path = data_dir / split / filename
            if not input_path.is_file():
                print(f"  skipped {split}_s{source}: file not found")
                continue
            create_clean_table(
                connection,
                table_name=f"{split}_s{source}",
                input_path=input_path,
                row_limit=args.row_limit,
            )

    connection.close()
    print("Done. Next step is candidate-block generation.")


if __name__ == "__main__":
    main()
