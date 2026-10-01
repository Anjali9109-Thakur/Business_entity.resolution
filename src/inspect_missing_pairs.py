"""Print examples of true validation links missed by a candidate table."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--validation-fold", default=0, type=int)
    parser.add_argument("--limit", default=20, type=int)
    args = parser.parse_args()
    if not args.candidate_table.replace("_", "").isalnum():
        raise SystemExit("--candidate-table may contain only letters, digits, and underscores.")

    connection = duckdb.connect(str(args.database.resolve()), read_only=True)
    rows = connection.execute(
        f"""
        WITH missing AS (
            SELECT truth.source1_entity_id, truth.candidate_entity_id
            FROM train_truth AS truth
            ANTI JOIN {args.candidate_table} AS candidate
                USING (source1_entity_id, candidate_entity_id)
            WHERE hash(truth.source1_entity_id) % 5 = {args.validation_fold}
            LIMIT {args.limit}
        )
        SELECT
            missing.source1_entity_id,
            missing.candidate_entity_id,
            source.country_norm,
            source.name_norm AS source_name,
            target.name_norm AS target_name,
            source.address_norm AS source_address,
            target.address_norm AS target_address,
            source.postal_code AS source_postal,
            target.postal_code AS target_postal
        FROM missing
        INNER JOIN train_s1 AS source
            ON missing.source1_entity_id = source.entity_id
        INNER JOIN train_targets AS target
            ON missing.candidate_entity_id = target.entity_id
        """
    ).fetchall()
    for number, row in enumerate(rows, start=1):
        print(f"\n--- missed pair {number} ---")
        labels = (
            "source1_entity_id",
            "candidate_entity_id",
            "country",
            "source_name",
            "target_name",
            "source_address",
            "target_address",
            "source_postal",
            "target_postal",
        )
        for label, value in zip(labels, row):
            print(f"{label}: {value}")
    print(f"\nPrinted {len(rows)} missed true pairs.")
    connection.close()


if __name__ == "__main__":
    main()
