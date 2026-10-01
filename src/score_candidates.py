"""Score candidate pairs in bounded batches and store probabilities as Parquet.

The candidate table remains in DuckDB.  This avoids materialising millions of
pairs in pandas/RAM while preserving the exact candidate set seen by the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_training_features import feature_select_sql


def valid_table_name(name: str) -> bool:
    return name.replace("_", "").isalnum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--source-split", choices=("train", "test"), required=True)
    parser.add_argument("--target-split", choices=("train", "test"), required=True)
    parser.add_argument("--model-path", default="models/entity_matcher.joblib", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=100_000, type=int)
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()

    if not valid_table_name(args.candidate_table):
        raise SystemExit("--candidate-table may contain only letters, digits, and underscores.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive.")
    if not args.model_path.is_file():
        raise SystemExit(f"Model not found: {args.model_path}")

    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    feature_names: list[str] = artifact["feature_names"]
    connection = duckdb.connect(str(args.database.resolve()), read_only=True)
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET threads = {args.threads}")
    query = f"""
        SELECT {feature_select_sql()}
        FROM {args.candidate_table} AS candidate
        INNER JOIN {args.source_split}_s1 AS source
            ON candidate.source1_entity_id = source.entity_id
        INNER JOIN {args.target_split}_targets AS target
            ON candidate.candidate_entity_id = target.entity_id
    """
    reader = connection.execute(query).to_arrow_reader(batch_size=args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("source1_entity_id", pa.string()),
            pa.field("candidate_entity_id", pa.string()),
            pa.field("match_probability", pa.float32()),
        ]
    )
    writer = pq.ParquetWriter(args.output, schema=schema, compression="zstd")
    total = 0
    try:
        for batch in reader:
            columns = {name: batch.column(index) for index, name in enumerate(batch.schema.names)}
            matrix = np.column_stack(
                [np.asarray(columns[name], dtype=np.float32) for name in feature_names]
            )
            probabilities = model.predict_proba(matrix)[:, 1].astype(np.float32)
            output_batch = pa.RecordBatch.from_arrays(
                [
                    columns["source1_entity_id"],
                    columns["candidate_entity_id"],
                    pa.array(probabilities, type=pa.float32()),
                ],
                schema=schema,
            )
            writer.write_batch(output_batch)
            total += batch.num_rows
            if total % 1_000_000 < args.batch_size:
                print(f"  scored {total:,} candidate pairs")
    finally:
        writer.close()
        connection.close()
    print(f"Wrote {total:,} scored pairs to {args.output}")


if __name__ == "__main__":
    main()
