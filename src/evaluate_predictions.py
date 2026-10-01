"""Choose a precision-oriented decision threshold using macro F_0.5."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Thresholds must be comma-separated numbers.") from error
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("Every threshold must be between 0 and 1.")
    return sorted(set(thresholds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--validation-fold", default=0, type=int)
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.93,0.95,0.97,0.99"),
    )
    args = parser.parse_args()
    if not args.scores.is_file():
        raise SystemExit(f"Scores file not found: {args.scores}")
    if not 0 <= args.validation_fold <= 4:
        raise SystemExit("--validation-fold must be from 0 through 4.")

    connection = duckdb.connect(str(args.database.resolve()), read_only=True)
    score_relation = "read_parquet(?)"
    results: list[tuple[float, float, int]] = []
    for threshold in args.thresholds:
        macro_f05, predicted_pairs = connection.execute(
            f"""
            WITH validation_s1 AS (
                SELECT entity_id AS source1_entity_id
                FROM train_s1
                WHERE hash(entity_id) % 5 = {args.validation_fold}
            ), truth_counts AS (
                SELECT truth.source1_entity_id, count(*) AS true_count
                FROM train_truth AS truth
                INNER JOIN validation_s1 AS s1 USING (source1_entity_id)
                GROUP BY truth.source1_entity_id
            ), predictions AS (
                SELECT source1_entity_id, candidate_entity_id
                FROM {score_relation}
                WHERE match_probability >= ?
            ), prediction_counts AS (
                SELECT source1_entity_id, count(*) AS predicted_count
                FROM predictions
                GROUP BY source1_entity_id
            ), true_positive_counts AS (
                SELECT prediction.source1_entity_id, count(*) AS true_positive_count
                FROM predictions AS prediction
                INNER JOIN train_truth AS truth
                    USING (source1_entity_id, candidate_entity_id)
                GROUP BY prediction.source1_entity_id
            )
            SELECT
                avg(
                    CASE
                        WHEN truth_counts.true_count IS NULL THEN
                            CASE WHEN prediction_counts.predicted_count IS NULL THEN 1.0 ELSE 0.0 END
                        ELSE 1.25 * coalesce(true_positive_counts.true_positive_count, 0)::DOUBLE /
                            (0.25 * truth_counts.true_count +
                             coalesce(prediction_counts.predicted_count, 0))
                    END
                ) AS macro_f05,
                coalesce(sum(prediction_counts.predicted_count), 0)::BIGINT AS predicted_pairs
            FROM validation_s1 AS s1
            LEFT JOIN truth_counts USING (source1_entity_id)
            LEFT JOIN prediction_counts USING (source1_entity_id)
            LEFT JOIN true_positive_counts USING (source1_entity_id)
            """,
            [str(args.scores.resolve()), threshold],
        ).fetchone()
        results.append((threshold, float(macro_f05), int(predicted_pairs)))
        print(f"threshold={threshold:.3f}  macro_F0.5={macro_f05:.6f}  predicted_pairs={predicted_pairs:,}")
    connection.close()
    best = max(results, key=lambda item: (item[1], item[0]))
    print(f"BEST threshold={best[0]:.3f}  macro_F0.5={best[1]:.6f}")


if __name__ == "__main__":
    main()
