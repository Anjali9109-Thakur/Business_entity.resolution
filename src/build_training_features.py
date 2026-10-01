"""Create a balanced numeric training sample from blocked candidate pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def valid_table_name(name: str) -> bool:
    return name.replace("_", "").isalnum()


def feature_select_sql() -> str:
    """SQL feature expressions shared by train and inference feature builders."""
    return """
        candidate.source1_entity_id,
        candidate.candidate_entity_id,
        candidate.blocking_score::DOUBLE AS blocking_score,
        candidate.strongest_rule_weight::DOUBLE AS strongest_rule_weight,
        CASE WHEN greatest(length(source.name_norm), length(target.name_norm)) > 0
             THEN jaro_winkler_similarity(source.name_norm, target.name_norm)
             ELSE 0.0 END AS name_jaro,
        CASE WHEN greatest(length(source.name_core), length(target.name_core)) > 0
             THEN jaro_winkler_similarity(source.name_core, target.name_core)
             ELSE 0.0 END AS name_core_jaro,
        CASE WHEN greatest(length(source.address_norm), length(target.address_norm)) > 0
             THEN jaro_winkler_similarity(source.address_norm, target.address_norm)
             ELSE 0.0 END AS address_jaro,
        CASE WHEN greatest(length(source.name_norm), length(target.name_norm)) > 0
             THEN 1.0 - levenshtein(source.name_norm, target.name_norm)::DOUBLE /
                  greatest(length(source.name_norm), length(target.name_norm))
             ELSE 0.0 END AS name_edit_similarity,
        CASE WHEN greatest(length(source.address_norm), length(target.address_norm)) > 0
             THEN 1.0 - levenshtein(source.address_norm, target.address_norm)::DOUBLE /
                  greatest(length(source.address_norm), length(target.address_norm))
             ELSE 0.0 END AS address_edit_similarity,
        list_count(
            list_intersect(
                list_distinct(string_split(source.name_norm, ' ')),
                list_distinct(string_split(target.name_norm, ' '))
            )
        )::DOUBLE /
            greatest(
                list_count(
                    list_distinct(
                        list_concat(
                            string_split(source.name_norm, ' '),
                            string_split(target.name_norm, ' ')
                        )
                    )
                ),
                1
            ) AS name_token_jaccard,
        list_count(
            list_intersect(
                list_distinct(string_split(source.address_norm, ' ')),
                list_distinct(string_split(target.address_norm, ' '))
            )
        )::DOUBLE /
            greatest(
                list_count(
                    list_distinct(
                        list_concat(
                            string_split(source.address_norm, ' '),
                            string_split(target.address_norm, ' ')
                        )
                    )
                ),
                1
            ) AS address_token_jaccard,
        list_count(
            list_intersect(
                regexp_extract_all(source.address_norm, '[0-9]+'),
                regexp_extract_all(target.address_norm, '[0-9]+')
            )
        )::INTEGER AS shared_address_number_count,
        least(length(source.name_norm), length(target.name_norm))::DOUBLE /
            greatest(greatest(length(source.name_norm), length(target.name_norm)), 1)
            AS name_length_ratio,
        least(length(source.address_norm), length(target.address_norm))::DOUBLE /
            greatest(greatest(length(source.address_norm), length(target.address_norm)), 1)
            AS address_length_ratio,
        (source.name_compact <> '' AND source.name_compact = target.name_compact)::INTEGER
            AS exact_name,
        (source.name_core <> '' AND source.name_core = target.name_core)::INTEGER
            AS exact_name_core,
        (source.address_compact <> '' AND source.address_compact = target.address_compact)::INTEGER
            AS exact_address,
        (source.name_first <> '' AND source.name_first = target.name_first)::INTEGER
            AS name_first_equal,
        (source.postal_code <> '' AND source.postal_code = target.postal_code)::INTEGER
            AS postal_equal,
        (source.leading_address_number <> '' AND
             source.leading_address_number = target.leading_address_number)::INTEGER
            AS leading_number_equal,
        starts_with(candidate.candidate_entity_id, 'S2-')::INTEGER AS candidate_is_s2
    """


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--output-table", default="model_training_features")
    parser.add_argument("--positive-modulus", default=4, type=int)
    parser.add_argument("--negative-modulus", default=16, type=int)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    if not all(valid_table_name(name) for name in (args.candidate_table, args.output_table)):
        raise SystemExit("Table names may contain only letters, digits, and underscores.")
    if args.positive_modulus < 1 or args.negative_modulus < 1:
        raise SystemExit("Sampling moduli must be positive integers.")

    connection = duckdb.connect(str(args.database.resolve()))
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET threads = {args.threads}")
    connection.execute(f"DROP TABLE IF EXISTS {args.output_table}")
    connection.execute(
        f"""
        CREATE TABLE {args.output_table} AS
        WITH labeled_candidates AS (
            SELECT
                candidate.*,
                CASE WHEN truth.candidate_entity_id IS NULL THEN 0 ELSE 1 END AS label
            FROM {args.candidate_table} AS candidate
            LEFT JOIN train_truth AS truth
                USING (source1_entity_id, candidate_entity_id)
        ), sampled_candidates AS (
            SELECT *
            FROM labeled_candidates
            WHERE (label = 1 AND
                   hash(source1_entity_id || candidate_entity_id) % {args.positive_modulus} = 0)
               OR (label = 0 AND
                   hash(source1_entity_id || candidate_entity_id) % {args.negative_modulus} = 0)
        )
        SELECT
            {feature_select_sql()},
            candidate.label
        FROM sampled_candidates AS candidate
        INNER JOIN train_s1 AS source
            ON candidate.source1_entity_id = source.entity_id
        INNER JOIN train_targets AS target
            ON candidate.candidate_entity_id = target.entity_id
        """
    )
    counts = connection.execute(
        f"SELECT label, count(*) FROM {args.output_table} GROUP BY label ORDER BY label"
    ).fetchall()
    print("Created " + args.output_table + ": " + ", ".join(
        f"label {label}={count:,}" for label, count in counts
    ))
    connection.close()


if __name__ == "__main__":
    main()
