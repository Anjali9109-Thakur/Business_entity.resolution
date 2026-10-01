"""Shared SQL fragments for scalable, deterministic candidate blocking."""

from __future__ import annotations


def block_key_lateral(alias: str) -> str:
    """Return SQL that emits six whole-field blocking keys for one record.

    The rule prefix is part of every key, so similar text produced by different
    rules remains distinguishable.  A longer/exacter key receives more weight.
    """
    return f"""
    CROSS JOIN LATERAL (
        SELECT * FROM (
            VALUES
                (7, CASE WHEN {alias}.country_norm <> ''
                               AND length({alias}.name_compact) >= 4
                    THEN concat('N|', {alias}.country_norm, '|', {alias}.name_compact)
                    END),
                (6, CASE WHEN {alias}.country_norm <> ''
                               AND length(replace({alias}.name_core, ' ', '')) >= 4
                    THEN concat('C|', {alias}.country_norm, '|',
                                replace({alias}.name_core, ' ', ''))
                    END),
                (5, CASE WHEN {alias}.country_norm <> ''
                               AND length({alias}.address_compact) >= 8
                    THEN concat('A|', {alias}.country_norm, '|', {alias}.address_compact)
                    END),
                (4, CASE WHEN {alias}.country_norm <> ''
                               AND {alias}.postal_code <> ''
                               AND length({alias}.name_first) >= 3
                    THEN concat('P1|', {alias}.country_norm, '|', {alias}.postal_code,
                                '|', {alias}.name_first)
                    END),
                (3, CASE WHEN {alias}.country_norm <> ''
                               AND {alias}.postal_code <> ''
                               AND length({alias}.name_compact) >= 8
                    THEN concat('P8|', {alias}.country_norm, '|', {alias}.postal_code,
                                '|', left({alias}.name_compact, 8))
                    END),
                (2, CASE WHEN {alias}.country_norm <> ''
                               AND {alias}.leading_address_number <> ''
                               AND length({alias}.name_compact) >= 8
                    THEN concat('H8|', {alias}.country_norm, '|',
                                {alias}.leading_address_number, '|',
                                left({alias}.name_compact, 8))
                    END)
        ) AS raw_keys(block_weight, block_key)
        WHERE block_key IS NOT NULL
    ) AS keys
    """


ADDRESS_STOP_TOKENS = (
    "and", "avenue", "block", "building", "city", "colony", "district", "drive",
    "east", "floor", "house", "india", "lane", "limited", "near", "north", "post",
    "private", "road", "south", "state", "street", "town", "township", "village",
    "west",
)


def emitted_block_keys_sql(relation: str, alias: str, output_id: str) -> str:
    """Return SQL emitting whole-field plus rare-name/address-token keys.

    Individual tokens create recall for reordered, partially missing, or
    misspelled records.  The caller later removes overly frequent keys through
    ``block_size``; that cap is essential to keep the candidate set compact.
    """
    address_stop_list = ", ".join(f"'{token}'" for token in ADDRESS_STOP_TOKENS)
    return f"""
        SELECT
            {alias}.entity_id AS {output_id},
            keys.block_weight,
            keys.block_key
        FROM {relation} AS {alias}
        {block_key_lateral(alias)}

        UNION ALL

        SELECT DISTINCT
            {alias}.entity_id AS {output_id},
            2 AS block_weight,
            concat('NT|', {alias}.country_norm, '|', name_token.token) AS block_key
        FROM {relation} AS {alias}
        CROSS JOIN UNNEST(string_split({alias}.name_norm, ' ')) AS name_token(token)
        WHERE {alias}.country_norm <> ''
              AND length(name_token.token) >= 5

        UNION ALL

        SELECT DISTINCT
            {alias}.entity_id AS {output_id},
            1 AS block_weight,
            concat('NF3|', {alias}.country_norm, '|', left(name_token.token, 3)) AS block_key
        FROM {relation} AS {alias}
        CROSS JOIN UNNEST(string_split({alias}.name_norm, ' ')) AS name_token(token)
        WHERE {alias}.country_norm <> ''
              AND length(name_token.token) >= 4

        UNION ALL

        SELECT DISTINCT
            {alias}.entity_id AS {output_id},
            1 AS block_weight,
            concat('NS3|', {alias}.country_norm, '|', right(name_token.token, 3)) AS block_key
        FROM {relation} AS {alias}
        CROSS JOIN UNNEST(string_split({alias}.name_norm, ' ')) AS name_token(token)
        WHERE {alias}.country_norm <> ''
              AND length(name_token.token) >= 4

        UNION ALL

        SELECT DISTINCT
            {alias}.entity_id AS {output_id},
            1 AS block_weight,
            concat('NG4|', {alias}.country_norm, '|',
                   substr({alias}.name_compact, ngram.position, 4)) AS block_key
        FROM {relation} AS {alias}
        CROSS JOIN LATERAL range(
            1, greatest(length({alias}.name_compact) - 2, 1)
        ) AS ngram(position)
        WHERE {alias}.country_norm <> ''
              AND length({alias}.name_compact) >= 6

        UNION ALL

        SELECT DISTINCT
            {alias}.entity_id AS {output_id},
            1 AS block_weight,
            concat('AT|', {alias}.country_norm, '|', address_token.token) AS block_key
        FROM {relation} AS {alias}
        CROSS JOIN UNNEST(string_split({alias}.address_norm, ' ')) AS address_token(token)
        WHERE {alias}.country_norm <> ''
              AND length(address_token.token) >= 5
              AND address_token.token NOT IN ({address_stop_list})

        UNION ALL

        SELECT DISTINCT
            {alias}.entity_id AS {output_id},
            1 AS block_weight,
            concat('AP4|', {alias}.country_norm, '|', left(address_token.token, 4))
                AS block_key
        FROM {relation} AS {alias}
        CROSS JOIN UNNEST(string_split({alias}.address_norm, ' ')) AS address_token(token)
        WHERE {alias}.country_norm <> ''
              AND length(address_token.token) >= 6
              AND address_token.token NOT IN ({address_stop_list})
    """


def emitted_address_number_keys_sql(relation: str, alias: str, output_id: str) -> str:
    """Return selective address-number keys in a separate, optional index.

    The baseline address-prefix rule loses recall when suppliers reorder an
    address (``"410 32nd St Purcellville"`` versus ``"Purcellville 410 32nd
    St"``). A country-scoped alphanumeric building/house token is invariant
    to that ordering. It intentionally lives in its own index because this
    recall rule can be evaluated independently from the baseline blocks.
    """
    return f"""
        SELECT DISTINCT
            {alias}.entity_id AS {output_id},
            3 AS block_weight,
            concat('AN|', {alias}.country_norm, '|', address_token.token) AS block_key
        FROM {relation} AS {alias}
        CROSS JOIN UNNEST(string_split({alias}.address_norm, ' ')) AS address_token(token)
        WHERE {alias}.country_norm <> ''
              AND length(address_token.token) >= 2
              AND regexp_matches(address_token.token, '[0-9]')
    """
