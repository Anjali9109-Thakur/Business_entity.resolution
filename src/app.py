"""Local Streamlit dashboard for inspecting entity-resolution predictions.

Run from the repository root:
    streamlit run src/app.py

It queries existing local DuckDB/Parquet artifacts only; it never calls an
external business database or API and never loads the full TSV outputs into RAM.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "work" / "entity_resolution.duckdb"
SCORES_PATH = ROOT / "work" / "test_scores.parquet"
THRESHOLD = 0.910

st.set_page_config(page_title="Techmanthan · Entity Resolution", page_icon="🔗", layout="wide")


def sql_path(path: Path) -> str:
    """Return a DuckDB SQL literal for a Windows-safe path."""
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def artifact_status() -> tuple[bool, list[str]]:
    required = {
        "DuckDB model data": DATABASE_PATH,
        "Model scores": SCORES_PATH,
        "Final matches": ROOT / "output" / "matching_results.tsv",
        "Final candidates": ROOT / "output" / "candidate_pairs.tsv",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    return not missing, missing


def connect() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(DATABASE_PATH), read_only=True)
    connection.execute("SET memory_limit = '1GB'")
    connection.execute("SET threads = 2")
    return connection


@st.cache_data(show_spinner=False)
def sample_source1(country: str) -> pd.DataFrame:
    """Fetch a small deterministic quick-selection list."""
    connection = connect()
    try:
        country_filter = "" if country == "All countries" else "WHERE country_norm = ?"
        parameters = [] if country == "All countries" else [country.lower()]
        return connection.execute(
            f"""
            SELECT entity_id, name_norm AS business_name,
                   address_norm AS business_address, country_norm AS country
            FROM test_s1
            {country_filter}
            ORDER BY hash(entity_id)
            LIMIT 100
            """,
            parameters,
        ).fetchdf()
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def source_record(source1_id: str) -> pd.DataFrame:
    connection = connect()
    try:
        return connection.execute(
            """
            SELECT entity_id, name_norm AS business_name,
                   address_norm AS business_address, country_norm AS country
            FROM test_s1
            WHERE entity_id = ?
            """,
            [source1_id],
        ).fetchdf()
    finally:
        connection.close()


@st.cache_data(show_spinner="Loading scored candidates…")
def candidate_details(source1_id: str) -> pd.DataFrame:
    """Retrieve one Source-1 entity's candidates and model evidence."""
    connection = connect()
    try:
        return connection.execute(
            f"""
            SELECT
                candidate.candidate_rank AS rank,
                candidate.candidate_entity_id AS candidate_id,
                target.candidate_source AS source,
                score.match_probability AS probability,
                score.match_probability >= {THRESHOLD} AS predicted_match,
                candidate.blocking_score,
                candidate.strongest_rule_weight,
                round(jaro_winkler_similarity(source.name_norm, target.name_norm), 4)
                    AS name_similarity,
                round(jaro_winkler_similarity(source.address_norm, target.address_norm), 4)
                    AS address_similarity,
                target.name_norm AS candidate_name,
                target.address_norm AS candidate_address,
                target.country_norm AS candidate_country
            FROM test_candidates_fuzzy AS candidate
            INNER JOIN test_s1 AS source
                ON candidate.source1_entity_id = source.entity_id
            INNER JOIN test_targets AS target
                ON candidate.candidate_entity_id = target.entity_id
            INNER JOIN read_parquet({sql_path(SCORES_PATH)}) AS score
                USING (source1_entity_id, candidate_entity_id)
            WHERE candidate.source1_entity_id = ?
            ORDER BY candidate.candidate_rank
            """,
            [source1_id],
        ).fetchdf()
    finally:
        connection.close()


def probability_columns() -> dict[str, st.column_config.Column]:
    return {
        "probability": st.column_config.ProgressColumn(
            "Match probability", min_value=0.0, max_value=1.0, format="%.3f"
        ),
        "name_similarity": st.column_config.NumberColumn(format="%.4f"),
        "address_similarity": st.column_config.NumberColumn(format="%.4f"),
    }


def main() -> None:
    ready, missing = artifact_status()
    st.title("🔗 Techmanthan · Business Entity Resolution")
    st.caption(
        "Inspect local test predictions from the blocking + classifier pipeline. "
        "No external data or API is used."
    )
    if not ready:
        st.error("Required local artifacts are missing.")
        st.code("\n".join(missing))
        st.stop()

    with st.sidebar:
        st.header("Lookup")
        country = st.selectbox("Quick sample country", ("All countries", "India", "US", "France"))
        examples = sample_source1(country)
        choices = {
            f"{row.entity_id} — {row.business_name[:48]}": row.entity_id
            for row in examples.itertuples(index=False)
        }
        selected_label = st.selectbox("Choose a sample Source-1 record", ("", *choices))
        typed_id = st.text_input("Or enter an exact Source-1 ID", placeholder="S1-714132312")
        st.divider()
        st.metric("Decision threshold", f"{THRESHOLD:.3f}")
        st.caption("Only candidates with probability ≥ 0.910 are final matches.")

    source1_id = typed_id.strip() or choices.get(selected_label, "")
    if not source1_id:
        st.info("Choose a sample or enter a Source-1 entity ID to inspect its candidates.")
        return
    if not source1_id.startswith("S1-"):
        st.warning("Enter an ID that starts with `S1-`.")
        return

    source = source_record(source1_id)
    if source.empty:
        st.error(f"`{source1_id}` does not exist in test_source1.")
        return
    record = source.iloc[0]
    st.subheader(f"Source-1 record: {record['entity_id']}")
    left, middle, right = st.columns((1.1, 2.5, 2.5))
    left.metric("Country", str(record["country"]).upper())
    middle.markdown(f"**Business name (normalized)**  \n{record['business_name'] or '—'}")
    right.markdown(f"**Address (normalized)**  \n{record['business_address'] or '—'}")

    candidates = candidate_details(source1_id)
    match_count = int(candidates["predicted_match"].sum()) if not candidates.empty else 0
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Candidates scored", f"{len(candidates):,}")
    metric_2.metric("Final matches", f"{match_count:,}")
    metric_3.metric(
        "Highest probability",
        f"{candidates['probability'].max():.3f}" if not candidates.empty else "—",
    )
    if candidates.empty:
        st.warning("Blocking produced no candidate for this Source-1 record; the final prediction is empty.")
        return

    matches = candidates[candidates["predicted_match"]].copy()
    if matches.empty:
        st.warning("Candidates were found, but none crossed the high-precision 0.910 threshold.")
    else:
        st.success(f"{match_count} high-confidence final match(es) selected.")
        st.dataframe(matches, hide_index=True, use_container_width=True, column_config=probability_columns())

    with st.expander("All candidates scored by the model", expanded=False):
        st.dataframe(candidates, hide_index=True, use_container_width=True, column_config=probability_columns())


if __name__ == "__main__":
    main()
