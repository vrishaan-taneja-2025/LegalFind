"""Prepare CaseHOLD data and inverted indexes without dropping existing data."""

from sqlalchemy import text

from retrieval import dataset as dataset_export
from retrieval.db import DATABASE_BACKEND, engine
from retrieval.download_dataset import download_dataset
from retrieval.rebuild_all_indexes import main as build_all_indexes
from retrieval.rebuild_database import load_source_data


BASE_TABLES = {"examples", "holdings", "candidates"}
INDEX_TABLES = {
    "terms",
    "term_stats",
    "indexed_text_stats",
    "postings",
    "posting_lists",
    "citation_terms",
    "citation_posting_lists",
}
BATCH_SIZE = 5000


def _insert_batches(connection, statement, rows):
    for offset in range(0, len(rows), BATCH_SIZE):
        connection.execute(text(statement), rows[offset:offset + BATCH_SIZE])


def _create_casehold_tables(connection):
    connection.execute(text("""
        CREATE TABLE examples (
            example_id BIGINT PRIMARY KEY,
            citing_prompt TEXT NOT NULL,
            label SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 4)
        )
    """))
    connection.execute(text("""
        CREATE TABLE holdings (
            holding_id TEXT PRIMARY KEY,
            holding_text TEXT NOT NULL
        )
    """))
    connection.execute(text("""
        CREATE TABLE candidates (
            example_id BIGINT NOT NULL REFERENCES examples(example_id),
            candidate_index SMALLINT NOT NULL
                CHECK (candidate_index BETWEEN 0 AND 4),
            holding_id TEXT NOT NULL REFERENCES holdings(holding_id),
            PRIMARY KEY (example_id, candidate_index)
        )
    """))


def _validate_casehold_tables(connection, expected):
    for table, expected_count in expected.items():
        actual = connection.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar_one()
        if actual != expected_count:
            raise RuntimeError(
                f"Existing {table} table has {actual:,} rows; "
                f"the downloaded dataset requires {expected_count:,}. "
                "No existing rows were changed. Resolve this database "
                "mismatch before continuing."
            )

    invalid = connection.execute(text("""
        SELECT COUNT(*)
        FROM (
            SELECT e.example_id
            FROM examples e
            LEFT JOIN candidates c ON c.example_id = e.example_id
            GROUP BY e.example_id, e.label
            HAVING COUNT(c.candidate_index) <> 5
                OR COUNT(*) FILTER (
                    WHERE c.candidate_index = e.label
                ) <> 1
        ) AS invalid_examples
    """)).scalar_one()
    if invalid:
        raise RuntimeError(
            f"Database integrity check found {invalid:,} invalid examples."
        )


def ensure_casehold_data():
    download_dataset()
    examples, holdings, candidates = load_source_data()
    expected = {
        "examples": len(examples),
        "holdings": len(holdings),
        "candidates": len(candidates),
    }

    with engine.begin() as connection:
        existing = set(connection.execute(text("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
        """)).scalars())
        present = existing & BASE_TABLES

        if present and present != BASE_TABLES:
            raise RuntimeError(
                "The database has only some CaseHOLD base tables. "
                "No tables were changed. Resolve the partial schema before setup."
            )

        if not present:
            _create_casehold_tables(connection)
            _insert_batches(connection, """
                INSERT INTO examples (example_id, citing_prompt, label)
                VALUES (:example_id, :citing_prompt, :label)
            """, examples)
            _insert_batches(connection, """
                INSERT INTO holdings (holding_id, holding_text)
                VALUES (:holding_id, :holding_text)
            """, holdings)
            _insert_batches(connection, """
                INSERT INTO candidates (example_id, candidate_index, holding_id)
                VALUES (:example_id, :candidate_index, :holding_id)
            """, candidates)
            print("CaseHOLD tables populated from the parquet dataset.")
        else:
            counts = {
                table: connection.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                ).scalar_one()
                for table in BASE_TABLES
            }
            if any(counts.values()):
                _validate_casehold_tables(connection, expected)
                print("Existing CaseHOLD tables match the dataset; reusing them.")
            else:
                _insert_batches(connection, """
                    INSERT INTO examples (example_id, citing_prompt, label)
                    VALUES (:example_id, :citing_prompt, :label)
                """, examples)
                _insert_batches(connection, """
                    INSERT INTO holdings (holding_id, holding_text)
                    VALUES (:holding_id, :holding_text)
                """, holdings)
                _insert_batches(connection, """
                    INSERT INTO candidates (example_id, candidate_index, holding_id)
                    VALUES (:example_id, :candidate_index, :holding_id)
                """, candidates)
                print("Empty CaseHOLD tables populated from the parquet dataset.")

        _validate_casehold_tables(connection, expected)

    return expected


def _indexes_are_ready(expected):
    with engine.connect() as connection:
        tables = set(connection.execute(text("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
        """)).scalars())
        if not INDEX_TABLES.issubset(tables):
            return False

        base_counts = connection.execute(text("""
            SELECT text_type, COUNT(*)
            FROM indexed_text_stats
            GROUP BY text_type
        """)).all()
        if dict(base_counts) != {
            "holding": expected["holdings"],
            "prompt": expected["examples"],
        }:
            return False

        posting_counts = dict(connection.execute(text("""
            SELECT text_type, COUNT(*)
            FROM postings
            GROUP BY text_type
        """)).all())
        df_counts = dict(connection.execute(text("""
            SELECT text_type, SUM(document_frequency)
            FROM term_stats
            GROUP BY text_type
        """)).all())
        if posting_counts != df_counts:
            return False

        holding_list_count, holding_list_postings = connection.execute(text("""
            SELECT COUNT(*), COALESCE(SUM(JSONB_ARRAY_LENGTH(posting_list)), 0)
            FROM posting_lists
        """)).one()
        expected_holding_terms, expected_holding_postings = connection.execute(text("""
            SELECT COUNT(DISTINCT p.term_id), COUNT(*)
            FROM postings p
            WHERE p.text_type = 'holding'
        """)).one()
        if (holding_list_count, holding_list_postings) != (
            expected_holding_terms, expected_holding_postings
        ):
            return False

        citation_term_count, citation_postings = connection.execute(text("""
            SELECT COUNT(*),
                   COALESCE(SUM(JSONB_ARRAY_LENGTH(posting_list)), 0)
            FROM citation_terms t
            JOIN citation_posting_lists p USING (term_id)
        """)).one()
        citation_expected = connection.execute(text("""
            SELECT COUNT(*) FROM postings WHERE text_type = 'prompt'
        """)).scalar_one()
        citation_rows = connection.execute(text("""
            SELECT COUNT(*) FROM citation_terms
            WHERE corpus_size = :corpus_size
        """), {"corpus_size": expected["examples"]}).scalar_one()
        return (
            citation_term_count > 0
            and citation_rows == citation_term_count
            and citation_postings == citation_expected
        )


def main():
    print(f"Preparing CaseHOLD data for {DATABASE_BACKEND}...", flush=True)
    download_dataset()
    print("Generating the holdings JSONL export...", flush=True)
    dataset_export.main()
    if DATABASE_BACKEND == "sqlite":
        from retrieval.sqlite_backend import initialize_instance

        print("Preparing SQLite base tables and search indexes...", flush=True)
        initialize_instance()
        print("SQLite database and all search indexes are ready.")
        return

    expected = ensure_casehold_data()
    print(
        f"CaseHOLD ready: {expected['examples']:,} examples, "
        f"{expected['holdings']:,} unique holdings."
    )

    if _indexes_are_ready(expected):
        print("Existing inverted indexes are complete; reusing them.")
    else:
        print("Building missing or incomplete inverted indexes...")
        build_all_indexes()

    if not _indexes_are_ready(expected):
        raise RuntimeError("Inverted-index validation failed after indexing.")
    print("Database and all search indexes are ready.")


if __name__ == "__main__":
    main()
