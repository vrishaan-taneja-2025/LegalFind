"""Build dedicated persisted inverted-index tables for citing prompts.

This adds/replaces only ``citation_terms`` and ``citation_posting_lists``;
the existing holdings index tables are left untouched.
"""

from sqlalchemy import text

from retrieval.db import engine
from retrieval.preprocessing import tokenize


TERM_BATCH_SIZE = 500
EXAMPLE_BATCH_SIZE = 500
TOKEN_INSERT_BATCH_SIZE = 25000


def build_citation_index():
    with engine.begin() as connection:
        example_count = connection.execute(
            text("SELECT COUNT(*) FROM examples")
        ).scalar_one()
        if not example_count:
            raise RuntimeError("No examples are available to index.")

        connection.execute(text("""
            CREATE TEMP TABLE citation_tokens (
                doc_id TEXT NOT NULL,
                term TEXT NOT NULL
            ) ON COMMIT DROP
        """))
        connection.execute(text("""
            CREATE TEMP TABLE citation_doc_stats (
                doc_id TEXT PRIMARY KEY,
                document_length INTEGER NOT NULL
            ) ON COMMIT DROP
        """))

        with engine.connect() as reader:
            examples = reader.execution_options(
                stream_results=True,
                yield_per=EXAMPLE_BATCH_SIZE,
            ).execute(text("""
                SELECT example_id, citing_prompt
                FROM examples
                ORDER BY example_id
            """))

            while batch := examples.fetchmany(EXAMPLE_BATCH_SIZE):
                token_rows = []
                stats_rows = []
                for example_id, citing_prompt in batch:
                    doc_id = str(example_id)
                    terms = tokenize(citing_prompt)
                    stats_rows.append({
                        "doc_id": doc_id,
                        "document_length": len(terms),
                    })
                    token_rows.extend(
                        {"doc_id": doc_id, "term": term}
                        for term in terms
                    )

                connection.execute(text("""
                    INSERT INTO citation_doc_stats (doc_id, document_length)
                    VALUES (:doc_id, :document_length)
                """), stats_rows)
                for start in range(0, len(token_rows), TOKEN_INSERT_BATCH_SIZE):
                    connection.execute(text("""
                        INSERT INTO citation_tokens (doc_id, term)
                        VALUES (:doc_id, :term)
                    """), token_rows[start:start + TOKEN_INSERT_BATCH_SIZE])

        connection.execute(text("""
            CREATE TEMP TABLE citation_doc_terms ON COMMIT DROP AS
            SELECT doc_id, term, COUNT(*)::INTEGER AS term_frequency
            FROM citation_tokens
            GROUP BY doc_id, term
        """))

        corpus = connection.execute(text("""
            SELECT AVG(document_length)::DOUBLE PRECISION AS average_length
            FROM citation_doc_stats
        """)).mappings().one()
        average_length = corpus["average_length"]
        if not average_length:
            raise RuntimeError("Citing prompts contain no searchable terms.")

        connection.execute(text("DROP TABLE IF EXISTS citation_posting_lists_new"))
        connection.execute(text("DROP TABLE IF EXISTS citation_terms_new"))
        connection.execute(text("""
            CREATE TABLE citation_terms_new (
                term_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                term TEXT NOT NULL UNIQUE,
                document_frequency BIGINT NOT NULL,
                idf DOUBLE PRECISION NOT NULL,
                corpus_size BIGINT NOT NULL,
                average_document_length DOUBLE PRECISION NOT NULL
            )
        """))

        connection.execute(text("""
            INSERT INTO citation_terms_new (
                term,
                document_frequency,
                idf,
                corpus_size,
                average_document_length
            )
            SELECT
                d.term,
                d.document_frequency,
                LN((CAST(:corpus_size AS DOUBLE PRECISION) + 1.0)
                    / (d.document_frequency::DOUBLE PRECISION + 1.0)) + 1.0,
                :corpus_size,
                :average_length
            FROM (
                SELECT term, COUNT(*)::BIGINT AS document_frequency
                FROM citation_doc_terms
                GROUP BY term
            ) d
        """), {
            "corpus_size": example_count,
            "average_length": average_length,
        })

        connection.execute(text("""
            CREATE TEMP TABLE citation_doc_norms ON COMMIT DROP AS
            SELECT
                d.doc_id,
                SQRT(SUM(POWER((1.0 + LN(d.term_frequency)) * t.idf, 2)))
                    AS tfidf_norm
            FROM citation_doc_terms d
            JOIN citation_terms_new t ON t.term = d.term
            GROUP BY d.doc_id
        """))

        connection.execute(text("""
            CREATE TABLE citation_posting_lists_new (
                term_id BIGINT PRIMARY KEY
                    REFERENCES citation_terms_new(term_id),
                posting_list JSONB NOT NULL
            )
        """))

        term_ids = connection.execute(text(
            "SELECT term_id FROM citation_terms_new ORDER BY term_id"
        )).scalars().all()

        for start in range(0, len(term_ids), TERM_BATCH_SIZE):
            batch = term_ids[start:start + TERM_BATCH_SIZE]
            connection.execute(text("""
                INSERT INTO citation_posting_lists_new (term_id, posting_list)
                SELECT
                    t.term_id,
                    JSONB_AGG(
                        JSONB_BUILD_OBJECT(
                            'doc_id', d.doc_id,
                            'tf', d.term_frequency,
                            'doc_len', s.document_length,
                            'tfidf_norm', n.tfidf_norm
                        ) ORDER BY d.doc_id
                    )
                FROM citation_terms_new t
                JOIN citation_doc_terms d ON d.term = t.term
                JOIN citation_doc_stats s ON s.doc_id = d.doc_id
                JOIN citation_doc_norms n ON n.doc_id = d.doc_id
                WHERE t.term_id = ANY(CAST(:term_ids AS BIGINT[]))
                GROUP BY t.term_id
            """), {"term_ids": batch})

        term_count = connection.execute(text(
            "SELECT COUNT(*) FROM citation_terms_new"
        )).scalar_one()
        list_count = connection.execute(text(
            "SELECT COUNT(*) FROM citation_posting_lists_new"
        )).scalar_one()
        posting_count = connection.execute(text("""
            SELECT COALESCE(SUM(JSONB_ARRAY_LENGTH(posting_list)), 0)
            FROM citation_posting_lists_new
        """)).scalar_one()
        expected_postings = connection.execute(text(
            "SELECT COUNT(*) FROM citation_doc_terms"
        )).scalar_one()

        if term_count != list_count or posting_count != expected_postings:
            raise RuntimeError(
                "Citation-index validation failed: term, list, and posting "
                "counts do not match."
            )

        connection.execute(text("DROP TABLE IF EXISTS citation_posting_lists"))
        connection.execute(text("DROP TABLE IF EXISTS citation_terms"))
        connection.execute(text(
            "ALTER TABLE citation_terms_new RENAME TO citation_terms"
        ))
        connection.execute(text(
            "ALTER TABLE citation_posting_lists_new "
            "RENAME TO citation_posting_lists"
        ))

    print(f"Citation records indexed: {example_count:,}")
    print(f"Unique citation terms: {term_count:,}")
    print(f"Citation postings: {posting_count:,}")


if __name__ == "__main__":
    build_citation_index()
