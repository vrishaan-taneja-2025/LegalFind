
from sqlalchemy import text
from tqdm import tqdm

from retrieval.build_citation_index import build_citation_index
from retrieval.db import engine


BATCH_SIZE = 500


def rebuild_index():
    with engine.begin() as conn:
        print("Checking the existing inverted index...")

        required_tables = {
            "terms",
            "term_stats",
            "indexed_text_stats",
            "postings",
        }

        existing_tables = set(
            conn.execute(
                text("""
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                """)
            ).scalars().all()
        )

        missing = required_tables - existing_tables

        if missing:
            raise RuntimeError(
                f"Missing source index tables: {sorted(missing)}"
            )

        corpus = conn.execute(
            text("""
                SELECT
                    COUNT(*)::BIGINT AS document_count,
                    AVG(document_length)::DOUBLE PRECISION AS avg_length
                FROM indexed_text_stats
                WHERE text_type = 'holding'
            """)
        ).mappings().one()

        document_count = corpus["document_count"]
        average_length = corpus["avg_length"]

        if not document_count or average_length is None:
            raise RuntimeError("No indexed holdings were found.")

        print(f"Holdings: {document_count:,}")
        print(f"Average document length: {average_length:.2f} tokens")

        # Remove leftover staging tables from a previous failed attempt.
        conn.execute(text("DROP TABLE IF EXISTS posting_lists_new"))
        conn.execute(text("DROP TABLE IF EXISTS terms_new"))

        # TABLE 1: One row per unique term.
        conn.execute(text("""
            CREATE TABLE terms_new (
                term_id BIGINT PRIMARY KEY,
                term TEXT UNIQUE NOT NULL,
                document_frequency BIGINT NOT NULL,
                idf DOUBLE PRECISION NOT NULL,
                corpus_size BIGINT NOT NULL,
                average_document_length DOUBLE PRECISION NOT NULL
            )
        """))

        # TABLE 2: One JSONB posting list per term.
        conn.execute(text("""
            CREATE TABLE posting_lists_new (
                term_id BIGINT PRIMARY KEY
                    REFERENCES terms_new(term_id),
                posting_list JSONB NOT NULL
            )
        """))

        print("Building the term dictionary and IDF values...")

        # Reuse existing term IDs. Keep only terms found in holdings.
        conn.execute(text("""
            WITH corpus AS (
                SELECT
                    COUNT(*)::BIGINT AS n,
                    AVG(document_length)::DOUBLE PRECISION AS avg_len
                FROM indexed_text_stats
                WHERE text_type = 'holding'
            )
            INSERT INTO terms_new (
                term_id,
                term,
                document_frequency,
                idf,
                corpus_size,
                average_document_length
            )
            SELECT
                t.term_id,
                t.term,
                ts.document_frequency,
                LN(
                    (c.n::DOUBLE PRECISION + 1.0)
                    / (ts.document_frequency::DOUBLE PRECISION + 1.0)
                ) + 1.0,
                c.n,
                c.avg_len
            FROM terms t
            JOIN term_stats ts
                ON ts.term_id = t.term_id
               AND ts.text_type = 'holding'
            CROSS JOIN corpus c
        """))

        term_count = conn.execute(
            text("SELECT COUNT(*) FROM terms_new")
        ).scalar_one()

        if term_count == 0:
            raise RuntimeError("No terms were found for holdings.")

        # Calculate each holding's TF-IDF vector norm.
        # Uses exactly the same TF and IDF weighting as the index.
        print("Calculating TF-IDF vector norms...")

        conn.execute(text("""
            CREATE TEMP TABLE holding_norms ON COMMIT DROP AS
            WITH corpus AS (
                SELECT COUNT(*)::DOUBLE PRECISION AS n
                FROM indexed_text_stats
                WHERE text_type = 'holding'
            )
            SELECT
                p.text_id,
                SQRT(
                    SUM(
                        POWER(
                            (
                                1.0 + LN(
                                    p.term_frequency::DOUBLE PRECISION
                                )
                            )
                            *
                            (
                                LN(
                                    (c.n + 1.0)
                                    /
                                    (
                                        ts.document_frequency
                                        ::DOUBLE PRECISION + 1.0
                                    )
                                ) + 1.0
                            ),
                            2
                        )
                    )
                ) AS tfidf_norm
            FROM postings p
            JOIN term_stats ts
                ON ts.term_id = p.term_id
               AND ts.text_type = p.text_type
            CROSS JOIN corpus c
            WHERE p.text_type = 'holding'
            GROUP BY p.text_id
        """))

        # Fetch term IDs for progress reporting.
        term_ids = conn.execute(
            text("SELECT term_id FROM terms_new ORDER BY term_id")
        ).scalars().all()

        print(f"Building posting lists for {len(term_ids):,} terms...")

        # Aggregate postings in batches to limit intermediate memory use.
        for start in tqdm(
            range(0, len(term_ids), BATCH_SIZE),
            desc="Building posting lists",
            unit="batch",
        ):
            batch = term_ids[start:start + BATCH_SIZE]

            conn.execute(
                text("""
                    INSERT INTO posting_lists_new (
                        term_id,
                        posting_list
                    )
                    SELECT
                        p.term_id,
                        JSONB_AGG(
                            JSONB_BUILD_OBJECT(
                                'doc_id', p.text_id,
                                'tf', p.term_frequency,
                                'doc_len', s.document_length,
                                'tfidf_norm', n.tfidf_norm
                            )
                            ORDER BY p.text_id
                        )
                    FROM postings p
                    JOIN indexed_text_stats s
                        ON s.text_type = p.text_type
                       AND s.text_id = p.text_id
                    JOIN holding_norms n
                        ON n.text_id = p.text_id
                    WHERE p.text_type = 'holding'
                      AND p.term_id = ANY(CAST(:term_ids AS BIGINT[]))
                    GROUP BY p.term_id
                """),
                {"term_ids": batch},
            )

        # Validate the new index before replacing the old tables.
        new_term_count = conn.execute(
            text("SELECT COUNT(*) FROM terms_new")
        ).scalar_one()

        list_count = conn.execute(
            text("SELECT COUNT(*) FROM posting_lists_new")
        ).scalar_one()

        new_posting_count = conn.execute(text("""
            SELECT COALESCE(
                SUM(JSONB_ARRAY_LENGTH(posting_list)::BIGINT),
                0
            )
            FROM posting_lists_new
        """)).scalar_one()

        expected_postings = conn.execute(text("""
            SELECT COUNT(*)
            FROM postings
            WHERE text_type = 'holding'
        """)).scalar_one()

        print("\nValidation:")
        print(f"  Terms: {new_term_count:,}")
        print(f"  Posting-list rows: {list_count:,}")
        print(f"  Individual postings: {new_posting_count:,}")
        print(f"  Expected postings: {expected_postings:,}")

        if new_term_count != list_count:
            raise RuntimeError(
                "Validation failed: some terms have no posting-list row."
            )

        if new_posting_count != expected_postings:
            raise RuntimeError(
                "Validation failed: posting counts do not match."
            )

        # All changes are inside one transaction.
        # If anything fails, PostgreSQL rolls back the replacement.
        print("\nValidation passed. Replacing the old index tables...")

        conn.execute(text("DROP TABLE postings"))
        conn.execute(text("DROP TABLE term_stats"))
        conn.execute(text("DROP TABLE indexed_text_stats"))
        conn.execute(text("DROP TABLE terms"))

        conn.execute(text("ALTER TABLE terms_new RENAME TO terms"))
        conn.execute(
            text("ALTER TABLE posting_lists_new RENAME TO posting_lists")
        )

        print("\nTwo-table inverted index built successfully.")
        print(f"Terms: {new_term_count:,}")
        print(f"Posting lists: {list_count:,}")
        print(f"Individual postings: {new_posting_count:,}")

    print("\nBuilding the citation prompt index...")
    build_citation_index()


if __name__ == "__main__":
    rebuild_index()
