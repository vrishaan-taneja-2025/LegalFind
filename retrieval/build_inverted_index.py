
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from sqlalchemy import text
from tqdm import tqdm

from retrieval.db import engine
from retrieval.preprocessing import tokenize


WORKERS = 4
TEXT_BATCH_SIZE = 250
INSERT_BATCH_SIZE = 5000


SOURCE_QUERY = """
    SELECT
        'holding'::text AS text_type,
        holding_id::text AS text_id,
        holding_text AS content
    FROM holdings

    UNION ALL

    SELECT
        'prompt'::text AS text_type,
        example_id::text AS text_id,
        citing_prompt AS content
    FROM examples
"""


def tokenize_document(item):
    """Tokenize one text and record each term's positions."""
    text_type, text_id, content = item

    tokens = tokenize(content)
    positions = defaultdict(list)

    for position, term in enumerate(tokens):
        positions[term].append(position)

    return {
        "text_type": text_type,
        "text_id": text_id,
        "document_length": len(tokens),
        "terms": dict(positions),
    }


def insert_batches(connection, statement, rows):
    """Insert rows in manageable batches."""
    for start in range(0, len(rows), INSERT_BATCH_SIZE):
        connection.execute(
            text(statement),
            rows[start:start + INSERT_BATCH_SIZE],
        )


def create_staging_tables():
    """Create temporary-build tables without touching the active index."""
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS _build_postings"))
        connection.execute(text("DROP TABLE IF EXISTS _build_term_stats"))
        connection.execute(
            text("DROP TABLE IF EXISTS _build_indexed_text_stats")
        )
        connection.execute(text("DROP TABLE IF EXISTS _build_terms"))

        connection.execute(text("""
            CREATE TABLE _build_terms (
                term_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                term TEXT NOT NULL UNIQUE
            )
        """))

        connection.execute(text("""
            CREATE TABLE _build_term_stats (
                term_id BIGINT NOT NULL
                    REFERENCES _build_terms(term_id),
                text_type TEXT NOT NULL
                    CHECK (text_type IN ('holding', 'prompt')),
                document_frequency BIGINT NOT NULL,
                PRIMARY KEY (term_id, text_type)
            )
        """))

        connection.execute(text("""
            CREATE TABLE _build_indexed_text_stats (
                text_type TEXT NOT NULL
                    CHECK (text_type IN ('holding', 'prompt')),
                text_id TEXT NOT NULL,
                document_length INTEGER NOT NULL,
                PRIMARY KEY (text_type, text_id)
            )
        """))

        connection.execute(text("""
            CREATE TABLE _build_postings (
                term_id BIGINT NOT NULL
                    REFERENCES _build_terms(term_id),
                text_type TEXT NOT NULL,
                text_id TEXT NOT NULL,
                term_frequency INTEGER NOT NULL,
                positions INTEGER[] NOT NULL,
                PRIMARY KEY (term_id, text_type, text_id),
                FOREIGN KEY (text_type, text_id)
                    REFERENCES _build_indexed_text_stats(
                        text_type, text_id
                    )
            )
        """))


def process_batch(connection, tokenized):
    """Store one batch of processed texts and update its statistics."""
    text_stats = []
    posting_data = []
    batch_df = Counter()
    batch_terms = set()

    for document in tokenized:
        text_type = document["text_type"]
        text_id = document["text_id"]
        terms = document["terms"]

        text_stats.append({
            "text_type": text_type,
            "text_id": text_id,
            "document_length": document["document_length"],
        })

        for term, positions in terms.items():
            batch_terms.add(term)
            batch_df[(text_type, term)] += 1

            posting_data.append({
                "term": term,
                "text_type": text_type,
                "text_id": text_id,
                "term_frequency": len(positions),
                "positions": positions,
            })

    # Parent text records must exist before their postings.
    insert_batches(
        connection,
        """
        INSERT INTO _build_indexed_text_stats
            (text_type, text_id, document_length)
        VALUES
            (:text_type, :text_id, :document_length)
        """,
        text_stats,
    )

    # Add new vocabulary terms. Existing terms are left unchanged.
    insert_batches(
        connection,
        """
        INSERT INTO _build_terms (term)
        VALUES (:term)
        ON CONFLICT (term) DO NOTHING
        """,
        [{"term": term} for term in sorted(batch_terms)],
    )

    # Resolve the database ID of each term in this batch.
    term_ids = dict(connection.execute(
        text("""
            SELECT term, term_id
            FROM _build_terms
            WHERE term = ANY(CAST(:terms AS text[]))
        """),
        {"terms": list(batch_terms)},
    ).fetchall())

    # Document frequency is tracked separately for holdings and prompts.
    term_stats = [
        {
            "term_id": term_ids[term],
            "text_type": text_type,
            "document_frequency": frequency,
        }
        for (text_type, term), frequency in batch_df.items()
    ]

    insert_batches(
        connection,
        """
        INSERT INTO _build_term_stats
            (term_id, text_type, document_frequency)
        VALUES
            (:term_id, :text_type, :document_frequency)
        ON CONFLICT (term_id, text_type)
        DO UPDATE SET document_frequency =
            _build_term_stats.document_frequency
            + EXCLUDED.document_frequency
        """,
        term_stats,
    )

    postings = [
        {
            "term_id": term_ids[item["term"]],
            "text_type": item["text_type"],
            "text_id": item["text_id"],
            "term_frequency": item["term_frequency"],
            "positions": item["positions"],
        }
        for item in posting_data
    ]

    insert_batches(
        connection,
        """
        INSERT INTO _build_postings
            (term_id, text_type, text_id, term_frequency, positions)
        VALUES
            (:term_id, :text_type, :text_id,
             :term_frequency, :positions)
        """,
        postings,
    )

    return len(postings)


def publish_index():
    """Replace the active index only after the build succeeds."""
    with engine.begin() as connection:
        # Drop dependent tables before their referenced tables.
        connection.execute(text("DROP TABLE IF EXISTS posting_lists"))
        connection.execute(text("DROP TABLE IF EXISTS postings"))
        connection.execute(text("DROP TABLE IF EXISTS term_stats"))
        connection.execute(
            text("DROP TABLE IF EXISTS indexed_text_stats")
        )
        connection.execute(text("DROP TABLE IF EXISTS terms"))

        connection.execute(
            text("ALTER TABLE _build_terms RENAME TO terms")
        )
        connection.execute(
            text("ALTER TABLE _build_term_stats RENAME TO term_stats")
        )
        connection.execute(text(
            "ALTER TABLE _build_indexed_text_stats "
            "RENAME TO indexed_text_stats"
        ))
        connection.execute(
            text("ALTER TABLE _build_postings RENAME TO postings")
        )

        connection.execute(text(
            "ALTER INDEX idx_postings_text_build "
            "RENAME TO idx_postings_text"
        ))


def build_index():
    with engine.connect() as connection:
        total_texts = connection.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM holdings)
                + (SELECT COUNT(*) FROM examples)
        """)).scalar_one()

    print(f"Texts to index: {total_texts}")
    print(f"Parallel workers: {WORKERS}")

    create_staging_tables()
    total_postings = 0

    try:
        # Create workers only for text processing, never database writes.
        with ProcessPoolExecutor(
            max_workers=WORKERS,
            mp_context=get_context("spawn"),
        ) as pool:
            with engine.connect() as source_connection:
                result = source_connection.execution_options(
                    stream_results=True
                ).execute(text(SOURCE_QUERY))

                with tqdm(
                    total=total_texts,
                    desc="Building index",
                    unit="text",
                ) as progress:
                    while True:
                        batch = result.fetchmany(TEXT_BATCH_SIZE)

                        if not batch:
                            break

                        inputs = [
                            (row.text_type, row.text_id, row.content)
                            for row in batch
                        ]

                        tokenized = list(pool.map(
                            tokenize_document,
                            inputs,
                            chunksize=8,
                        ))

                        # Each batch is committed independently.
                        with engine.begin() as connection:
                            total_postings += process_batch(
                                connection,
                                tokenized,
                            )

                        progress.update(len(batch))
                        progress.set_postfix(
                            postings=f"{total_postings:,}",
                            refresh=False,
                        )

        # Build the secondary lookup index after bulk insertion.
        with engine.begin() as connection:
            connection.execute(text("""
                CREATE INDEX idx_postings_text_build
                ON _build_postings (text_type, text_id)
            """))

            counts = connection.execute(text("""
                SELECT
                    (SELECT COUNT(*)
                     FROM _build_indexed_text_stats),
                    (SELECT COUNT(*) FROM _build_terms),
                    (SELECT COUNT(*) FROM _build_postings),
                    (SELECT COALESCE(SUM(document_frequency), 0)
                     FROM _build_term_stats)
            """)).one()

            indexed_texts, unique_terms, postings_count, df_sum = counts

            if indexed_texts != total_texts:
                raise RuntimeError(
                    f"Expected {total_texts} texts, got {indexed_texts}"
                )

            if postings_count != df_sum:
                raise RuntimeError(
                    "Document-frequency consistency check failed"
                )

        publish_index()

        print("\nIndex built and published successfully.")
        print(f"Indexed texts: {indexed_texts:,}")
        print(f"Unique terms: {unique_terms:,}")
        print(f"Postings: {postings_count:,}")

    except BaseException:
        # Keep the old published index if a build fails.
        # The next run removes incomplete staging tables.
        raise


if __name__ == "__main__":
    build_index()
