
import json

from sqlalchemy import text

from retrieval.db import engine
from retrieval.dataset import OUTPUT_PATH

BATCH_SIZE = 1000


def import_documents(connection):
    """Import unique legal holdings into PostgreSQL."""
    batch = []
    processed = 0

    with OUTPUT_PATH.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            batch.append({
                "doc_id": record["doc_id"],
                "text": record["text"],
            })

            if len(batch) >= BATCH_SIZE:
                connection.execute(
                    text("""
                        INSERT INTO documents (doc_id, text)
                        VALUES (:doc_id, :text)
                        ON CONFLICT (doc_id) DO NOTHING
                    """),
                    batch,
                )
                processed += len(batch)
                batch.clear()

    if batch:
        connection.execute(
            text("""
                INSERT INTO documents (doc_id, text)
                VALUES (:doc_id, :text)
                ON CONFLICT (doc_id) DO NOTHING
            """),
            batch,
        )
        processed += len(batch)

    return processed


def import_sources(connection):
    """Import the original CaseHOLD source records."""
    batch = []
    processed = 0

    with OUTPUT_PATH.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            for source in record["sources"]:
                batch.append({
                    "doc_id": record["doc_id"],
                    "example_id": source["example_id"],
                    "candidate_index": source["candidate_index"],
                    "is_correct": source["is_correct"],
                    "citing_prompt": source["citing_prompt"],
                })

                if len(batch) >= BATCH_SIZE:
                    connection.execute(
                        text("""
                            INSERT INTO document_sources (
                                doc_id, example_id, candidate_index,
                                is_correct, citing_prompt
                            )
                            VALUES (
                                :doc_id, :example_id, :candidate_index,
                                :is_correct, :citing_prompt
                            )
                            ON CONFLICT (
                                doc_id, example_id, candidate_index
                            ) DO NOTHING
                        """),
                        batch,
                    )
                    processed += len(batch)
                    batch.clear()

    if batch:
        connection.execute(
            text("""
                INSERT INTO document_sources (
                    doc_id, example_id, candidate_index,
                    is_correct, citing_prompt
                )
                VALUES (
                    :doc_id, :example_id, :candidate_index,
                    :is_correct, :citing_prompt
                )
                ON CONFLICT (
                    doc_id, example_id, candidate_index
                ) DO NOTHING
            """),
            batch,
        )
        processed += len(batch)

    return processed


def main():
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {OUTPUT_PATH}")

    # Both imports run in one transaction.
    with engine.begin() as connection:
        print("Importing holdings...")
        holdings = import_documents(connection)
        print(f"Holdings processed: {holdings:,}")

        print("Importing source records...")
        sources = import_sources(connection)
        print(f"Source records processed: {sources:,}")

        counts = connection.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM documents),
                (SELECT COUNT(*) FROM document_sources)
        """)).one()

    print(f"Unique holdings in database: {counts[0]:,}")
    print(f"Source records in database: {counts[1]:,}")
    print("Import completed.")


if __name__ == "__main__":
    main()
