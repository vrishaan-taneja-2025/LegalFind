from sqlalchemy import text

from retrieval.db import engine


def create_schema():
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                text TEXT NOT NULL
            )
        """))

        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS document_sources (
                doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                example_id BIGINT NOT NULL,
                candidate_index SMALLINT NOT NULL
                    CHECK (candidate_index BETWEEN 0 AND 4),
                is_correct BOOLEAN NOT NULL,
                citing_prompt TEXT NOT NULL,
                PRIMARY KEY (doc_id, example_id, candidate_index)
            )
        """))

        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_sources_example
            ON document_sources (example_id)
        """))


if __name__ == "__main__":
    create_schema()
    print("Schema created: documents, document_sources.")
