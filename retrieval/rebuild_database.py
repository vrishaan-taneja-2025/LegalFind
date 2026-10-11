
from pathlib import Path

from datasets import load_dataset
from sqlalchemy import text

from retrieval.db import engine


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "casehold-train.parquet"
BATCH_SIZE = 5000


def load_source_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataset = load_dataset(
        "parquet",
        data_files={"train": str(DATA_PATH)},
    )["train"]

    examples = []
    holdings = {}
    candidates = []

    print(f"Preparing {len(dataset):,} source examples for the database...", flush=True)
    for row_number, example in enumerate(dataset, start=1):
        example_id = int(example["example_id"])
        prompt = example["citing_prompt"]
        label = int(example["label"])

        examples.append({
            "example_id": example_id,
            "citing_prompt": prompt,
            "label": label,
        })

        for candidate_index in range(5):
            original_text = example[f"holding_{candidate_index}"].strip()
            normalized_text = " ".join(original_text.split()).casefold()

            if not normalized_text:
                raise ValueError(
                    f"Empty holding in example {example_id}, "
                    f"candidate {candidate_index}"
                )

            if normalized_text not in holdings:
                holding_id = f"H{len(holdings) + 1:06d}"
                holdings[normalized_text] = {
                    "holding_id": holding_id,
                    "holding_text": original_text,
                }

            candidates.append({
                "example_id": example_id,
                "candidate_index": candidate_index,
                "holding_id": holdings[normalized_text]["holding_id"],
            })

        if row_number % 5000 == 0 or row_number == len(dataset):
            print(
                f"Database source progress: {row_number:,}/{len(dataset):,} "
                f"examples; {len(holdings):,} unique holdings.",
                flush=True,
            )

    return examples, list(holdings.values()), candidates


def insert_batches(connection, statement, rows):
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        connection.execute(text(statement), batch)


def rebuild_database():
    examples, holdings, candidates = load_source_data()

    print(f"Source examples: {len(examples)}")
    print(f"Unique holdings: {len(holdings)}")
    print(f"Candidate records: {len(candidates)}")

    # Everything below runs in one transaction.
    # A failure rolls back the table changes and inserts.
    with engine.begin() as connection:

        # Drop only tables created for this project.
        # Drop referencing tables before referenced tables.
        connection.execute(text("DROP TABLE IF EXISTS candidates"))
        connection.execute(text("DROP TABLE IF EXISTS document_sources"))
        connection.execute(text("DROP TABLE IF EXISTS examples"))
        connection.execute(text("DROP TABLE IF EXISTS holdings"))
        connection.execute(text("DROP TABLE IF EXISTS documents"))

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
                example_id BIGINT NOT NULL
                    REFERENCES examples(example_id),
                candidate_index SMALLINT NOT NULL
                    CHECK (candidate_index BETWEEN 0 AND 4),
                holding_id TEXT NOT NULL
                    REFERENCES holdings(holding_id),
                PRIMARY KEY (example_id, candidate_index)
            )
        """))

        insert_batches(
            connection,
            """
            INSERT INTO examples (example_id, citing_prompt, label)
            VALUES (:example_id, :citing_prompt, :label)
            """,
            examples,
        )

        insert_batches(
            connection,
            """
            INSERT INTO holdings (holding_id, holding_text)
            VALUES (:holding_id, :holding_text)
            """,
            holdings,
        )

        insert_batches(
            connection,
            """
            INSERT INTO candidates
                (example_id, candidate_index, holding_id)
            VALUES
                (:example_id, :candidate_index, :holding_id)
            """,
            candidates,
        )

        # Verify counts before committing.
        expected_counts = {
            "examples": len(examples),
            "holdings": len(holdings),
            "candidates": len(candidates),
        }

        for table, expected in expected_counts.items():
            actual = connection.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar_one()

            if actual != expected:
                raise RuntimeError(
                    f"{table}: expected {expected} rows, got {actual}"
                )

        invalid_examples = connection.execute(text("""
            SELECT COUNT(*)
            FROM (
                SELECT e.example_id
                FROM examples e
                LEFT JOIN candidates c
                    ON c.example_id = e.example_id
                GROUP BY e.example_id, e.label
                HAVING COUNT(c.candidate_index) <> 5
                    OR COUNT(*) FILTER (
                        WHERE c.candidate_index = e.label
                    ) <> 1
            ) AS invalid
        """)).scalar_one()

        if invalid_examples != 0:
            raise RuntimeError(
                f"{invalid_examples} examples failed integrity checks"
            )

    print("Database rebuilt and verified successfully.")
    print("Tables: examples, holdings, candidates")


if __name__ == "__main__":
    rebuild_database()
