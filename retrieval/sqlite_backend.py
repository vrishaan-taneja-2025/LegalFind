"""SQLite storage and inverted-index fallback for LegalFind."""

import math
from collections import Counter

from sqlalchemy import text

from retrieval.db import engine
from retrieval.download_dataset import download_dataset
from retrieval.preprocessing import tokenize
from retrieval.rebuild_database import load_source_data


BATCH_SIZE = 5000


def _insert_batches(connection, statement, rows):
    for offset in range(0, len(rows), BATCH_SIZE):
        connection.execute(text(statement), rows[offset:offset + BATCH_SIZE])


def _create_base_schema(connection):
    connection.execute(text("""
        CREATE TABLE examples (
            example_id INTEGER PRIMARY KEY,
            citing_prompt TEXT NOT NULL,
            label INTEGER NOT NULL CHECK (label BETWEEN 0 AND 4)
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
            example_id INTEGER NOT NULL REFERENCES examples(example_id),
            candidate_index INTEGER NOT NULL
                CHECK (candidate_index BETWEEN 0 AND 4),
            holding_id TEXT NOT NULL REFERENCES holdings(holding_id),
            PRIMARY KEY (example_id, candidate_index)
        )
    """))


def _populate_base(connection, examples, holdings, candidates):
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


def _ensure_base_data():
    examples, holdings, candidates = load_source_data()
    expected = {
        "examples": len(examples),
        "holdings": len(holdings),
        "candidates": len(candidates),
    }

    with engine.begin() as connection:
        existing = {
            row[0] for row in connection.execute(text("""
                SELECT name FROM sqlite_master
                WHERE type = 'table'
            """))
        }
        required = set(expected)
        present = existing & required
        if present and present != required:
            raise RuntimeError(
                "SQLite contains only some CaseHOLD tables. No data was "
                "changed; resolve the partial database before setup."
            )

        if not present:
            _create_base_schema(connection)
            _populate_base(connection, examples, holdings, candidates)
            print("SQLite CaseHOLD tables populated from the parquet dataset.")
        else:
            counts = {
                table: connection.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                ).scalar_one()
                for table in required
            }
            if any(counts.values()):
                mismatches = {
                    table: (counts[table], expected[table])
                    for table in required
                    if counts[table] != expected[table]
                }
                if mismatches:
                    raise RuntimeError(
                        "SQLite CaseHOLD row counts do not match the dataset. "
                        "No existing data was overwritten. Remove or back up "
                        "data/casehold.sqlite3 to initialize a fresh copy."
                    )
                print("Existing SQLite CaseHOLD tables match; reusing them.")
            else:
                _populate_base(connection, examples, holdings, candidates)
                print("Empty SQLite CaseHOLD tables populated from parquet.")

    return expected


def _indexes_are_ready(expected):
    with engine.connect() as connection:
        tables = {
            row[0] for row in connection.execute(text("""
                SELECT name FROM sqlite_master
                WHERE type = 'table'
            """))
        }
        if not {"casehold_terms", "casehold_postings"}.issubset(tables):
            return False
        for scope, table, count in (
            ("holdings", "holdings", expected["holdings"]),
            ("citations", "examples", expected["examples"]),
        ):
            indexed_documents = connection.execute(text(
                "SELECT COUNT(DISTINCT doc_id) FROM casehold_postings "
                "WHERE scope = :scope"
            ), {"scope": scope}).scalar_one()
            term_count = connection.execute(text(
                "SELECT COUNT(*) FROM casehold_terms WHERE scope = :scope"
            ), {"scope": scope}).scalar_one()
            posting_count = connection.execute(text(
                "SELECT COUNT(*) FROM casehold_postings WHERE scope = :scope"
            ), {"scope": scope}).scalar_one()
            expected_docs = connection.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar_one()
            if (
                indexed_documents != expected_docs
                or expected_docs != count
                or not term_count
                or not posting_count
            ):
                return False
            posting_terms, posting_rows = connection.execute(text("""
                SELECT COUNT(DISTINCT term), COUNT(*)
                FROM casehold_postings
                WHERE scope = :scope
            """), {"scope": scope}).one()
            stored_terms, stored_postings = connection.execute(text("""
                SELECT COUNT(*), COALESCE(SUM(document_frequency), 0)
                FROM casehold_terms
                WHERE scope = :scope
            """), {"scope": scope}).one()
            if (posting_terms, posting_rows) != (stored_terms, stored_postings):
                return False
    return True


def _build_indexes(expected):
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS casehold_postings"))
        connection.execute(text("DROP TABLE IF EXISTS casehold_terms"))
        connection.execute(text("""
            CREATE TABLE casehold_terms (
                scope TEXT NOT NULL,
                term TEXT NOT NULL,
                document_frequency INTEGER NOT NULL,
                idf REAL NOT NULL,
                corpus_size INTEGER NOT NULL,
                average_document_length REAL NOT NULL,
                PRIMARY KEY (scope, term)
            )
        """))
        connection.execute(text("""
            CREATE TABLE casehold_postings (
                scope TEXT NOT NULL,
                term TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                term_frequency INTEGER NOT NULL,
                document_length INTEGER NOT NULL,
                tfidf_norm REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (scope, term, doc_id)
            )
        """))

        all_stats = {}
        for scope, source_table, id_column, text_column, document_count in (
            ("holdings", "holdings", "holding_id", "holding_text", expected["holdings"]),
            ("citations", "examples", "example_id", "citing_prompt", expected["examples"]),
        ):
            print(f"Building SQLite {scope} inverted index...")
            cursor = connection.execution_options(stream_results=True).execute(
                text(
                    f"SELECT {id_column}, {text_column} "
                    f"FROM {source_table} ORDER BY {id_column}"
                )
            )
            document_frequency = Counter()
            total_length = 0
            buffered = []
            processed_documents = 0
            for doc_id, content in cursor:
                processed_documents += 1
                frequencies = Counter(tokenize(content))
                doc_length = sum(frequencies.values())
                total_length += doc_length
                document_frequency.update(frequencies.keys())
                buffered.extend({
                    "scope": scope,
                    "term": term,
                    "doc_id": str(doc_id),
                    "term_frequency": frequency,
                    "document_length": doc_length,
                } for term, frequency in frequencies.items())
                if len(buffered) >= BATCH_SIZE:
                    connection.execute(text("""
                        INSERT INTO casehold_postings
                            (scope, term, doc_id, term_frequency, document_length)
                        VALUES
                            (:scope, :term, :doc_id, :term_frequency, :document_length)
                    """), buffered)
                    buffered.clear()
                if (
                    processed_documents % 5000 == 0
                    or processed_documents == document_count
                ):
                    print(
                        f"{scope.capitalize()} index progress: "
                        f"{processed_documents:,}/{document_count:,} documents.",
                        flush=True,
                    )
            if buffered:
                connection.execute(text("""
                    INSERT INTO casehold_postings
                        (scope, term, doc_id, term_frequency, document_length)
                    VALUES
                        (:scope, :term, :doc_id, :term_frequency, :document_length)
                """), buffered)

            average_length = total_length / document_count
            term_rows = []
            for term, df in document_frequency.items():
                term_rows.append({
                    "scope": scope,
                    "term": term,
                    "document_frequency": df,
                    "idf": math.log((document_count + 1) / (df + 1)) + 1,
                    "corpus_size": document_count,
                    "average_document_length": average_length,
                })
            for offset in range(0, len(term_rows), BATCH_SIZE):
                connection.execute(text("""
                    INSERT INTO casehold_terms
                        (scope, term, document_frequency, idf, corpus_size,
                         average_document_length)
                    VALUES
                        (:scope, :term, :document_frequency, :idf, :corpus_size,
                         :average_document_length)
                """), term_rows[offset:offset + BATCH_SIZE])
            all_stats[scope] = (len(document_frequency), total_length)

        # Norm updates are by document ID; create the lookup index before
        # updating each document so SQLite avoids rescanning all postings.
        connection.execute(text("""
            CREATE INDEX idx_casehold_postings_doc
            ON casehold_postings (scope, doc_id)
        """))

        for scope in ("holdings", "citations"):
            cursor = connection.execute(text("""
                SELECT p.doc_id, p.term_frequency, t.idf
                FROM casehold_postings p
                JOIN casehold_terms t
                  ON t.scope = p.scope AND t.term = p.term
                WHERE p.scope = :scope
                ORDER BY p.doc_id
            """), {"scope": scope})
            active_doc = None
            squared_norm = 0.0
            updates = []
            for doc_id, tf, idf in cursor:
                if active_doc is not None and doc_id != active_doc:
                    updates.append({
                        "scope": scope,
                        "doc_id": active_doc,
                        "norm": math.sqrt(squared_norm),
                    })
                    if len(updates) >= BATCH_SIZE:
                        connection.execute(text("""
                            UPDATE casehold_postings SET tfidf_norm = :norm
                            WHERE scope = :scope AND doc_id = :doc_id
                        """), updates)
                        updates.clear()
                    squared_norm = 0.0
                active_doc = doc_id
                squared_norm += ((1.0 + math.log(tf)) * idf) ** 2
            if active_doc is not None:
                updates.append({
                    "scope": scope,
                    "doc_id": active_doc,
                    "norm": math.sqrt(squared_norm),
                })
            if updates:
                connection.execute(text("""
                    UPDATE casehold_postings SET tfidf_norm = :norm
                    WHERE scope = :scope AND doc_id = :doc_id
                """), updates)

        for scope, (terms, _) in all_stats.items():
            actual = connection.execute(text(
                "SELECT COUNT(*) FROM casehold_terms WHERE scope = :scope"
            ), {"scope": scope}).scalar_one()
            if actual != terms:
                raise RuntimeError(f"SQLite {scope} index validation failed.")

    print("SQLite holdings and citation indexes are ready.")


def initialize_instance():
    download_dataset()
    expected = _ensure_base_data()
    if _indexes_are_ready(expected):
        print("Existing SQLite inverted indexes are complete; reusing them.")
    else:
        _build_indexes(expected)
    if not _indexes_are_ready(expected):
        raise RuntimeError("SQLite inverted-index validation failed.")


def ensure_search_indexes():
    """Repair missing SQLite search indexes from already loaded base tables."""
    with engine.connect() as connection:
        tables = {
            row[0] for row in connection.execute(text(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ))
        }
        missing = {"examples", "holdings", "candidates"} - tables
        if missing:
            raise RuntimeError(
                "SQLite CaseHOLD data is not initialized. Run setup.bat "
                "to download the dataset and prepare the database."
            )
        expected = {
            table: connection.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar_one()
            for table in ("examples", "holdings", "candidates")
        }

    if not expected["examples"] or not expected["holdings"]:
        raise RuntimeError(
            "SQLite CaseHOLD tables are empty. Run setup.bat to populate "
            "the database."
        )
    if not _indexes_are_ready(expected):
        print("SQLite search indexes are missing or incomplete; rebuilding...")
        _build_indexes(expected)
    if not _indexes_are_ready(expected):
        raise RuntimeError("SQLite inverted-index validation failed.")


def search(query, top_k=10, scope="holdings", method="bm25"):
    if scope not in {"holdings", "citations"}:
        raise ValueError("scope must be 'holdings' or 'citations'")
    if method not in {"bm25", "tfidf"}:
        raise ValueError("method must be 'bm25' or 'tfidf'")

    query_terms = Counter(tokenize(query))
    if not query_terms:
        return []

    from sqlalchemy import bindparam

    table, id_column, text_column = (
        ("holdings", "holding_id", "holding_text")
        if scope == "holdings"
        else ("examples", "example_id", "citing_prompt")
    )
    with engine.connect() as connection:
        terms = connection.execute(
            text("""
                SELECT term, idf, average_document_length
                FROM casehold_terms
                WHERE scope = :scope AND term IN :terms
            """).bindparams(bindparam("terms", expanding=True)),
            {"scope": scope, "terms": list(query_terms)},
        ).all()
        if not terms:
            return []

        idf = {term: value for term, value, _ in terms}
        average_length = terms[0][2]
        query_norm = math.sqrt(sum(
            ((1 + math.log(query_terms[term])) * idf[term]) ** 2
            for term in idf
        ))
        query_weights = {
            term: (1 + math.log(query_terms[term])) * idf[term]
            for term in idf
        }

        posting_rows = connection.execute(
            text("""
                SELECT term, doc_id, term_frequency, document_length, tfidf_norm
                FROM casehold_postings
                WHERE scope = :scope AND term IN :terms
            """).bindparams(bindparam("terms", expanding=True)),
            {"scope": scope, "terms": list(idf)},
        ).all()

        scores = {}
        for term, doc_id, tf, doc_len, doc_norm in posting_rows:
            if method == "tfidf":
                contribution = (
                    (1 + math.log(tf)) * idf[term] * query_weights[term]
                    / (doc_norm * query_norm)
                    if doc_norm and query_norm else 0.0
                )
            else:
                k1, b = 1.2, 0.75
                contribution = idf[term] * (
                    tf * (k1 + 1)
                    / (tf + k1 * (1 - b + b * doc_len / average_length))
                )
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        result_ids = [doc_id for doc_id, _ in ranked]
        documents = connection.execute(
            text(f"""
                SELECT CAST({id_column} AS TEXT), {text_column}
                FROM {table}
                WHERE CAST({id_column} AS TEXT) IN :doc_ids
            """).bindparams(bindparam("doc_ids", expanding=True)),
            {"doc_ids": result_ids},
        ).all() if result_ids else []
        text_by_id = {doc_id: content for doc_id, content in documents}

    return [
        {
            "doc_id": doc_id,
            "score": score,
            "holding_text": text_by_id.get(doc_id),
        }
        for doc_id, score in ranked
        if doc_id in text_by_id
    ]
