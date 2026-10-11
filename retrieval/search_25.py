"""Okapi BM25 search over the persisted PostgreSQL inverted index."""

from sqlalchemy import text

from retrieval.db import DATABASE_BACKEND, engine
from retrieval.preprocessing import tokenize


SCOPES = {
    "holdings": ("terms", "posting_lists", "holdings", "holding_id", "holding_text"),
    "citations": (
        "citation_terms",
        "citation_posting_lists",
        "examples",
        "example_id",
        "citing_prompt",
    ),
}


def search(
    query: str,
    top_k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
    scope: str = "holdings",
) -> list[dict]:
    """Return records ranked with BM25 using persisted term postings."""
    if scope not in SCOPES:
        raise ValueError("scope must be 'holdings' or 'citations'")
    if DATABASE_BACKEND == "sqlite":
        from retrieval.sqlite_backend import search as sqlite_search

        return sqlite_search(query, top_k=top_k, scope=scope, method="bm25")
    terms = list(set(tokenize(query)))
    if not terms:
        return []
    terms_table, lists_table, documents_table, id_column, text_column = SCOPES[scope]

    with engine.connect() as connection:
        rows = connection.execute(
            text(f"""
                WITH query_terms AS (
                    SELECT DISTINCT term
                    FROM unnest(CAST(:query_terms AS TEXT[])) AS q(term)
                ),
                matched AS (
                    SELECT
                        t.idf,
                        p.item->>'doc_id' AS doc_id,
                        (p.item->>'tf')::DOUBLE PRECISION AS tf,
                        (p.item->>'doc_len')::DOUBLE PRECISION AS doc_len
                    FROM query_terms q
                    JOIN {terms_table} t ON t.term = q.term
                    JOIN {lists_table} pl ON pl.term_id = t.term_id
                    CROSS JOIN LATERAL
                        jsonb_array_elements(pl.posting_list) AS p(item)
                ),
                corpus AS (
                    SELECT MAX(average_document_length)::DOUBLE PRECISION AS avg_len
                    FROM {terms_table}
                ),
                scored AS (
                    SELECT
                        m.doc_id,
                        SUM(
                            m.idf * (
                                m.tf * (:k1 + 1.0)
                                / NULLIF(
                                    m.tf + :k1 * (
                                        1.0 - :b
                                        + :b * m.doc_len / NULLIF(c.avg_len, 0)
                                    ),
                                    0
                                )
                            )
                        ) AS score
                    FROM matched m
                    CROSS JOIN corpus c
                    GROUP BY m.doc_id
                )
                SELECT s.doc_id, s.score, d.{text_column} AS holding_text
                FROM scored s
                JOIN {documents_table} d ON d.{id_column}::TEXT = s.doc_id
                ORDER BY s.score DESC, s.doc_id
                LIMIT :top_k
            """),
            {"query_terms": terms, "top_k": top_k, "k1": k1, "b": b},
        ).mappings().all()

    return [dict(row) for row in rows]
