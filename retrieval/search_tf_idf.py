"""TF-IDF cosine search over the persisted PostgreSQL inverted index."""

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


def search(query: str, top_k: int = 10, scope: str = "holdings") -> list[dict]:
    """Return records ranked by TF-IDF cosine similarity.

    Term frequencies, document vector norms, and inverse document frequencies
    are read from the persisted index tables for the selected corpus.
    """
    if scope not in SCOPES:
        raise ValueError("scope must be 'holdings' or 'citations'")
    if DATABASE_BACKEND == "sqlite":
        from retrieval.sqlite_backend import search as sqlite_search

        return sqlite_search(query, top_k=top_k, scope=scope, method="tfidf")
    terms = tokenize(query)
    if not terms:
        return []
    terms_table, lists_table, documents_table, id_column, text_column = SCOPES[scope]

    with engine.connect() as connection:
        rows = connection.execute(
            text(f"""
                WITH query_terms AS (
                    SELECT term, COUNT(*)::DOUBLE PRECISION AS query_tf
                    FROM unnest(CAST(:query_terms AS TEXT[])) AS q(term)
                    GROUP BY term
                ),
                weighted_query AS (
                    SELECT
                        qt.term,
                        qt.query_tf,
                        t.idf,
                        (1.0 + LN(qt.query_tf)) * t.idf AS query_weight
                    FROM query_terms qt
                    JOIN {terms_table} t ON t.term = qt.term
                ),
                query_norm AS (
                    SELECT SQRT(SUM(POWER(query_weight, 2))) AS norm
                    FROM weighted_query
                ),
                matched AS (
                    SELECT
                        p.item->>'doc_id' AS doc_id,
                        (p.item->>'tf')::DOUBLE PRECISION AS tf,
                        (p.item->>'tfidf_norm')::DOUBLE PRECISION AS document_norm,
                        w.query_weight,
                        w.idf
                    FROM weighted_query w
                    JOIN {terms_table} t ON t.term = w.term
                    JOIN {lists_table} pl ON pl.term_id = t.term_id
                    CROSS JOIN LATERAL
                        jsonb_array_elements(pl.posting_list) AS p(item)
                ),
                scored AS (
                    SELECT
                        m.doc_id,
                        SUM((1.0 + LN(m.tf)) * m.idf * m.query_weight)
                            / NULLIF(MAX(m.document_norm) * qn.norm, 0) AS score
                    FROM matched m
                    CROSS JOIN query_norm qn
                    GROUP BY m.doc_id, qn.norm
                )
                SELECT s.doc_id, s.score, d.{text_column} AS holding_text
                FROM scored s
                JOIN {documents_table} d ON d.{id_column}::TEXT = s.doc_id
                WHERE s.score > 0
                ORDER BY s.score DESC, s.doc_id
                LIMIT :top_k
            """),
            {"query_terms": terms, "top_k": top_k},
        ).mappings().all()

    return [dict(row) for row in rows]
