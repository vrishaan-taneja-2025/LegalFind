"""FastAPI adapter for the existing CaseHOLD retrieval implementations."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)
FRONTEND = Path(__file__).resolve().parent / "casehold-frontend.html"


@asynccontextmanager
async def lifespan(_app):
    from retrieval.db import DATABASE_BACKEND

    if DATABASE_BACKEND == "sqlite":
        from retrieval.sqlite_backend import ensure_search_indexes

        ensure_search_indexes()
    yield


app = FastAPI(title="LegalFind", version="1.0.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    scope: str = Field(default="holdings", pattern="^(holdings|citations)$")


class DetailsRequest(BaseModel):
    scope: str = Field(pattern="^(holdings|citations)$")
    doc_id: str = Field(min_length=1, max_length=100)


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(FRONTEND, media_type="text/html")


@app.post("/search")
def search(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Enter a search query.")
    try:
        from retrieval.search_25 import search as search_bm25
        from retrieval.search_tf_idf import search as search_tf_idf

        bm25_results = search_bm25(query, top_k=10, scope=request.scope)
        tfidf_results = search_tf_idf(query, top_k=10, scope=request.scope)
        return {
            "bm25": [
                {
                    "doc_id": str(item["doc_id"]),
                    "score": float(item["score"]),
                    "holding_text": item.get("holding_text", item.get("text")),
                }
                for item in bm25_results
            ],
            "tfidf": [
                {
                    "doc_id": str(item["doc_id"]),
                    "score": float(item["score"]),
                    "holding_text": item["holding_text"],
                }
                for item in tfidf_results
            ],
        }
    except Exception as exc:
        LOGGER.exception("LegalFind search failed")
        raise HTTPException(
            status_code=503,
            detail="Search is unavailable. Check that the database and retrieval indexes are ready.",
        ) from exc


@app.post("/details")
def record_details(request: DetailsRequest):
    """Return the full CaseHOLD context associated with a ranked record."""
    from sqlalchemy import text

    from retrieval.db import engine

    try:
        with engine.connect() as connection:
            if request.scope == "citations":
                try:
                    example_id = int(request.doc_id)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail="Invalid citation record.") from exc

                example = connection.execute(text("""
                    SELECT example_id, citing_prompt, label
                    FROM examples
                    WHERE example_id = :example_id
                """), {"example_id": example_id}).mappings().first()
                if example is None:
                    raise HTTPException(status_code=404, detail="Citation record not found.")

                candidates = connection.execute(text("""
                    SELECT c.candidate_index, c.holding_id, h.holding_text
                    FROM candidates c
                    JOIN holdings h ON h.holding_id = c.holding_id
                    WHERE c.example_id = :example_id
                    ORDER BY c.candidate_index
                """), {"example_id": example_id}).mappings().all()
                label = int(example["label"])
                return {
                    "scope": "citations",
                    "doc_id": str(example["example_id"]),
                    "citing_prompt": example["citing_prompt"],
                    "label": label,
                    "candidates": [
                        {
                            "candidate_index": int(candidate["candidate_index"]),
                            "holding_id": candidate["holding_id"],
                            "holding_text": candidate["holding_text"],
                            "is_label": int(candidate["candidate_index"]) == label,
                        }
                        for candidate in candidates
                    ],
                }

            holding = connection.execute(text("""
                SELECT holding_id, holding_text
                FROM holdings
                WHERE holding_id = :holding_id
            """), {"holding_id": request.doc_id}).mappings().first()
            if holding is None:
                raise HTTPException(status_code=404, detail="Holding not found.")

            sources = connection.execute(text("""
                SELECT e.example_id, e.citing_prompt, e.label, c.candidate_index
                FROM candidates c
                JOIN examples e ON e.example_id = c.example_id
                WHERE c.holding_id = :holding_id
                ORDER BY e.example_id, c.candidate_index
            """), {"holding_id": request.doc_id}).mappings().all()
            return {
                "scope": "holdings",
                "doc_id": holding["holding_id"],
                "holding_text": holding["holding_text"],
                "sources": [
                    {
                        "example_id": int(source["example_id"]),
                        "citing_prompt": source["citing_prompt"],
                        "label": int(source["label"]),
                        "candidate_index": int(source["candidate_index"]),
                        "is_label": int(source["candidate_index"]) == int(source["label"]),
                    }
                    for source in sources
                ],
            }
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("LegalFind record details lookup failed")
        raise HTTPException(
            status_code=503,
            detail="Full CaseHOLD context is unavailable. Check that the database is ready.",
        ) from exc
