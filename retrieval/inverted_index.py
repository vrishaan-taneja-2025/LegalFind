
from collections import defaultdict
from math import log

from sqlalchemy import text

from retrieval.db import engine
from retrieval.preprocessing import tokenize


class InvertedIndex:
    def __init__(self):
        # term -> {document ID: term frequency}
        self.postings = defaultdict(dict)

        # term -> number of documents containing the term
        self.document_frequency = {}

        # document ID -> number of tokens in that document
        self.document_lengths = {}

        # document ID -> original holding text
        self.documents = {}

        self.document_norms = {}

        self.total_documents = 0
        self.total_length = 0

    @property
    def average_document_length(self):
        if self.total_documents == 0:
            return 0.0

        return self.total_length / self.total_documents

    def add_document(self, doc_id: str, document_text: str):
        """Add one document and update the inverted index."""
        if doc_id in self.documents:
            return

        tokens = tokenize(document_text)
        term_frequencies = defaultdict(int)

        for token in tokens:
            term_frequencies[token] += 1

        self.documents[doc_id] = document_text
        self.document_lengths[doc_id] = len(tokens)

        for term, frequency in term_frequencies.items():
            self.postings[term][doc_id] = frequency

        self.total_documents += 1
        self.total_length += len(tokens)

    
    def finalize(self):
        """Calculate document frequencies and full TF-IDF document norms."""
        self.document_frequency = {
            term: len(documents)
            for term, documents in self.postings.items()
        }

        self.document_norms = defaultdict(float)

        for term, documents in self.postings.items():
            term_idf = self.idf(term)

            for doc_id, frequency in documents.items():
                weight = (1 + log(frequency)) * term_idf
                self.document_norms[doc_id] += weight ** 2

        self.document_norms = {
            doc_id: norm ** 0.5
            for doc_id, norm in self.document_norms.items()
        }


    def candidate_documents(self, query: str) -> set[str]:
        """Return documents containing at least one query term."""
        candidates = set()

        for term in set(tokenize(query)):
            candidates.update(self.postings.get(term, {}).keys())

        return candidates

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency."""
        df = self.document_frequency.get(term, 0)
        n = self.total_documents

        if n == 0:
            return 0.0

        return log(1 + (n - df + 0.5) / (df + 0.5))

    
    def tfidf_search(self, query: str, top_k: int = 10):
        """Rank documents using TF-IDF cosine similarity."""
        query_terms = tokenize(query)

        if not query_terms:
            return []

        # TF-IDF weights for the query
        query_tf = defaultdict(int)
        for term in query_terms:
            query_tf[term] += 1

        query_weights = {
            term: (1 + log(count)) * self.idf(term)
            for term, count in query_tf.items()
            if term in self.postings
        }

        if not query_weights:
            return []

        query_norm = sum(
            weight ** 2 for weight in query_weights.values()
        ) ** 0.5

        scores = defaultdict(float)

        for term, query_weight in query_weights.items():
            for doc_id, frequency in self.postings[term].items():
                doc_weight = (1 + log(frequency)) * self.idf(term)
                scores[doc_id] += doc_weight * query_weight

        results = []

        for doc_id, dot_product in scores.items():
            doc_norm = self.document_norms.get(doc_id, 0.0)

            if doc_norm == 0 or query_norm == 0:
                continue

            similarity = dot_product / (doc_norm * query_norm)

            results.append({
                "doc_id": doc_id,
                "text": self.documents[doc_id],
                "score": similarity,
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def bm25_search(self, query: str, top_k: int = 10, k1: float = 1.2, b: float = 0.75):
        """Rank this in-memory corpus with the project's Okapi BM25 settings."""
        query_terms = set(tokenize(query))
        if not query_terms or not self.documents or not self.average_document_length:
            return []

        scores = defaultdict(float)
        for term in query_terms:
            frequency_by_doc = self.postings.get(term, {})
            if not frequency_by_doc:
                continue
            df = len(frequency_by_doc)
            idf = log(1 + (self.total_documents - df + 0.5) / (df + 0.5))
            for doc_id, frequency in frequency_by_doc.items():
                length = self.document_lengths[doc_id]
                denominator = frequency + k1 * (
                    1 - b + b * length / self.average_document_length
                )
                scores[doc_id] += idf * frequency * (k1 + 1) / denominator

        results = [
            {"doc_id": doc_id, "text": self.documents[doc_id], "score": score}
            for doc_id, score in scores.items()
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]



def build_index(scope: str = "holdings"):
    """Load holdings or citing prompts into the existing in-memory index."""
    if scope not in {"holdings", "citations"}:
        raise ValueError("scope must be 'holdings' or 'citations'")

    index = InvertedIndex()

    with engine.connect() as connection:
        if scope == "citations":
            statement = text(
                "SELECT example_id::text, citing_prompt FROM examples ORDER BY example_id"
            )
        else:
            tables = connection.execute(text("""
                SELECT
                    to_regclass('public.holdings') IS NOT NULL AS has_holdings,
                    to_regclass('public.documents') IS NOT NULL AS has_documents
            """)).one()

            if tables.has_holdings:
                statement = text(
                    "SELECT holding_id, holding_text FROM holdings ORDER BY holding_id"
                )
            elif tables.has_documents:
                statement = text(
                    "SELECT doc_id, text FROM documents ORDER BY doc_id"
                )
            else:
                raise RuntimeError(
                    "Neither the holdings nor documents table exists in the database."
                )

        result = connection.execute(statement)

        for doc_id, document_text in result:
            index.add_document(doc_id, document_text)

    index.finalize()
    return index


if __name__ == "__main__":
    print("Building inverted index from PostgreSQL...")
    index = build_index()

    print(f"Documents indexed: {index.total_documents:,}")
    print(f"Unique terms: {len(index.postings):,}")
    print(f"Average document length: {index.average_document_length:.2f}")
    print(f"Documents containing 'contract': {index.document_frequency.get('contract', 0):,}")
