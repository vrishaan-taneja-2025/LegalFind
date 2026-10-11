
import math
import time
from collections import Counter
from itertools import groupby

from sqlalchemy import text
from tqdm import tqdm

from retrieval.db import engine
from retrieval.preprocessing import tokenize


def empty_metrics():
    return {
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "mrr": 0.0,
        "ndcg3": 0.0,
        "ndcg5": 0.0,
        "zero_score": 0,
    }


def evaluate():
    start_time = time.perf_counter()

    print("Loading term statistics...")

    with engine.connect() as conn:
        term_rows = conn.execute(
            text("SELECT term, idf FROM terms")
        ).mappings().all()

        idf = {row["term"]: float(row["idf"]) for row in term_rows}

        corpus = conn.execute(
            text("""
                SELECT
                    MAX(corpus_size) AS document_count,
                    MAX(average_document_length) AS avg_length
                FROM terms
            """)
        ).mappings().one()

        example_count = conn.execute(
            text("SELECT COUNT(*) FROM examples")
        ).scalar_one()

        holding_count = conn.execute(
            text("SELECT COUNT(*) FROM holdings")
        ).scalar_one()

    avg_length = float(corpus["avg_length"])

    if not idf or avg_length <= 0:
        raise RuntimeError("Term statistics are missing or invalid.")

    print(f"Vocabulary terms: {len(idf):,}")
    print(f"Examples to evaluate: {example_count:,}")
    print(f"Unique holdings: {holding_count:,}")
    print(f"Average holding length: {avg_length:.2f} tokens")

    # Cache each holding's term frequencies, length, and TF-IDF norm.
    print("\nPreparing holding vectors...")

    holding_vectors = {}
    holding_lengths = {}
    holding_norms = {}

    with engine.connect() as conn:
        result = conn.execution_options(
            stream_results=True,
            yield_per=1000,
        ).execute(
            text("""
                SELECT holding_id, holding_text
                FROM holdings
                ORDER BY holding_id
            """)
        ).mappings()

        for row in tqdm(
            result,
            total=holding_count,
            desc="Preparing holdings",
            unit="holding",
        ):
            doc_id = row["holding_id"]
            tf = Counter(tokenize(row["holding_text"]))

            holding_vectors[doc_id] = tf
            holding_lengths[doc_id] = sum(tf.values())

            norm_squared = sum(
                ((1.0 + math.log(freq)) * idf[term]) ** 2
                for term, freq in tf.items()
                if term in idf
            )

            holding_norms[doc_id] = math.sqrt(norm_squared)

    metrics = {
        "BM25": empty_metrics(),
        "TF-IDF cosine": empty_metrics(),
    }

    total = 0

    print("\nEvaluating ranking metrics...")

    with engine.connect() as conn:
        result = conn.execution_options(
            stream_results=True,
            yield_per=1000,
        ).execute(
            text("""
                SELECT
                    e.example_id,
                    e.citing_prompt,
                    e.label,
                    c.candidate_index,
                    c.holding_id
                FROM examples e
                JOIN candidates c
                    ON c.example_id = e.example_id
                ORDER BY e.example_id, c.candidate_index
            """)
        ).mappings()

        grouped_examples = groupby(
            result,
            key=lambda row: row["example_id"],
        )

        for example_id, group in tqdm(
            grouped_examples,
            total=example_count,
            desc="Evaluating examples",
            unit="example",
        ):
            candidates = list(group)

            if len(candidates) != 5:
                raise RuntimeError(
                    f"Example {example_id} has "
                    f"{len(candidates)} candidates, expected 5."
                )

            query = candidates[0]["citing_prompt"]
            correct_label = candidates[0]["label"]
            query_tf = Counter(tokenize(query))

            # TF-IDF weights for the query.
            query_weights = {
                term: (1.0 + math.log(freq)) * idf[term]
                for term, freq in query_tf.items()
                if term in idf
            }

            query_norm = math.sqrt(
                sum(weight * weight for weight in query_weights.values())
            )

            bm25_rankings = []
            cosine_rankings = []

            for candidate in candidates:
                doc_id = candidate["holding_id"]
                candidate_index = candidate["candidate_index"]

                doc_tf = holding_vectors[doc_id]
                doc_len = holding_lengths[doc_id]

                # BM25: k1 = 1.2, b = 0.75.
                bm25_score = 0.0

                for term in query_tf:
                    if term not in idf:
                        continue

                    freq = doc_tf.get(term, 0)
                    if freq == 0:
                        continue

                    length_norm = (
                        0.25 + 0.75 * doc_len / avg_length
                    )

                    bm25_score += (
                        idf[term]
                        * freq * 2.2
                        / (freq + 1.2 * length_norm)
                    )

                # TF-IDF cosine similarity.
                dot_product = sum(
                    query_weight
                    * (1.0 + math.log(doc_tf[term]))
                    * idf[term]
                    for term, query_weight in query_weights.items()
                    if doc_tf.get(term, 0) > 0
                )

                doc_norm = holding_norms[doc_id]

                cosine_score = (
                    dot_product / (query_norm * doc_norm)
                    if query_norm > 0 and doc_norm > 0
                    else 0.0
                )

                bm25_rankings.append(
                    (candidate_index, bm25_score)
                )
                cosine_rankings.append(
                    (candidate_index, cosine_score)
                )

            # Stable tie-breaking for reproducible results.
            bm25_rankings.sort(key=lambda x: (-x[1], x[0]))
            cosine_rankings.sort(key=lambda x: (-x[1], x[0]))

            for name, rankings in (
                ("BM25", bm25_rankings),
                ("TF-IDF cosine", cosine_rankings),
            ):
                correct_rank = next(
                    rank
                    for rank, (candidate_index, _) in enumerate(
                        rankings, start=1
                    )
                    if candidate_index == correct_label
                )

                all_scores_zero = all(
                    score <= 0 for _, score in rankings
                )

                values = metrics[name]

                values["top1"] += correct_rank == 1
                values["top3"] += correct_rank <= 3
                values["top5"] += correct_rank <= 5
                values["mrr"] += 1.0 / correct_rank
                values["ndcg3"] += (
                    1.0 / math.log2(correct_rank + 1)
                    if correct_rank <= 3 else 0.0
                )
                values["ndcg5"] += (
                    1.0 / math.log2(correct_rank + 1)
                    if correct_rank <= 5 else 0.0
                )
                values["zero_score"] += all_scores_zero

            total += 1

    if total == 0:
        raise RuntimeError("No examples were evaluated.")

    # Display final metrics.
    print("\n" + "=" * 58)
    print("CASEHOLD RETRIEVAL EVALUATION")
    print("=" * 58)
    print(f"Examples evaluated: {total:,}")
    print("Candidates per example: 5")
    print()

    for name, values in metrics.items():
        print(name)
        print("-" * len(name))

        print(
            f"Accuracy@1:    {values['top1'] / total:.4f}"
            f" ({100 * values['top1'] / total:.2f}%)"
        )
        print(
            f"Accuracy@3:    {values['top3'] / total:.4f}"
            f" ({100 * values['top3'] / total:.2f}%)"
        )
        print(
            f"Accuracy@5:    {values['top5'] / total:.4f}"
            f" ({100 * values['top5'] / total:.2f}%)"
        )
        print(f"MRR:           {values['mrr'] / total:.4f}")
        print(f"nDCG@3:        {values['ndcg3'] / total:.4f}")
        print(f"nDCG@5:        {values['ndcg5'] / total:.4f}")
        print(
            f"Zero-score:    {values['zero_score'] / total:.4f}"
            f" ({100 * values['zero_score'] / total:.2f}%)"
        )
        print()

    print(f"Elapsed time: {time.perf_counter() - start_time:.1f} seconds")


if __name__ == "__main__":
    evaluate()
