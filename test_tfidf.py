
from retrieval.inverted_index import build_index


def main():
    index = build_index()

    query = "government contract damages"
    results = index.tfidf_search(query, top_k=5)

    print(f"\nQuery: {query}")
    print("-" * 80)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Document: {result['doc_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Holding: {result['text'][:300]}")


if __name__ == "__main__":
    main()
