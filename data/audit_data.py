
from collections import Counter
from pathlib import Path

from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "casehold-train.parquet"


def normalize(text):
    return " ".join(str(text).split()).casefold()


def main():
    dataset = load_dataset(
        "parquet",
        data_files={"train": str(DATA_PATH)},
    )["train"]

    print("=== DATASET STRUCTURE ===")
    print(f"Examples: {len(dataset):,}")
    print(f"Columns: {dataset.column_names}")

    label_counts = Counter()
    unique_prompts = set()
    unique_holdings = set()
    candidate_counts = Counter()
    empty_fields = Counter()
    example_ids = set()
    duplicate_ids = 0

    total_candidate_instances = 0

    for row in dataset:
        example_id = row["example_id"]
        prompt = row["citing_prompt"]
        label = int(row["label"])

        if example_id in example_ids:
            duplicate_ids += 1
        example_ids.add(example_id)

        label_counts[label] += 1
        unique_prompts.add(normalize(prompt))

        if not normalize(prompt):
            empty_fields["citing_prompt"] += 1

        for i in range(5):
            text = row[f"holding_{i}"]
            normalized = normalize(text)

            candidate_counts[i] += 1
            total_candidate_instances += 1

            if not normalized:
                empty_fields[f"holding_{i}"] += 1
                continue

            unique_holdings.add(normalized)

    print("\n=== EXAMPLE AND CANDIDATE COUNTS ===")
    print(f"Unique example IDs: {len(example_ids):,}")
    print(f"Duplicate example IDs: {duplicate_ids:,}")
    print(f"Candidate instances: {total_candidate_instances:,}")
    print(f"Unique normalized citing prompts: {len(unique_prompts):,}")
    print(f"Unique normalized holdings: {len(unique_holdings):,}")

    print("\n=== CORRECT ANSWER LABELS ===")
    for label, count in sorted(label_counts.items()):
        print(f"holding_{label}: {count:,} examples")

    print("\n=== EMPTY FIELDS ===")
    print(empty_fields if empty_fields else "None found")

    print("\n=== FIRST EXAMPLE ===")
    row = dataset[0]
    print(f"Example ID: {row['example_id']}")
    print(f"Correct candidate: holding_{row['label']}")
    print(f"\nCiting prompt:\n{row['citing_prompt'][:700]}")

    for i in range(5):
        marker = " [CORRECT]" if i == int(row["label"]) else ""
        print(f"\nholding_{i}{marker}:")
        print(row[f"holding_{i}"][:400])


if __name__ == "__main__":
    main()
