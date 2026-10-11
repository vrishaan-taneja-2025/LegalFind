
import json
from pathlib import Path

from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "casehold-train.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "holdings.jsonl"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if (
        OUTPUT_PATH.exists()
        and OUTPUT_PATH.stat().st_mtime >= DATA_PATH.stat().st_mtime
    ):
        print(f"Dataset export already exists: {OUTPUT_PATH}")
        return

    dataset = load_dataset(
        "parquet",
        data_files={"train": str(DATA_PATH)},
    )["train"]

    documents = {}

    print(f"Exporting {len(dataset):,} CaseHOLD examples to unique holdings...", flush=True)
    for row_number, example in enumerate(dataset, start=1):
        example_id = example["example_id"]
        prompt = example["citing_prompt"]
        label = int(example["label"])

        for index in range(5):
            text = example[f"holding_{index}"].strip()

            if not text:
                continue

            normalized = " ".join(text.split()).casefold()

            if normalized not in documents:
                documents[normalized] = {
                    "doc_id": f"H{len(documents) + 1}",
                    "text": text,
                    "sources": [],
                }

            documents[normalized]["sources"].append({
                "example_id": example_id,
                "candidate_index": index,
                "is_correct": index == label,
                "citing_prompt": prompt,
            })

        if row_number % 5000 == 0 or row_number == len(dataset):
            print(
                f"Export progress: {row_number:,}/{len(dataset):,} examples; "
                f"{len(documents):,} unique holdings.",
                flush=True,
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for document in documents.values():
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    print(f"Examples processed: {len(dataset)}")
    print(f"Unique holdings: {len(documents)}")
    print(f"Corpus saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
