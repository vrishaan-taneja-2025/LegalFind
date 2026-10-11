"""Download the official CaseHOLD training split as the project's parquet file."""

import argparse
import os
from pathlib import Path

from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "casehold-train.parquet"
REPOSITORY = "casehold/casehold"
CONFIGURATION = "all"
SPLIT = "train"
REQUIRED_COLUMNS = {
    "example_id",
    "citing_prompt",
    "label",
    *(f"holding_{index}" for index in range(5)),
}


def download_dataset(force: bool = False) -> Path:
    if DATA_PATH.exists() and not force:
        print(f"Dataset already exists: {DATA_PATH}")
        print("Use --force to download it again.")
        return DATA_PATH

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = DATA_PATH.with_suffix(".parquet.part")
    try:
        print(f"Downloading {REPOSITORY} ({CONFIGURATION}/{SPLIT}) from Hugging Face...")
        dataset = load_dataset(REPOSITORY, CONFIGURATION, split=SPLIT)
        missing = REQUIRED_COLUMNS.difference(dataset.column_names)
        if missing:
            raise ValueError(
                "Downloaded CaseHOLD data is missing required columns: "
                + ", ".join(sorted(missing))
            )
        if len(dataset) == 0:
            raise ValueError("The official CaseHOLD training split is empty.")

        dataset.to_parquet(str(temporary_path))
        os.replace(temporary_path, DATA_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Saved {len(dataset):,} training examples to {DATA_PATH}")
    return DATA_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="replace the local parquet file"
    )
    args = parser.parse_args()
    download_dataset(force=args.force)


if __name__ == "__main__":
    main()
