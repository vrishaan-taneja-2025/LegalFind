"""Download the official CaseHOLD training split as the project's parquet file."""

import argparse
import os
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "casehold-train.parquet"
REPOSITORY = "casehold/casehold"
CONFIGURATION = "all"
SPLIT = "train"
PARQUET_REVISION = "refs/convert/parquet"
PARQUET_FILE = f"{CONFIGURATION}/{SPLIT}/0000.parquet"
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
        print(
            f"Downloading {REPOSITORY} ({CONFIGURATION}/{SPLIT}) "
            "Parquet from Hugging Face..."
        )
        parquet_path = hf_hub_download(
            repo_id=REPOSITORY,
            repo_type="dataset",
            filename=PARQUET_FILE,
            revision=PARQUET_REVISION,
        )
        dataset = load_dataset(
            "parquet", data_files={SPLIT: parquet_path}, split=SPLIT
        )
        missing = REQUIRED_COLUMNS.difference(dataset.column_names)
        if missing:
            raise ValueError(
                "Downloaded CaseHOLD data is missing required columns: "
                + ", ".join(sorted(missing))
            )
        if len(dataset) == 0:
            raise ValueError("The official CaseHOLD training split is empty.")

        shutil.copyfile(parquet_path, temporary_path)
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
