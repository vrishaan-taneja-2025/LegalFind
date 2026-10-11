from datasets import load_dataset

def main():
    path = "casehold-train.parquet"

    print("Loading local CaseHOLD Parquet file...")

    dataset = load_dataset(
        "parquet",
        data_files={"train": path},
    )

    train = dataset["train"]

    print("\nDataset overview:")
    print(dataset)

    print("\nColumns:")
    print(train.column_names)

    print("\nNumber of records:", len(train))

    print("\nFirst record:")
    print(train[0])


if __name__ == "__main__":
    main()