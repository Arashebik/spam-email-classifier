from pathlib import Path

from src import create_dataset
from src import dataloader
from src import train
from src import predict


def _ask(prompt: str, valid_options):
    """Prompt the user until a valid option is provided."""

    while True:
        value = input(f"{prompt}: ").strip()
        if value in valid_options:
            return value
        print(f"Please choose one of: {', '.join(valid_options)}")


def _dataset_exists() -> bool:
    return dataloader.DATASET_PATH.exists()


def _run_dataset_step() -> bool:
    try:
        create_dataset.create_dataset_main()
        _ = dataloader.load_and_clean_dataset()
        return True
    except Exception as exc:  # pragma: no cover
        print(f"Dataset creation failed: {exc}")
        return False


def _run_validation_step() -> None:
    try:
        df = dataloader.load_and_clean_dataset()
    except Exception as exc:
        print(f"Could not load dataset: {exc}")
        return

    print(f"Loaded {len(df)} cleaned emails")
    print(f"Label counts: {df['label'].value_counts().to_dict()}")


def _run_train_step() -> bool:
    try:
        train.main()
        return True
    except Exception as exc:  # pragma: no cover
        print(f"Training failed: {exc}")
        return False


def _run_predict_step() -> None:
    try:
        predict.main()
    except Exception as exc:  # pragma: no cover
        print(f"Prediction failed: {exc}")


def _run_full_pipeline() -> None:
    print("Running full pipeline: create -> train -> predict")

    if not _run_dataset_step():
        print("Full pipeline failed at dataset creation.")
        return

    if not _run_train_step():
        print("Full pipeline failed at training.")
        return

    if not Path(predict.MODEL_PATH).exists():
        print("Model file not found after training.")
        return

    _run_predict_step()


def main():
    print("Hello! Welcome to spam email classifier")

    while True:
        print("\n--- Main Menu ---")
        print("1) Step 1: Create dataset")
        print("2) Step 2: Validate dataset")
        print("3) Step 3: Train model")
        print("4) Step 4: Predict")
        print("5) Run full pipeline")
        print("6) Exit")

        choice = _ask("Choose an option", ["1", "2", "3", "4", "5", "6"])

        if choice == "1":
            print("\nStep 1 started")
            if _run_dataset_step():
                print("Step 1 ended")
            else:
                print("Step 1 ended with errors")

        elif choice == "2":
            print("\nStep 2 started")
            if not _dataset_exists():
                print(f"Dataset missing at {dataloader.DATASET_PATH}")
                print("Step 2 ended")
                continue
            _run_validation_step()
            print("Step 2 ended")

        elif choice == "3":
            print("\nStep 3 started")
            if not _dataset_exists():
                print(f"Dataset missing at {dataloader.DATASET_PATH}")
                print("Train step skipped")
                continue
            if _run_train_step():
                print("Step 3 ended")
            else:
                print("Step 3 ended with errors")

        elif choice == "4":
            print("\nStep 4 started")
            if not Path(predict.MODEL_PATH).exists():
                print(f"Model missing at {predict.MODEL_PATH}")
                print("Step 4 ended")
                continue
            _run_predict_step()
            print("Step 4 ended")

        elif choice == "5":
            _run_full_pipeline()

        else:
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
