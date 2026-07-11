from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "emails.csv"


def load_dataset() -> pd.DataFrame:
    """
    Load the dataset from data/emails.csv.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on the dataset.
    """

    df = df.dropna(subset=["email", "label"])

    df = df.drop_duplicates(subset="email")

    df["email"] = df["email"].astype(str).str.strip()

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Remove very short emails
    df = df[df["email"].str.len() > 20]

    # Keep only valid labels
    df = df[df["label"].isin(["spam", "ham"])]

    df = df.reset_index(drop=True)

    return df


def load_and_clean_dataset() -> pd.DataFrame:
    """
    Convenience function used by train.py.
    """

    df = load_dataset()
    df = clean_dataset(df)

    return df

