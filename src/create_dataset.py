from pathlib import Path
from email import policy
from email.parser import BytesParser

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = BASE_DIR / "data" / "emails.csv"


def extract_email_content(file_path: Path) -> str:
    """
    Extract only the Subject and plain-text body from an email.
    """

    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        subject = msg["subject"] or ""

        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if (
                    part.get_content_type() == "text/plain"
                    and part.get_content_disposition() is None
                ):
                    body += part.get_content()
        else:
            body = msg.get_content()

        text = f"{subject}\n{body}"

        return " ".join(text.split())

    except Exception:
        return ""


def read_folder(folder: Path, label: str):
    rows = []

    for file in folder.iterdir():

        if not file.is_file():
            continue

        text = extract_email_content(file)

        if not text:
            continue

        rows.append(
            {
                "email": text,
                "label": label,
            }
        )

    return rows


def main():

    dataset = []

    dataset.extend(read_folder(RAW_DATA_DIR / "easy_ham", "ham"))
    dataset.extend(read_folder(RAW_DATA_DIR / "hard_ham", "ham"))
    dataset.extend(read_folder(RAW_DATA_DIR / "spam", "spam"))

    df = pd.DataFrame(dataset)

    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 50)
    print("Dataset Created Successfully")
    print("=" * 50)
    print(f"Total Emails : {len(df)}")
    print(df["label"].value_counts())
    print()
    print(df.head())


if __name__ == "__main__":
    main()