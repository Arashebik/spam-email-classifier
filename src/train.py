from pathlib import Path

import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from dataloader import load_and_clean_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():

    print("[1/6] Loading dataset...")

    df = load_and_clean_dataset()

    X_text = df["email"]
    y = df["label"]

    print(f"Total emails: {len(df)}")

    print("\n[2/6] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("[3/6] Building pipeline...")

    pipeline = Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    lowercase=True,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                MultinomialNB(),
            ),
        ]
    )

    print("[4/6] Training model...")

    pipeline.fit(X_train, y_train)

    print("[5/6] Evaluating model...")

    y_pred = pipeline.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")

    print("Classification Report")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\n[6/6] Saving model...")

    joblib.dump(
        pipeline,
        MODELS_DIR / "spam_classifier.pkl",
    )

    print("\nTraining completed successfully!")
    print(f"Model saved to: {MODELS_DIR / 'spam_classifier.pkl'}")


if __name__ == "__main__":
    main()