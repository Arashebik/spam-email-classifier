from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"


def load_model():
    """
    Load the trained spam classifier.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Train the model first by running train.py."
        )

    return joblib.load(MODEL_PATH)


def predict_email(model, email: str) -> str:
    """
    Predict whether an email is spam or ham.
    """

    prediction = model.predict([email])[0]

    return prediction


def main():

    print("=" * 50)
    print("Spam Email Classifier")
    print("=" * 50)

    model = load_model()

    while True:

        print("\nType 'exit' to quit.\n")

        email = input("Enter an email:\n> ").strip()

        if email.lower() == "exit":
            print("\nGoodbye!")
            break

        if not email:
            print("\nEmail cannot be empty.")
            continue

        prediction = predict_email(model, email)

        if prediction == "spam":
            print("\n🚨 Prediction: SPAM")
        else:
            print("\n✅ Prediction: HAM")


if __name__ == "__main__":
    main()