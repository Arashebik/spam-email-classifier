import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("data/emails.csv")

# Features
X_text = df["email"]

# Labels
y = df["label"]

# Text -> Numbers
vectorizer = CountVectorizer(stop_words="english",lowercase=True)

X = vectorizer.fit_transform(X_text)

# Create model
model = MultinomialNB()

# Train model
model.fit(X, y)

new_email = [
    "Congratulations! You have won free money."
]

X_new = vectorizer.transform(new_email)

prediction = model.predict(X_new)

print(prediction)