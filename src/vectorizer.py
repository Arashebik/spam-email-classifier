import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

# reading the dataset
df = pd.read_csv("data/emails.csv")

# vectorizing
vectorizer = CountVectorizer(stop_words="english",lowercase=True)

# convert the text to matrix
X = vectorizer.fit_transform(df["email"])
# show the vocabulary words
print(vectorizer.get_feature_names_out())

# show the matrix
print(X.toarray())