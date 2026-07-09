import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def download_nltk_resource(resource_path, package_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name)


download_nltk_resource("tokenizers/punkt", "punkt")
download_nltk_resource("tokenizers/punkt_tab/english", "punkt_tab")
download_nltk_resource("corpora/stopwords", "stopwords")

df = pd.read_csv("data/emails.csv")

stop_words = set(stopwords.words("english"))

def preprocess(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

df["tokens"] = df["email"].apply(preprocess)

print(df[["email", "tokens"]].head())