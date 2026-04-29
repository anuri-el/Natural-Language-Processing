import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    df = pd.read_csv("./outputs/l3_topics.csv")

    TOPIC_NAMES = df["topic"].unique().tolist()
    UNKNOWN_LABEL = "невідомо / unknown"

    descriptions = df["description"].tolist()
    TOPIC_DOCS = {}
    for topic in TOPIC_NAMES:
        topic_descriptions = df[df["topic"] == topic]["description"].tolist()
        TOPIC_DOCS[topic] = " ".join(topic_descriptions)

    tfidf_vec, tfidf_matrix, bow_vec, bow_matrix = build_vectorizers(TOPIC_DOCS)

    # while True:
    #     try:
    #         text = input("\n > ").strip()
    #     except Exception:
    #         break

    #     if not text or text.lower() in ("exit", "quit", "q"):
    #         break


def build_vectorizers(topic_docs):
    corpus = list(topic_docs.values())

    tfidf_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1,2), min_df=1, sublinear_tf=True)
    tfidf_matrix = tfidf_vec.fit_transform(corpus)

    bow_vec = CountVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1)
    bow_matrix = bow_vec.fit_transform(corpus)

    return tfidf_vec, tfidf_matrix, bow_vec, bow_matrix


if __name__ == "__main__":
    main()