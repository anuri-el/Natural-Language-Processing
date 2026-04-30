import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    df = pd.read_csv("./outputs/l3_topics.csv")

    TOPIC_NAMES = df["topic"].unique().tolist()

    descriptions = df["description"].tolist()
    TOPIC_DOCS = {}
    for topic in TOPIC_NAMES:
        topic_descriptions = df[df["topic"] == topic]["description"].tolist()
        TOPIC_DOCS[topic] = " ".join(topic_descriptions)

    while True:
        try:
            text = input("\n > ").strip()
        except Exception:
            break

        if not text or text.lower() in ("exit", "quit", "q"):
            break
    
        pred = classify(text, TOPIC_DOCS, TOPIC_NAMES)
        for method in ["TF-IDF", "BoW"]:
            p = pred[method]
            print(f"\n{method}: {p['predicted']} (confidence={p['confidence']:.4f})")
            for t in TOPIC_NAMES:
                s = p["scores"][t]
                bar = "=" * int(s * 10)
                print(f"{t:<22} {s:.4f} {bar}")
            print(f"Consensus: {pred['consensus']}")


def build_vectorizers(topic_docs):
    corpus = list(topic_docs.values())

    tfidf_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1,2), min_df=1, sublinear_tf=True)
    tfidf_matrix = tfidf_vec.fit_transform(corpus)

    bow_vec = CountVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1)
    bow_matrix = bow_vec.fit_transform(corpus)

    return tfidf_vec, tfidf_matrix, bow_vec, bow_matrix


def classify(text: str, topic_docs, topic_names):
    results = {}
    tfidf_vec, tfidf_matrix, bow_vec, bow_matrix = build_vectorizers(topic_docs)

    for method, vec, matrix in [("TF-IDF", tfidf_vec, tfidf_matrix), ("BoW", bow_vec, bow_matrix)]:
        q = vec.transform([text])
        sims = cosine_similarity(q, matrix)[0]
        scores = {t: float(round(s, 4)) for t, s in zip(topic_names, sims)}
        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]

        THRESHOLD = 0.05
        UNKNOWN_LABEL = "невідомо / unknown"

        label = best_topic if best_score >= THRESHOLD else UNKNOWN_LABEL
        results[method] = {
            "scores": scores,
            "predicted": label,
            "confidence": best_score
        }

    pred_tfidf = results["TF-IDF"]["predicted"]
    pred_bow = results["BoW"]["predicted"]
    if pred_tfidf == pred_bow:
        consensus = pred_tfidf
    elif pred_tfidf == UNKNOWN_LABEL or pred_bow == UNKNOWN_LABEL:
        consensus = UNKNOWN_LABEL
    else:
        if results["TF-IDF"]["confidence"] >= results["BoW"]["confidence"]:
            consensus = pred_tfidf
        else:
            consensus = pred_bow
    
    results["consensus"] = consensus
    return results



if __name__ == "__main__":
    main()