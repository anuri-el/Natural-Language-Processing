import os
import re
import csv
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp_en = spacy.load("en_core_web_lg")
nlp_uk = spacy.load("uk_core_news_lg")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_DATA = "./outputs/l3_topics.csv"
TEST_DATA = "./outputs/l3_test_data.csv"


def main():

    df = pd.read_csv(RAW_DATA)

    TOPIC_NAMES = df["topic"].unique().tolist()

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
            
        text = preprocess(text)
        pred = classify(text, TOPIC_DOCS, TOPIC_NAMES)
        for method in ["TF-IDF", "BoW"]:
            p = pred[method]
            print(f"\n{method}: {p['predicted']} (confidence={p['confidence']:.4f})")
            for t in TOPIC_NAMES:
                s = p["scores"][t]
                bar = "=" * int(s * 10)
                print(f"{t:<22} {s:.4f} {bar}")
            print(f"Consensus: {pred['consensus']}")
    
    results = []
    correct_tfidf = correct_bow = correct_consensus = 0

    test_samples = []
    with open(TEST_DATA, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            test_samples.append({
                "id": idx,
                "text": row["description"].strip(),
                "expected": row["topic"].strip()
            })
    
    n = len(test_samples)

    for sample in test_samples:
        text = preprocess(sample["text"])
        pred = classify(text, TOPIC_DOCS, TOPIC_NAMES)
        exp = sample["expected"]
        
        ok_tfidf = pred["TF-IDF"]["predicted"] == exp
        ok_bow = pred["BoW"]["predicted"] == exp
        ok_consensus = pred["consensus"] == exp
        
        correct_tfidf += ok_tfidf
        correct_bow += ok_bow
        correct_consensus += ok_consensus
        
        results.append({
            "id": sample["id"],
            "text": sample["text"][:80] + "…" if len(sample["text"]) > 80 else sample["text"],
            "expected": exp,
            "tfidf_pred": pred["TF-IDF"]["predicted"],
            "tfidf_confidence": pred["TF-IDF"]["confidence"],
            "tfidf_scores": pred["TF-IDF"]["scores"],
            "bow_pred": pred["BoW"]["predicted"],
            "bow_confidence": pred["BoW"]["confidence"],
            "bow_scores": pred["BoW"]["scores"],
            "consensus": pred["consensus"],
            "ok_tfidf": ok_tfidf,
            "ok_bow": ok_bow,
            "ok_consensus": ok_consensus,
        })

    accuracy = {
        "TF-IDF": round(correct_tfidf / n * 100, 1),
        "BoW": round(correct_bow / n * 100, 1),
        "Consensus": round(correct_consensus / n * 100, 1),
        "n": n,
    }

    output_path = "./outputs/l3_output.csv"
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            "id", "text", "expected", 
            "tfidf_pred", "tfidf_confidence", 
            "tfidf_радіоелектроніка", "tfidf_програмування", "tfidf_машинобудування",
            "bow_pred", "bow_confidence",
            "bow_радіоелектроніка", "bow_програмування", "bow_машинобудування",
            "consensus", "ok_tfidf", "ok_bow", "ok_consensus"
        ])
        
        for r in results:
            writer.writerow([
                r["id"], r["text"], r["expected"],
                r["tfidf_pred"], r["tfidf_confidence"],
                r["tfidf_scores"].get("радіоелектроніка", ""),
                r["tfidf_scores"].get("програмування", ""),
                r["tfidf_scores"].get("машинобудування", ""),
                r["bow_pred"], r["bow_confidence"],
                r["bow_scores"].get("радіоелектроніка", ""),
                r["bow_scores"].get("програмування", ""),
                r["bow_scores"].get("машинобудування", ""),
                r["consensus"], r["ok_tfidf"], r["ok_bow"], r["ok_consensus"]
            ])
    
    print(f"Results saved to: {output_path}")
    print(f"\nAccuracy Summary:")
    print(f"  TF-IDF: {accuracy['TF-IDF']:>10}%")
    print(f"  BoW: {accuracy['BoW']:>12}%")
    print(f"  Consensus: {accuracy['Consensus']:>6}%")
    print(f"  Total: {accuracy['n']:>8} samples")


def detect_language(text: str):
    cyrillic_chars = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
    latin_chars = sum(1 for char in text if char.isalpha() and (char < '\u0400' or char > '\u04FF'))
    
    if cyrillic_chars > latin_chars:
        return "uk"
    else:
        return "en"


def filter_text(text: str):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\u0400-\u04FF\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str):
    return text.lower().strip()


def preprocess(text: str, remove_stopwords: bool = True, lemmatize: bool = True):
    text = filter_text(text)
    text = normalize_text(text)

    if not text:
        return ""

    lang = detect_language(text)
    nlp = nlp_uk if lang == "uk" else nlp_en

    doc = nlp(text)
    
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space or len(token.text) <= 2:
            continue
        
        if remove_stopwords and token.is_stop:
            continue
        
        if lemmatize and token.lemma_:
            token_text = token.lemma_.lower()
        else:
            token_text = token.text.lower()
        
        tokens.append(token_text)
    
    return " ".join(tokens)


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
        UNKNOWN_LABEL = "unknown"

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