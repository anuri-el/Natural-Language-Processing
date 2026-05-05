import os
import re
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


SEP = "=" * 67

load_dotenv()
API_KEY = os.environ.get("NEWS_API_KEY")

SOURCES = {
    "bbc-news": "BBC News",
    "cnn": "CNN",
    "associated-press": "Associated Press",
    "cbs-news": "CBS News",
    "al-jazeera-english": "Al Jazeera English",
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILENAME = os.path.join(OUTPUT_DIR, "l5_articles.json")


def main():
    articles_by_souce = load_or_save_articles(CSV_FILENAME)
    print(f"{SEP}")
    print("Level 1")

    r = comparative_analysis(articles_by_souce)

    print("Cosine similarity between sources (TF-IDF):")
    print(f"  {'':20}", end="")
    for s in r["src_names"]:
        print(f"{s[:12]:>14}", end="")
    print()
    for i, s1 in enumerate(r["src_names"]):
        print(f"  {s1:<20}", end="")
        for j, _ in enumerate(r["src_names"]):
            print(f"{r['sim_matrix'][i,j]:>14.4f}", end="")
        print()

    print("Top 10 unique terms by source (TF-IDF):")
    for src in r["src_names"]:
        print(f"  {src:<16}: {', '.join(t for t,_ in r['top_terms'][src][:6])}")

    print("Dictionary cross-section between sources:")
    
    print(f"Common all: {len(set.intersection(*r['vocab_sets'].values()))}")
    n = len(r["src_names"])
    for i in range(n):
        for j in range(i+1, n):
            print(f"  {r['src_names'][i]} and {r['src_names'][j]}: {len(r['intersection'])} words")

    


    plot_sim_heatmap(r["sim_matrix"], r["src_names"], "l5_cosine_similarity_heatmap.png")
    plot_top_terms_per_source(r["top_terms"], "l5_top_terms_per_source.png")
    plot_lsa_scatter(r["X_lsa"], r["all_labels"], r["src_names"], "l5_lsa_document_scatter.png")
    


def scrape_newsapi(api_key: str, page_size: int = 30):
    result = {}

    for src_id, name in SOURCES.items():
        articles = []
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": api_key,
            "sources": src_id,
            "language": "en",
            "pageSize": page_size
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("status") != "ok":
                print(f"[{name}]: {data.get('message','err')}")
                continue
            
            for art in data.get("articles", []):
                text = " ".join(filter(None, [
                    art.get("title", ""),
                    art.get("description", ""),
                    art.get("content", ""),
                ]))

                text = re.sub(r"\[\+\d+ chars?\]", "", text).strip()
                if len(text.split()) < 8:
                    continue

                articles.append({
                    "title": art.get("title", ""),
                    "text": text,
                    "url": art.get("url", ""),
                    "date": (art.get("publishedAt", "") or "")[:10],
                    "source": name,
                })
            print(f"{name:<10} - {len(articles):3d} articles")
        
        except Exception as e:
            print(f"[{name}]: {e}")

        result[name] = articles

    return result


def load_or_save_articles(articles_file):
    if os.path.exists(articles_file):
        try:
            with open(articles_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"Loaded articles from {articles_file}")
            return articles
        except Exception as e:
            print(f"Could not load from {articles_file}: {e}")
            articles = None
    else:
        articles = None
    
    if articles is None:
        articles = scrape_newsapi(API_KEY)
        
        try:
            with open(articles_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"Saved articles to {articles_file}")
        
        except Exception as e:
            print(f"Could not save to {articles_file}: {e}")

        return articles


def filter_text(text: str):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str):
    return text.lower().strip()


def tokenize_text(text: str):
    return word_tokenize(text)


STOP_WORDS = set(stopwords.words("english"))
# CUSTOM_STOP_WORDS = {
#     "new", "say", "says", "news", "one", "com", "time", "may", "could", "latest", "across", "bbc",
#     "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "march", "april",
#     "today", "yesterday", "week", "month", "year", "first", "last","daily", "post",
#     "said", "also", "according", "report", "get", "like", "make",
# }

def remove_stopwords(tokens):
    stop_words = STOP_WORDS.copy()
    # stop_words.update(CUSTOM_STOP_WORDS)
    return [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]


lemmatizer = WordNetLemmatizer()

def lemmatize_text(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str):
    text = filter_text(text)
    text = normalize_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    lemmas = lemmatize_text(tokens)
    return " ".join(lemmas)


def get_tokens(text: str):
    text = filter_text(text)
    text = normalize_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    return tokens


def comparative_analysis(articles_by_source: dict):
    src_names = list(SOURCES.values())
    src_text = {s: " ".join(preprocess(a["text"]) for a in arts) for s, arts in articles_by_source.items()}
    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=1, sublinear_tf=True)
    corpus_vecs = tfidf.fit_transform(list(src_text.values()))

    n = len(src_names)
    sim_matrix = cosine_similarity(corpus_vecs)

    feat = tfidf.get_feature_names_out()
    src_top_terms = {}
    for i, src in enumerate(src_names):
        row = corpus_vecs[i].toarray().flatten()
        order = row.argsort()[::-1][:10]
        terms = [(feat[j], float(row[j])) for j in order]
        src_top_terms[src] = terms

    vocab_sets = {s: set(get_tokens(" ".join(a["text"] for a in arts))) for s, arts in articles_by_source.items()}
    
    for i in range(n):
        for j in range(i+1, n):
            inter = vocab_sets[src_names[i]] & vocab_sets[src_names[j]]

    stats = {}
    for src, arts in articles_by_source.items():
        all_tok = [t for a in arts for t in get_tokens(a["text"])]
        lens = [len(get_tokens(a["text"])) for a in arts]
        stats[src] = {
            "count": len(arts),
            "avg_len": round(sum(lens)/len(lens), 1) if lens else 0,
            "vocab_size": len(set(all_tok)),
            "total_tokens": len(all_tok),
        }

    all_docs = [preprocess(a["text"]) for arts in articles_by_source.values() for a in arts]
    all_labels = [src for src, arts in articles_by_source.items() for _ in arts]
    tfidf_all = TfidfVectorizer(max_features=3000, ngram_range=(1,1), min_df=1, sublinear_tf=True)
    X_all = tfidf_all.fit_transform(all_docs)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_lsa = normalize(svd.fit_transform(X_all))


    return {
        "sim_matrix": sim_matrix,
        "src_names": src_names,
        "top_terms": src_top_terms,
        "vocab_sets": vocab_sets,
        "intersection": inter,
        "stats": stats,
        "X_all": X_all, "X_lsa": X_lsa,
        "all_labels": all_labels, "all_docs": all_docs,
        "tfidf": tfidf_all,
    }





def plot_sim_heatmap(matrix, names, fname):
    path = os.path.join(OUTPUT_DIR, fname)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white" if matrix[i,j] > 0.6 else "black")
    ax.set_title("Cosine Similarity Heatmap (TF-IDF)")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_top_terms_per_source(src_top, fname):
    path = os.path.join(OUTPUT_DIR, fname)

    n = len(src_top)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    for ax, (src, terms) in zip(axes, src_top.items()):
        ws = [t for t,_ in terms[:8]]; sc = [s for _,s in terms[:8]]
        bars = ax.barh(list(reversed(ws)), list(reversed(sc)), alpha=0.82, edgecolor="white")
        ax.set_title(src)
        ax.set_xlabel("TF-IDF score")
        ax.tick_params(labelsize=9)
    
    fig.suptitle("Top Terms per Source")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_lsa_scatter(X_lsa, labels, src_names, fname):
    path = os.path.join(OUTPUT_DIR, fname)

    svd2 = TruncatedSVD(n_components=2, random_state=42)
    X2   = svd2.fit_transform(X_lsa)
    fig, ax = plt.subplots(figsize=(9, 7))
    for src in src_names:
        mask = np.array(labels) == src
        ax.scatter(X2[mask, 0], X2[mask, 1], s=60, alpha=0.75, label=src, edgecolors="white", linewidth=0.4)
    
    ax.set_title("LSA Document Scatter")
    ax.set_xlabel("LSA-1")
    ax.set_ylabel("LSA-2")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()