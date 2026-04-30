import os
import re
import requests
import json
import spacy
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

SEP = "=" * 67

load_dotenv()
API_KEY = os.environ.get("NEWS_API_KEY")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILENAME = "./outputs/l4_articles.csv"

DOMAIN_LABELS  = ["economic", "political", "social", "technology", "sports"]
N_CLUSTERS = 5
RANDOM_STATE = 42

nlp = spacy.load("en_core_web_lg")

def main():
    articles = load_or_save_articles(CSV_FILENAME)
    print(f"Loaded {len(articles)} articles from {CSV_FILENAME}")

    dist = Counter(article["label"] for article in articles)
    for lbl, cnt in dist.items():
        print(f"{lbl:<10} - {cnt:3d}")

    print(f"\n{SEP}")
    print("level 1")

    ctx = supervised_cluster(articles)
    print(f"Silhouette score: {ctx["sil"]:.4f}")
    print(f"Adjusted Rand Index: {ctx["ari"]:.4f}")
    print(f"Mapping accuracy: {ctx["accuracy"]:.4f}")

    print("Per-class accuracy:")
    for i, dom in enumerate(DOMAIN_LABELS):
        total = sum(1 for t in ctx["labels"] if t == i)
        correct_i = sum(1 for p, t in zip(ctx["preds"], ctx["labels"]) if t==i and p==i)
        pct = correct_i/total * 100 if total else 0
        bar = "=" * int(pct/10)
        print(f"{dom:<14} {pct:5.1f}% ({correct_i}/{total}) {bar}")

    cm = confusion_matrix(ctx["labels"], ctx["preds"], labels=list(range(N_CLUSTERS)))
    plot_confusion(cm, "./outputs/l4_cm.png")

    print("\nTop terms per cluster")
    print(ctx["top_terms"])
    

def scrape_newsapi(api_key: str, per_category: int = 15):
    articles = []
    category_map = {
        "economic": "business",
        "social": "health",
        "technology": "technology",
        "sports": "sports",
        "political": "general",
    }

    for label, cat in category_map.items():
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": api_key,
            "category": cat,
            "language": "en",
            "pageSize": per_category,
            "country": "us",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("status") != "ok":
                print(f"[{cat}]: {data.get('message','err')}")
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

                articles.append({"label": label, "text": text})

            print(f"{label:<10} - {len(data.get('articles',[]))} articles")
        
        except Exception as e:
            print(f"[{label}]: {e}")

    return articles


def load_or_save_articles(articles_file):
    
    if os.path.exists(articles_file):
        try:
            with open(articles_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            return articles
        except Exception as e:
            print(f"Could not load from {articles_file}: {e}")
            articles = None
    else:
        articles = None
    
    if articles is None:
        articles = scrape_newsapi(API_KEY, per_category=30)
        
        try:
            with open(articles_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(articles)} articles to {articles_file}")
        
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

def remove_stopwords(tokens):
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 2]


def preprocess(text: str):
    text = filter_text(text)
    text = normalize_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


SEED_WORDS = {
    "economic": ["economy", "finance", "stock", "investment", "gdp", "inflation", "interest", "rate"],
    "political": ["government", "election", "parliament", "senate", "congress", "vote", "democracy", "law"],
    "social": ["health", "education", "poverty", "welfare", "housing", "climate", "environment"],
    "technology": ["technology", "artificial", "intelligence", "software", "computer", "internet", "cyber"],
    "sports": ["football", "soccer", "basketball", "tennis", "olympic", "championship", "league", "player"],
}


def expand_keywords(seed_words, nlp, top_n=10):
    all_keywords = set(seed_words.copy())

    for word in seed_words:
        token = nlp(word)
        if token.has_vector:
            similar = token.vocab.vectors.most_similar(token.vector.reshape(1, -1), n=top_n)

            for i, key, in enumerate(similar[0][0]):
                word_text = token.vocab.strings[key]
                if len(word_text) > 2 and not word_text.isdigit():
                    all_keywords.add(word_text.lower())
    
    return " ".join(all_keywords)


FRAMEWORK = {}
for category, seeds in SEED_WORDS.items():
    expanded = expand_keywords(seeds, nlp, top_n=10)
    FRAMEWORK[category] = expanded


def supervised_cluster(article_items: list[dict]):
    text_raw = [item["text"] for item in article_items]
    labels_str = [item["label"] for item in article_items]
    texts_pp = [preprocess(t) for t in text_raw]
    label_ids = [DOMAIN_LABELS.index(l) if l in DOMAIN_LABELS else 0 for l in labels_str]

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=1, sublinear_tf=True)
    X = tfidf.fit_transform(texts_pp)

    seed_texts = [preprocess(FRAMEWORK[d]) for d in DOMAIN_LABELS]
    seed_vecs = tfidf.transform(seed_texts)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, max_iter=300, random_state=RANDOM_STATE)
    km.fit(X)

    cluster_map = {}
    for c in range(N_CLUSTERS):
        sims = cosine_similarity(km.cluster_centers_[c:c+1], seed_vecs)[0]
        cluster_map[c] = int(np.argmax(sims))
    
    preds_raw = km.predict(X)
    preds_mapped = np.array([cluster_map[p] for p in preds_raw])

    sil = silhouette_score(X, preds_raw, metric="cosine")
    ari = adjusted_rand_score(label_ids, preds_mapped)

    correct = sum(p == t for p, t in zip(preds_mapped, label_ids))
    acc = correct / len(label_ids) * 100

    feat_names = tfidf.get_feature_names_out()
    top_terms  = {}
    for c in range(N_CLUSTERS):
        dom = DOMAIN_LABELS[cluster_map[c]]
        order = km.cluster_centers_[c].argsort()[::-1][:8]
        terms = [feat_names[i] for i in order]
        top_terms[dom] = terms

    return {"X": X, "tfidf": tfidf, "labels": label_ids, "cluster_map": cluster_map, "preds": preds_mapped, "texts_pp": texts_pp, "texts_raw": text_raw, "accuracy": acc, "sil": sil, "ari" : ari, "top_terms": top_terms}


def plot_confusion(cm, output_path):
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(N_CLUSTERS))
    ax.set_yticks(range(N_CLUSTERS))
    ax.set_xticklabels(DOMAIN_LABELS, ha="right")
    ax.set_xticklabels(DOMAIN_LABELS)

    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            ax.text(j, i, str(cm[i, j]), color="white" if cm[i, j] > cm.max() / 2 else "black")

    title = "Confusion Matrix"
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    main()