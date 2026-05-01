import os
import re
import requests
import json
import math
import spacy
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage

SEP = "=" * 67

load_dotenv()
API_KEY = os.environ.get("NEWS_API_KEY")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILENAME = "./outputs/l4_articles.csv"

DOMAIN_LABELS  = ["economic", "political", "social", "technology", "sports"]
DOMAIN_COLORS  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
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

    sup_cl = supervised_cluster(articles)
    print(f"Silhouette score: {sup_cl["silhouette"]:.4f}")
    print(f"Adjusted Rand Index: {sup_cl["ARI"]:.4f}")
    print(f"Mapping accuracy: {sup_cl["accuracy"]:.4f}")

    print("Per-class accuracy:")
    for i, dom in enumerate(DOMAIN_LABELS):
        total = sum(1 for t in sup_cl["labels"] if t == i)
        correct_i = sum(1 for p, t in zip(sup_cl["preds"], sup_cl["labels"]) if t==i and p==i)
        pct = correct_i/total * 100 if total else 0
        bar = "=" * int(pct/10)
        print(f"{dom:<14} {pct:5.1f}% ({correct_i}/{total}) {bar}")

    cm = confusion_matrix(sup_cl["labels"], sup_cl["preds"], labels=list(range(N_CLUSTERS)))
    plot_confusion_matrix(cm, "l4_cm.png")

    print("\nTop terms per cluster")
    print(sup_cl["top_terms"])
    


    print(f"\n{SEP}")
    print("level 2")

    freq = frequency_analysis(sup_cl)
    print("\nTF-IDF top terms:")
    print(freq["tfidf_by_category"])

    plot_tfidf_bars(freq["tfidf_by_category"], "l4_tfidf.png")

    for w, positions in freq["dispersion_data"].items():
        print(f" {w:<10} - {len(positions):4d} positions")

    plot_lexical_dispersion(freq["dispersion_data"], freq["top_terms"], len(sup_cl["article_items"]), "l4_lexical_dispersion.png")

    plot_word_length_dist(freq["lengths"], "l4_word_length_distribution.png")
    
    top_bigrams = freq["bigrams"].most_common(20)
    for bg, cnt in top_bigrams[:10]:
        print(f" {' '.join(bg):<28} {cnt:4d}")

    plot_bigrams(top_bigrams, "l4_bigrams.png")


    print(f"\n{SEP}")
    print("level 3")

    results = unsupervised_cluster(sup_cl)

    print("KMeans")
    print(f"Silhouette: {results["KMeans"]["silhouette"]:.4f}   ARI: {results["KMeans"]["ARI"]:.4f}")
    
    print("Agglomerative")
    print(f"Silhouette: {results["Agglomerative"]["silhouette"]:.4f}   ARI: {results["Agglomerative"]["ARI"]:.4f}")

    plot_dendrogram(results["Agglomerative"]["Z"], "l4_dendrogram.png")
    
    print("DBSCAN")
    print(f"Silhouette: {results["DBSCAN"]["silhouette"]:.4f}   ARI: {results["DBSCAN"]["ARI"]:.4f}")


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
CUSTOM_STOP_WORDS = {"cnn", "char", "bbc", "news", "latest", "com", "new", "york", "times", "post"}

def remove_stopwords(tokens):
    stop_words = STOP_WORDS.copy()
    stop_words.update(CUSTOM_STOP_WORDS)
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

    return {"X": X, 
            "tfidf": tfidf, 
            "labels": label_ids, 
            "cluster_map": cluster_map, 
            "preds": preds_mapped, 
            "texts_pp": texts_pp, 
            "texts_raw": text_raw, 
            "accuracy": acc, 
            "silhouette": sil, 
            "ARI" : ari, 
            "top_terms": top_terms,
            "article_items": article_items}


def frequency_analysis(sup_cl):
    all_tokens = []
    for item in sup_cl["article_items"]:
        all_tokens.extend(get_tokens(item["text"]))
    
    by_cat = defaultdict(list)
    for item in sup_cl["article_items"]:
        by_cat[item["label"]].append(preprocess(item["text"]))
 
    cat_tfidf_results = {}
    for cat, docs in by_cat.items():
        tv = TfidfVectorizer(max_features=500, ngram_range=(1,1), min_df=1)
        M = tv.fit_transform(docs)
        scores = np.array(M.sum(axis=0)).flatten()
        order = scores.argsort()[::-1][:15]
        terms = [(tv.get_feature_names_out()[i], float(scores[i])) for i in order]
        cat_tfidf_results[cat] = terms
        print(f" {cat:<14}: {', '.join(t for t,_ in terms[:8])}")


    freq = Counter(all_tokens)
    top8 = [w for w, _ in freq.most_common(8)]
    disp_data = {w: [] for w in top8}
    for i, item in enumerate(sup_cl["article_items"]):
        toks = get_tokens(item["text"])
        for w in top8:
            disp_data[w].extend([i + pos/max(len(toks),1) for pos, t in enumerate(toks) if t == w])
    

    lengths = [len(w) for w in all_tokens]
    length_freq = Counter(lengths)
    for l in sorted(length_freq)[:12]:
        bar = "=" * int(length_freq[l] / max(length_freq.values()) * 10)
        print(f"{l:2d} letters : {length_freq[l]} {bar}")


    bigrams = Counter()
    for item in sup_cl["article_items"]:
        toks = get_tokens(item["text"])
        for i in range(len(toks)-1):
            bigrams[(toks[i], toks[i+1])] += 1
    

    return {
        "tfidf_by_category": cat_tfidf_results,
        "dispersion_data" : disp_data,
        "top_terms" : top8,
        "lengths": lengths,
        "word_length_distribution": dict(sorted(length_freq.items())),
        "bigrams": bigrams,
    }


def unsupervised_cluster(sup_cl: dict):
    X = sup_cl["X"]
    y_true = sup_cl["labels"]
    tfidf = sup_cl["tfidf"]
    texts = sup_cl["texts_pp"]

    km_u = KMeans(n_clusters=N_CLUSTERS, n_init=15, max_iter=500, random_state=RANDOM_STATE)
    km_u.fit(X)
    km_labels = km_u.labels_
    sil_km = silhouette_score(X, km_labels, metric="cosine")
    ari_km = adjusted_rand_score(y_true, km_labels)

    fn = tfidf.get_feature_names_out()
    km_cluster_terms = {}
    for c in range(N_CLUSTERS):
        order = km_u.cluster_centers_[c].argsort()[::-1][:6]
        terms = [fn[i] for i in order]
        km_cluster_terms[c] = terms


    svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
    X_r = svd.fit_transform(X)
    X_rn = normalize(X_r)

    agg = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
    agg_labels = agg.fit_predict(X_rn)
    sil_agg = silhouette_score(X_rn, agg_labels)
    ari_agg = adjusted_rand_score(y_true, agg_labels)

    sample_idx = np.random.choice(len(texts), min(40, len(texts)), replace=False)
    Z = linkage(X_rn[sample_idx], method="ward")


    db = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
    db_labels = db.fit_predict(X_r)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    if n_clusters_db > 1:
        sil_db = silhouette_score(X_rn, db_labels)
    else:
        sil_db = float("nan")
    
    ari_db = adjusted_rand_score(y_true, db_labels)



    results = {
        "KMeans": {"silhouette": sil_km,  "ARI": ari_km, "km_cluster_terms": km_cluster_terms},
        "Agglomerative": {"silhouette": sil_agg,  "ARI": ari_agg, "Z": Z},
        "DBSCAN": {"silhouette": None if math.isnan(sil_db) else sil_db, "ARI": ari_db, "n_clusters": n_clusters_db, "n_noise": n_noise},
    }

    return results


def plot_confusion_matrix(cm, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(N_CLUSTERS))
    ax.set_yticks(range(N_CLUSTERS))
    ax.set_xticklabels(DOMAIN_LABELS)
    ax.set_yticklabels(DOMAIN_LABELS)

    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            ax.text(j, i, str(cm[i, j]), color="white" if cm[i, j] > cm.max() / 2 else "black")

    title = "Supervised KMeans - Confusion Matrix"
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"{title} saved to: {output_path}")


def plot_tfidf_bars(cat_results: dict, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, axes = plt.subplots(1, len(cat_results), figsize=(4*len(cat_results), 5), sharey=False)
    
    if len(cat_results) == 1:
        axes = [axes]
    
    for ax, (cat, terms) in zip(axes, cat_results.items()):
        ws = [t for t, _ in terms[:10]]
        sc = [s for _, s in terms[:10]]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(ws)))
        
        ax.barh(list(reversed(ws)), list(reversed(sc)), color=list(reversed(colors)))
        ax.set_title(cat, fontsize=10)
        ax.set_xlabel("TF-IDF score")
    
    title = "TF-IDF top-10 by category"
    fig.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"{title} saved to: {output_path}")


def plot_lexical_dispersion(disp_data: dict, words: list, n_docs: int, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, w in enumerate(words):
        positions = disp_data[w]
        ax.scatter(positions, [idx]*len(positions), marker="|", s=50, color=DOMAIN_COLORS[idx % len(DOMAIN_COLORS)], alpha=0.7)
    
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.set_xlabel("Position", fontsize=11)
    
    title = "Lexical Dispersion"
    ax.set_title(title)
    ax.set_xlim(0, n_docs)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"{title} saved to: {output_path}")


def plot_word_length_dist(lengths: list, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lc = Counter(lengths)
    xs = sorted(lc.keys())
    ys = [lc[x] for x in xs]
 
    axes[0].bar(xs, ys, color="#42A5F5", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Word length")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Word length distribution")
 
    total = sum(ys)
    pmf = [y/total for y in ys]
    axes[1].plot(xs, pmf, "o-", color="#EF5350", linewidth=2, markersize=6)
    axes[1].fill_between(xs, pmf, alpha=0.15, color="#EF5350")
    axes[1].set_xlabel("Word length", fontsize=11)
    axes[1].set_ylabel("PMF", fontsize=11)
    axes[1].set_title("Probability Mass Function")
  
    title = "Word length distibution"
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"{title} saved to: {output_path}")


def plot_bigrams(top_bigrams: list, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    labels = [" ".join(bg) for bg, _ in top_bigrams[:15]]
    counts = [c for _, c in top_bigrams[:15]]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(reversed(labels)), list(reversed(counts)), color=list(reversed(colors)))
    ax.set_xlabel("Frequency")
    
    title = "Top-15 bigrams"
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"{title} saved to: {output_path}")


def plot_dendrogram(Z, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=7, color_threshold=0.7*max(Z[:,2]))

    title = "Dendrogram (Agglomerative, Ward linkage)"
    ax.set_title(title)
    ax.set_xlabel("Документ (індекс)")
    ax.set_ylabel("Відстань")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"{title} saved to: {output_path}")


if __name__ == "__main__":
    main()