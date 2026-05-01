from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import nltk
import os
import csv
import re
import json
import requests

for pkg in ['punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)

SEP = "=" * 67

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()
API_KEY = os.environ.get("NEWS_API_KEY")


SOURCES = {
    "bbc-news": "BBC News",
    "cnn": "CNN",
    "associated-press": "Associated Press",
    "cbs-news": "CBS News",
    "al-jazeera-english": "Al Jazeera English",
}

DAYS_BACK = 14
TODAY = datetime.now().date()
TOP_N = 10

RAW_DATA = "./outputs/l2_raw_articles.json"

def main():
    print(f"\n{SEP}")
    print("Level 2")
    articles = load_or_save_articles(RAW_DATA)

    print(f"\n{SEP}")
    print("Filtering")
    filtered = filter_articles(articles)
    headers = list(filtered[0].keys())
    print(filtered[0]["filtered"])
    save_csv(filtered, headers, "l2_filtered_articles.csv")

    print(f"\n{SEP}")
    print("Normalization")
    normalized = normalize_articles(filtered)
    headers = list(normalized[0].keys())
    print(normalized[0]["normalized"])
    save_csv(normalized, headers, "l2_normalized_articles.csv")

    print(f"\n{SEP}")
    print("Tokenization")
    tokenized = tokenize_article(normalized)
    headers = list(tokenized[0].keys())
    print("tokens_word:\n", tokenized[0]["tokens_word"])
    print("tokens_sentence:\n", tokenized[0]["tokens_sentence"])
    print("tokens_tweet\n", tokenized[0]["tokens_tweet"])
    save_csv(tokenized, headers, "l2_tokenized_articles.csv")

    print(f"\n{SEP}")
    print("Removing stop words")
    no_stopwords = apply_stopwords(tokenized)
    headers = list(no_stopwords[0].keys())
    print(no_stopwords[0]["tokens_clean"])
    save_csv(no_stopwords, headers, "l2_no_stopwords_articles.csv")
    
    print(f"\n{SEP}")
    print("Lemmatization")
    lemmatized = lemmatize_articles(no_stopwords)
    headers = list(lemmatized[0].keys())
    print(lemmatized[0]["lemmas"])
    save_csv(lemmatized, headers, "l2_lemmatized_articles.csv")
    
    print(f"\n{SEP}")
    print("Stemming")
    stemmed = stem_articles(lemmatized)
    headers = list(stemmed[0].keys())
    save_csv(stemmed, headers, "l2_stemmed_articles.csv")

    print("stems_porter\n", stemmed[0]["stems_porter"])
    print("stems_snowball\n", stemmed[0]["stems_snowball"])

    print(f"\n{SEP}")
    print("Top 10")
    top_data = compute_top_words(stemmed)
    save_json(top_data, "l2_top_words.csv")

    print(f"\n{SEP}")
    print("global_top10:")
    for rank, (w, c) in enumerate(top_data["global_top10"], 1):
        print(f"  {rank:2}. {w:<15} {c}")

    print(f"\n{SEP}")
    print("by_source:")
    for src, words in top_data["by_source"].items():
        print(f"\n{src}")
        for rank, (w, c) in enumerate(words, 1):
            print(f"  {rank:2}. {w:<15} {c}")

    print(f"\n{SEP}")
    print("by_week:")
    for wk in ["week_1", "week_2"]:
        print(f"\n{wk}")
        for rank, (w, c) in enumerate(top_data["by_week"].get(wk, []), 1):
            print(f"  {rank:2}. {w:<15} {c}")

    w1_words = {w for w, _ in top_data["by_week"].get("week_1", [])}
    w2_words = {w for w, _ in top_data["by_week"].get("week_2", [])}
    
    new_in_1 = w1_words - w2_words
    
    print("\nnew in week 1:")
    print(new_in_1)

    print(f"\n{SEP}")
    plot_top10(top_data, "l2_top10.png")
    plot_wk1_vs_wk2(top_data, "l2_wk1_vs_wk2.png")
    plot_articles_by_source(articles, "l2_articles_by_source.png")
    plot_articles_by_date(articles, "l2_articles_by_date.png")


def parse_date(entry):
    for attr in ('published_parsed', 'updated_parsed'):
        t = getattr(entry, attr, None)
        if t:
            return datetime(*t[:6]).date()
    return TODAY         


def fetch_source(source_id, source_label, from_date, to_date):
    articles = []
    page = 1
    while True:
        params = {
            "apiKey":   API_KEY,
            "sources":  source_id,
            "from":     from_date.isoformat(),
            "to":       to_date.isoformat(),
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 100,
            "page":     page,
        }
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        data = resp.json()
 
        if data.get("status") != "ok":
            print(f"[err] {source_label}: {data.get('message', 'unknown')}")
            break
 
        batch = data.get("articles", [])
        if not batch:
            break
 
        for art in batch:
            pub = art.get("publishedAt", "")[:10]
            title = art.get("title") or ""
            desc = art.get("description") or ""
            content = art.get("content") or ""
            articles.append({
                "source": source_label,
                "date": pub,
                "title": title,
                "summary": desc,
                "text": f"{title} {desc} {content}",
            })
 
        total = data.get("totalResults", 0)
        retrieved = page * 100
        print(f" {source_label}: page {page}, retrieved {min(retrieved, total)}/{total}")
 
        if retrieved >= total or retrieved >= 1000:
            break
        page += 1
 
    return articles


def scrape_all_sources():
    to_date = TODAY
    from_date = TODAY - timedelta(days=DAYS_BACK)
    print(f"Range: {from_date} - {to_date}")
 
    all_articles = []
    for src_id, src_label in SOURCES.items():
        print(f"\n{src_label}  (id: {src_id})")
        arts = fetch_source(src_id, src_label, from_date, to_date)
        print(f"Articles count: {len(arts)}")
        all_articles.extend(arts)
 
    print(f"\nAll articles: {len(all_articles)}")
    return all_articles


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
        articles = scrape_all_sources()
        
        try:
            with open(articles_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(articles)} articles to {articles_file}")
        
        except Exception as e:
            print(f"Could not save to {articles_file}: {e}")

        return articles


def filter_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_articles(articles):
    filtered = []
    for a in articles:
        fc = filter_text(a["text"])
        filtered.append({**a, "filtered": fc})
    return filtered


def normalize_text(text: str):
    return text.lower().strip()


def normalize_articles(articles):
    normalized = []
    for a in articles:
        n = normalize_text(a["filtered"])
        normalized.append({**a, "normalized": n})
    return normalized


tweet_tok = TweetTokenizer(preserve_case=False, strip_handles=True)

def tokenize_article(articles):
    for a in articles:
        text = a["normalized"]
        a["tokens_word"] = word_tokenize(text)
        a["tokens_sentence"] = sent_tokenize(a["filtered"])
        a["tokens_tweet"] = tweet_tok.tokenize(text)
    
    return articles


STOP_WORDS = set(stopwords.words("english"))
CUSTOM_STOP_WORDS = {"cnn", "char", "chars", "bbc", "news", "latest", "say", "said", "cbs"}

def remove_stopwords(tokens):
    stop_words = STOP_WORDS.copy()
    stop_words.update(CUSTOM_STOP_WORDS)
    return [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]


def apply_stopwords(articles):
    for a in articles:
        a["tokens_clean"] = remove_stopwords(a["tokens_word"])
    return articles


lemmatizer = WordNetLemmatizer()

def lemmatize_articles(articles):
    for a in articles:
        a["lemmas"] = [lemmatizer.lemmatize(t) for t in a["tokens_clean"]]
    return articles


porter = PorterStemmer()
snowball = SnowballStemmer("english")

def stem_articles(articles):
    for a in articles:
        a["stems_porter"] = [porter.stem(t) for t in a["tokens_clean"]]
        a["stems_snowball"] = [snowball.stem(t) for t in a["tokens_clean"]]
    return articles


def get_week_label(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    delta = (TODAY - d).days
    return "week_1" if delta <= 7 else "week_2"


def compute_top_words(articles):
    global_counter = Counter()
    for a in articles:
        global_counter.update(a["lemmas"])
    top10_global = global_counter.most_common(TOP_N)

    by_source = defaultdict(Counter)
    for a in articles:
        by_source[a["source"]].update(a["lemmas"])
    top_by_source = {s: c.most_common(TOP_N) for s, c in by_source.items()}

    by_week = defaultdict(Counter)
    for a in articles:
        wk = get_week_label(a["date"])
        by_week[wk].update(a["lemmas"])
    top_by_week = {w: c.most_common(TOP_N) for w, c in by_week.items()}

    result = {
        "global_top10": top10_global,
        "by_source": top_by_source,
        "by_week": top_by_week,
        "analysis_period": f"{TODAY - timedelta(DAYS_BACK)} - {TODAY}",
    }
    
    return result


def plot_top10(top_data, filename):
    words, counts = zip(*top_data["global_top10"])
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(reversed(words)), list(reversed(counts)))

    for bar, cnt in zip(bars, list(reversed(counts))):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, str(cnt), va='center', fontsize=10)
    
    title = "Top-10 words"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("frequency", fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.show()
    print(f"{title} saved to: {path}")


def plot_wk1_vs_wk2(top_data, filename):
    w1 = dict(top_data["by_week"].get("week_1", []))
    w2 = dict(top_data["by_week"].get("week_2", []))
    all_keys = sorted(set(list(w1.keys())[:8] + list(w2.keys())[:8]))[:12]
    x = np.arange(len(all_keys))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width/2, [w1.get(k, 0) for k in all_keys], width, label="current week", alpha=0.85)
    ax.bar(x + width/2, [w2.get(k, 0) for k in all_keys], width, label="previous week", alpha=0.85)
    
    title = "Current vs Previous Week"
    ax.set_title(title)
    ax.set_xticklabels(all_keys)
    ax.set_xticks(x)
    ax.set_ylabel("Frequency")
    ax.legend()
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.show()
    print(f"{title} saved to: {path}")


def plot_articles_by_source(articles, filename):
    src_counts = Counter(a["source"] for a in articles)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(src_counts.values(), labels=src_counts.keys(), autopct="%1.1f%%", startangle=140)
    
    title = "Articles by source"
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.show()
    print(f"{title} saved to: {path}")


def plot_articles_by_date(articles, filename):
    date_counts = Counter(a["date"] for a in articles)
    dates  = sorted(date_counts.keys())
    values = [date_counts[d] for d in dates]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(dates, values, alpha=0.4)
    ax.plot(dates, values, "o-", linewidth=2, markersize=5)

    title = "Articles by date"
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Amount")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.show()
    print(f"{title} saved to: {path}")


def save_csv(rows, headers, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{filename} saved to: {path}")


def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"{filename} saved to: {path}")


if __name__ == "__main__":
    main()