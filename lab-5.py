import os
import re
import requests
import json
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
CUSTOM_STOP_WORDS = {
    "new", "say", "says", "news", "one", "com", "time", "may", "could", "latest", "across", "bbc",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "march", "april",
    "today", "yesterday", "week", "month", "year", "first", "last","daily", "post",
    "said", "also", "according", "report", "get", "like", "make",
}

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


if __name__ == "__main__":
    main()