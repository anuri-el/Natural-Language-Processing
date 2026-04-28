from datetime import datetime
from bs4 import BeautifulSoup
import feedparser
import os
import csv
import re


SEP = "=" * 67

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCES = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "The Guardian": "https://www.theguardian.com/world/rss",
}

DAYS_BACK = 14
TODAY = datetime.now().date()


def main():
    print(f"\n{SEP}")
    print("Level 2")
    articles = scrape_all_sources()

    print(f"\n{SEP}")
    print("Filtering")
    filtered = filter_articles(articles)
    headers = list(filtered[0].keys())
    save_csv(filtered, headers, "l2_filtered_articles.csv")

    print(f"\n{SEP}")
    print("Normalization")
    normalized = normalize_articles(filtered)
    headers = list(normalized[0].keys())
    save_csv(normalized, headers, "l3_normalized_articles.csv")



def parse_date(entry):
    for attr in ('published_parsed', 'updated_parsed'):
        t = getattr(entry, attr, None)
        if t:
            return datetime(*t[:6]).date()
    return TODAY         


def scrape_all_sources():
    all_articles = []
    for source, url in SOURCES.items():
        print(f"{source} ({url})")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                pub_date = parse_date(entry)

                if (TODAY - pub_date).days > DAYS_BACK:
                    continue

                title = entry.get("title", "")
                summary = entry.get("summary", "")

                summary = BeautifulSoup(summary, "html.parser").get_text()

                all_articles.append({
                    "source":  source,
                    "date":    str(pub_date),
                    "title":   title,
                    "summary": summary,
                    "text":    title + " " + summary,
                })

        except Exception as e:
            print(f"[err] {e}")
        
    print(f"Articles_count: {len(all_articles)}")

    if all_articles:
        headers = ["source", "date", "title", "summary", "text"]
        save_csv(all_articles, headers, "l2_raw_articles.csv")

    return all_articles


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


def save_csv(rows, headers, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{filename} saved to: {path}")


if __name__ == "__main__":
    main()