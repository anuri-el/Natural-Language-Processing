import os
import re
import requests
import json
from dotenv import load_dotenv
from collections import Counter

SEP = "=" * 67

load_dotenv()
API_KEY = os.environ.get("NEWS_API_KEY")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILENAME = "./outputs/l4_articles.csv"


def main():
    articles = load_or_save_articles(CSV_FILENAME)
    print(f"Loaded {len(articles)} articles from {CSV_FILENAME}")

    dist = Counter(article["label"] for article in articles)
    for lbl, cnt in dist.items():
        print(f"{lbl:<10} - {cnt:3d}")
    

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


if __name__ == "__main__":
    main()