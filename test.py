import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("NEWS_API_KEY")

resp = requests.get(
    "https://newsapi.org/v2/top-headlines/sources",
    params={"apiKey": API_KEY}
)

if resp.status_code == 200:
    data = resp.json()
    print(f"Status: {data.get('status')}")
    print(f"Total sources: {len(data.get('sources', []))}\n")
    
    print("Available sources:")
    for source in data.get("sources", []):
        if source['category'] == "general" and source['country'] == "us":
            print(f"  {source['id']}: {source['name']} ({source['category']}, {source['country']})")
else:
    print(f"Error {resp.status_code}: {resp.text}")