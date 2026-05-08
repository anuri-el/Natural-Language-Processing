import os
import re
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import nltk
from dotenv import load_dotenv
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, opinion_lexicon, wordnet, brown
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import TruncatedSVD

from NumpyMLP import NumpyMLP



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


DOMAIN_LABELS  = ["economic", "political", "social", "technology", "sports"]
RANDOM_STATE = 42


def main():
    articles = load_or_save_articles(CSV_FILENAME)

    print(f"{SEP}")
    print("Level 1")
    tfidf = build_vectorizer(articles)
    print(f"TF-IDF vocab: {len(tfidf.vocabulary_)} terms")

    dom_data, sent_data = build_training_data(tfidf)
    print(f"  Domain corpus: {dom_data[0].shape}")
    print(f"  Sentiment corpus: {sent_data[0].shape}")



    print(f"\n{SEP}")
    r = comparative_analysis(articles, tfidf, dom_data, sent_data)
    print("Sklearn MLP for domain classification")
    print(f"    Точність на корпусі: {r['dom_train_acc']:.4f}")

    print("Custom NumpyMLP for domain")
    print(f"    Val accuracy (numpy MLP): {r['np_dom_val_acc']:.4f}")


    print("Predict domains for all articles")


    print("  Матриця схожості між джерелами (attention-based cosine):")
    print(f"  {'':20}", end="")
    src_names = list(SOURCES.values())
    for s in src_names: 
        print(f"{s[:12]:>14}", end="")
    print()
    for i,s in enumerate(src_names):
        print(f"  {s:<20}", end="")
        for j in range(len(src_names)): 
            print(f"{r['sim_matrix'][i,j]:>14.4f}", end="")
        print()   



def scrape_newsapi(api_key: str, page_size: int = 50):
    result = {}

    for src_id, name in SOURCES.items():
        articles = []
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": api_key,
            "sources": src_id,
            "language": "en",
            "pageSize": page_size,
            "sortBy": "publishedAt"
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
#     "today", "yesterday", "week", "month", "year", "first", "last", "daily", "post",
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


# def build_domain_corpus():
#     domain_mapping = {
#         'economic': ['news_editorial', 'reviews'],
#         'political': ['government', 'news_editorial'],
#         'social': ['hobbies', 'lore'],
#         'technology': ['science_fiction', 'science'],
#         'sports': ['adventure']
#     }
    
#     domain_corpus = defaultdict(list)
    
#     for domain, categories in domain_mapping.items():
#         for cat in categories:
#             if cat in brown.categories():
#                 words = brown.words(categories=cat)
#                 freq_dist = nltk.FreqDist(w.lower() for w in words)
#                 domain_corpus[domain].extend([w for w, _ in freq_dist.most_common(250)])
    
#     return dict(domain_corpus)


def get_related_words(words, pos_tag=wordnet.ADJ):
    related = set(words)
    for word in words:
        synsets = wordnet.synsets(word, pos=pos_tag)
        for synset in synsets:
            for lemma in synset.lemmas():
                related.add(lemma.name().replace("_", " "))
    return related


def build_sentiment_corpus():
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())

    return {
        "positive": list(get_related_words(positive_words))[:500],
        "negative": list(get_related_words(negative_words))[:500],
        "neutral": list(stopwords.words("english")),
    }



domain_corpus = {
    "economic": (
        "economy finance stock market investment gdp inflation interest rate "
        "trade deficit surplus budget revenue earnings profit loss fiscal monetary "
        "unemployment employment recession growth banking currency bond equity dividend "
        "venture capital merger acquisition ipo shareholder commodity oil price crude "
        "export import tariff subsidy federal reserve wall street bloomberg nasdaq dow "
        "consumer spending retail mortgage debt bankruptcy hedge fund quarter earnings "
        "revenue forecast outlook analyst investor portfolio asset liability balance sheet "
        "fed powell treasury secretary labor jobs payroll wages salary pension"
    ),
    "political": (
        "government election parliament senate congress vote democracy law policy "
        "president minister prime cabinet diplomat treaty international relations "
        "republican democrat liberal conservative party politician legislation bill "
        "amendment constitution court justice supreme ruling executive judicial "
        "campaign rally debate poll protest sanctions embassy foreign affairs trump "
        "biden ukraine russia china iran nato military army war conflict ceasefire "
        "diplomacy sanction embargo administration governor mayor white house pentagon "
        "department state secretary defense attorney general indictment impeach veto "
        "filibuster bipartisan legislation regulation executive order supreme court "
        "senate vote bill passed signed law reform immigration border tariff trade war"
    ),
    "social": (
        "health disease cancer study research hospital patient treatment drug medicine "
        "vaccine pandemic epidemic obesity diabetes heart stroke mental illness surgery "
        "doctor nurse healthcare education school university student poverty inequality "
        "welfare housing homelessness crime prison justice race gender climate "
        "environment pollution emissions disaster flood drought wildfire community "
        "family children elderly aging population birth death mortality life expectancy "
        "nutrition diet exercise fitness sleep stress anxiety depression"
    ),
    "technology": (
        "technology artificial intelligence software computer internet cyber security "
        "machine learning algorithm data cloud computing robot automation smartphone "
        "app platform startup silicon valley google apple microsoft amazon meta openai "
        "iphone android gpu chip semiconductor processor quantum blockchain crypto "
        "social media youtube twitter tiktok streaming video gaming hardware device "
        "electric vehicle battery solar energy innovation engineering programming code "
        "developer network bandwidth fiber broadband satellite space launch nasa spacex"
    ),
    "sports": (
        "football soccer basketball tennis olympic championship league player team coach "
        "match game score goal win loss draw tournament stadium athlete performance "
        "nfl nba nhl mlb fifa uefa transfer draft trade contract injury season playoffs "
        "final championship trophy medal record race marathon golf swimming cycling "
        "boxing wrestling mma combat sport referee umpire penalty foul stadium fans "
        "pitcher quarterback midfielder defender goalkeeper forward roster standings"
    ),
}


# domain_corpus = build_domain_corpus()
sentiment_corpus = build_sentiment_corpus()

def build_vectorizer(articles: dict):
    corpus_docs = [preprocess(t) for ts in domain_corpus.values() for t in ts]
    corpus_docs += [preprocess(t) for ts in sentiment_corpus.values() for t in ts]
    corpus_docs += [preprocess(a["text"]) for arts in articles.values() for a in arts]

    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf.fit(corpus_docs)

    return tfidf


def self_attention(X: np.array, d_k: int = 32):
    n, d = X.shape
    Wq = np.random.randn(d, d_k) * 0.1
    Wk = np.random.randn(d, d_k) * 0.1
    Wv = np.random.randn(d, d_k) * 0.1
    Q  = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = Q @ K.T / np.sqrt(d_k)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn /= attn.sum(axis=-1, keepdims=True)
    out = attn @ V
    return out, attn


def build_training_data(tfidf: TfidfVectorizer):
    dom_texts, dom_labels = [], []
    for label, docs in domain_corpus.items():
        for doc in docs:
            dom_texts.append(preprocess(doc))
            dom_labels.append(label)
    X_dom = tfidf.transform(dom_texts).toarray().astype(np.float32)

    sent_texts, sent_labels = [], []
    for label, docs in sentiment_corpus.items():
        for doc in docs:
            sent_texts.append(preprocess(doc))
            sent_labels.append(label)
    X_sent = tfidf.transform(sent_texts).toarray().astype(np.float32)

    return (X_dom, np.array(dom_labels)), (X_sent, np.array(sent_labels))


def comparative_analysis(articles: dict, tfidf: TfidfVectorizer, dom_data, sent_data):
    X_dom_tr, y_dom_tr = dom_data
    src_names = list(articles.keys())

    label_encoder = LabelEncoder()
    y_dom_tr = label_encoder.fit_transform(y_dom_tr)


    scaler_d = StandardScaler(with_mean=False)
    Xs_d = scaler_d.fit_transform(X_dom_tr)

    mlp_dom = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu", solver="adam",
        learning_rate_init=0.001, max_iter=300,
        alpha=1e-4, early_stopping=True, 
        validation_fraction=0.2,
        random_state=RANDOM_STATE, verbose=False,
    )
    mlp_dom.fit(Xs_d, y_dom_tr)
    dom_train_acc = accuracy_score(y_dom_tr, mlp_dom.predict(Xs_d))


    svd_d = TruncatedSVD(n_components=64, random_state=RANDOM_STATE)
    Xr_d = svd_d.fit_transform(X_dom_tr)
    np_mlp_dom = NumpyMLP(layer_sizes=[64, 128, 64, len(DOMAIN_LABELS)], lr=0.008, momentum=0.88, l2=1e-4, epochs=120, batch_size=4,)
    Xr_d_tr, Xr_d_val, yr_d_tr, yr_d_val = train_test_split(Xr_d, y_dom_tr, test_size=0.2, stratify=y_dom_tr, random_state=RANDOM_STATE)
    np_mlp_dom.fit(Xr_d_tr, yr_d_tr, Xr_d_val, yr_d_val)
    np_dom_val_acc = np_mlp_dom.history[-1].get("val_acc", 0)



    all_arts = [a for arts in articles.values() for a in arts]
    texts_pp = [preprocess(a["text"]) for a in all_arts]
    X_arts = tfidf.transform(texts_pp).toarray().astype(np.float32)
    Xs_arts = scaler_d.transform(X_arts)

    preds_dom = mlp_dom.predict(Xs_arts)
    for a, p in zip(all_arts, preds_dom):
        a["pred_domain"] = p

    for src in src_names:
        arts_src = [a for a in all_arts if a["source"] == src]
        dist = Counter(a["pred_domain"] for a in arts_src)
        print(f"    {src:<16}: {dict(dist)}")



    svd2 = TruncatedSVD(n_components=32, random_state=RANDOM_STATE)
    X_r32 = normalize(svd_d.fit_transform(X_arts))
    attn_out, attn_weights = self_attention(X_r32, d_k=16)

    src_vecs = {}
    for src in src_names:
        mask = np.array([a["source"] == src for a in all_arts])
        src_vecs[src] = attn_out[mask].mean(axis=0)

    sim_mat = np.zeros((len(src_names), len(src_names)))
    for i,s1 in enumerate(src_names):
        for j,s2 in enumerate(src_names):
            v1 = src_vecs[s1]; v2 = src_vecs[s2]
            sim_mat[i,j] = float(v1 @ v2 / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9))


    return {
        "mlp_dom": mlp_dom, 
        "np_mlp_dom": np_mlp_dom,
        "scaler_d": scaler_d, 
        "svd_d": svd_d,
        "sim_matrix": sim_mat, 
        "src_names": src_names,
        "all_arts": all_arts, 
        "texts_pp": texts_pp,
        "X_arts": X_arts, 
        "X_r32": X_r32, 
        "tfidf": tfidf,
        "dom_train_acc": dom_train_acc,
        "np_dom_val_acc": np_dom_val_acc,
    }






if __name__ == "__main__":
    main()