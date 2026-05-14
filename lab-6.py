import os
import re
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, opinion_lexicon, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from NumpyMLP import NumpyMLP
from NumpyAutoencoder import NumpyAutoencoder


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


DOMAIN_LABELS = ["economic", "political", "social", "technology", "sports"]
SENT_LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42


def main():
    articles = load_or_save_articles(CSV_FILENAME)

    print(f"{SEP}")
    tfidf = build_vectorizer(articles)
    print(f"TF-IDF vocab: {len(tfidf.vocabulary_)} terms")

    dom_data, sent_data = build_training_data(tfidf)
    print(f"Domain corpus: {dom_data[0].shape}")
    print(f"Sentiment corpus: {sent_data[0].shape}")


    print(f"\n{SEP}")
    print("Level 1")
    print("Sklearn MLP for domain classification")
    r = comparative_analysis(articles, tfidf, dom_data, sent_data)
    for src in r["src_names"]:
        arts_src = [a for a in r["all_arts"] if a["source"] == src]
        dist = Counter(a["pred_domain"] for a in arts_src)
        print(f"  {src:<16}: {dict(dist)}")

    print(f"  Точність на корпусі: {r['dom_train_acc']:.4f}")

    print(f"\n{SEP}")
    print("Custom NumpyMLP for domain")
    print(f" Val accuracy (numpy MLP): {r['np_dom_val_acc']:.4f}")


    print("Матриця схожості між джерелами (attention-based cosine):")
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


    s_ctx = sentiment_analysis(r, sent_data)
    print("\n Навчання MLP-класифікатора тональності (sklearn):")
    print(f"  Точність на корпусі: {s_ctx['train_acc']:.4f}")

    print("\n Власний Numpy MLP для тональності:")
    print(f"  Val accuracy (numpy MLP): {s_ctx['np_val_acc']:.4f}")

    print("\n Визначення тональності статей:")
    count = 0
    max_articles = 5
    for src in src_names:
        arts_src = [a for a in r["all_arts"] if a["source"] == src]
        for a in arts_src:
            if count >= max_articles:
                break
            p = a["pred_sentiment"]
            conf = a["sent_proba"].get(p, 0)
            print(f"  {p:<9} {conf:.2f}  {a['title'][:55]}")
            count += 1
        if count >= max_articles:
            break

    print("\n Навчання Autoencoder для латентного представлення")
    print(f"  Autoencoder final MSE loss: {s_ctx['ae'].history[-1]:.6f}")
    print(f"  Латентний простір: {s_ctx['X_ae'].shape} → {s_ctx['codes'].shape}")


    plot_training_curve(r["mlp_dom"].loss_curve_, getattr(r["mlp_dom"],"validation_scores_",[]), "l6_mlp_domain_loss.png", "MLP (sklearn) — Крива навчання (тематика)")
    plot_numpy_mlp_curve(r["np_mlp_dom"].history, "l6_numpy_mlp_domain_curve.png", "Numpy MLP — Loss та Val-accuracy (тематика)")
    plot_domain_distribution(r["all_arts"], src_names, "l6_domain_distribution.png")
    plot_sim_heatmap(r["sim_matrix"], src_names, "l6_attention_similarity.png")

    plot_training_curve(s_ctx["mlp_sent"].loss_curve_, getattr(s_ctx["mlp_sent"],"validation_scores_",[]), "l6_mlp_sentiment_loss.png", "MLP (sklearn) — Крива навчання (тональність)")
    plot_numpy_mlp_curve(s_ctx["np_mlp_sent"].history, "l6_numpy_mlp_sent_curve.png", "Numpy MLP — Loss та Val-accuracy (тональність)")
    plot_sentiment_bars(s_ctx["sent_agg"], src_names, "l6_sentiment_bars.png")
    plot_sentiment_heatmap(r["all_arts"], src_names, "l6_sentiment_heatmap.png")
    plot_autoencoder(s_ctx["ae"], s_ctx["codes"], r["all_arts"], src_names, "l6_autoencoder.png")
    plot_proba_violin(r["all_arts"], src_names, s_ctx["le_sent"], "l6_sentiment_proba_violin.png")


def scrape_newsapi(api_key: str, page_size: int = 100):
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
    corpus_docs = [preprocess(" ".join(ts.split()[i:i+8])) for ts in domain_corpus.values() for i in range(0, len(ts.split()), 8)]
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
        words = docs.split()
        for i in range(0, len(words), 8):
            dom_texts.append(preprocess(" ".join(words[i:i+8])))
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


    n_svd_d = min(64, X_dom_tr.shape[0] - 1)
    svd_d = TruncatedSVD(n_components=n_svd_d, random_state=RANDOM_STATE)
    Xr_d = svd_d.fit_transform(X_dom_tr)
    np_mlp_dom = NumpyMLP(layer_sizes=[n_svd_d, 128, 64, len(DOMAIN_LABELS)], lr=0.008, momentum=0.88, l2=1e-4, epochs=120, batch_size=4,)
    Xr_d_tr, Xr_d_val, yr_d_tr, yr_d_val = train_test_split(Xr_d, y_dom_tr, test_size=0.2, stratify=y_dom_tr, random_state=RANDOM_STATE)
    np_mlp_dom.fit(Xr_d_tr, yr_d_tr, Xr_d_val, yr_d_val)
    np_dom_val_acc = np_mlp_dom.history[-1].get("val_acc", 0)


    all_arts = [a for arts in articles.values() for a in arts]
    texts_pp = [preprocess(a["text"]) for a in all_arts]
    X_arts = tfidf.transform(texts_pp).toarray().astype(np.float32)
    Xs_arts = scaler_d.transform(X_arts)

    preds_dom = mlp_dom.predict(Xs_arts)
    preds_dom_labels = label_encoder.inverse_transform(preds_dom)
    for a, p in zip(all_arts, preds_dom_labels):
        a["pred_domain"] = p


    svd2 = TruncatedSVD(n_components=32, random_state=RANDOM_STATE)
    X_r32 = normalize(svd2.fit_transform(X_arts))
    attn_out, attn_weights = self_attention(X_r32, d_k=16)

    src_vecs = {}
    for src in src_names:
        mask = np.array([a["source"] == src for a in all_arts])
        src_vecs[src] = X_r32[mask].mean(axis=0)

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



def sentiment_analysis(ctx: dict, sent_data):
    X_sent_tr, y_sent_tr = sent_data
    all_arts = ctx["all_arts"]
    src_names = ctx["src_names"]

    label_encoder_sent = LabelEncoder()
    y_sent_tr = label_encoder_sent.fit_transform(y_sent_tr)


    scaler_s = StandardScaler(with_mean=False)
    Xs_s = scaler_s.fit_transform(X_sent_tr)

    mlp_sent = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation="relu", solver="adam",
        learning_rate_init=0.001, max_iter=400,
        alpha=1e-4, early_stopping=True,
        validation_fraction=0.25, random_state=RANDOM_STATE,
        verbose=False,
    )
    mlp_sent.fit(Xs_s, y_sent_tr)
    train_acc_s = accuracy_score(y_sent_tr, mlp_sent.predict(Xs_s))



    svd_s = TruncatedSVD(n_components=48, random_state=RANDOM_STATE)
    Xr_s  = svd_s.fit_transform(X_sent_tr)
    np_mlp_sent = NumpyMLP(
        layer_sizes=[48, 96, 48, len(SENT_LABELS)],
        lr=0.01, momentum=0.9, l2=1e-4, epochs=100, batch_size=4,
    )
    Xr_s_tr, Xr_s_val, yr_s_tr, yr_s_val = train_test_split(
        Xr_s, y_sent_tr, test_size=0.25, stratify=y_sent_tr, random_state=RANDOM_STATE)
    np_mlp_sent.fit(Xr_s_tr, yr_s_tr, Xr_s_val, yr_s_val)
    np_sent_val_acc = np_mlp_sent.history[-1].get("val_acc", 0)



    X_arts = ctx["X_arts"]
    Xs_arts_s = scaler_s.transform(X_arts)
    preds_sent  = mlp_sent.predict(Xs_arts_s)
    probas_sent = mlp_sent.predict_proba(Xs_arts_s)
    le_sent     = label_encoder_sent.inverse_transform(mlp_sent.classes_)  # string labels

    sent_agg = {}
    for a, p, prob in zip(all_arts, preds_sent, probas_sent):
        a["pred_sentiment"] = label_encoder_sent.inverse_transform([p])[0]
        a["sent_proba"]     = {cls: float(pr) for cls, pr in zip(le_sent, prob)}

    for src in src_names:
        arts_src = [a for a in all_arts if a["source"] == src]
        dist = Counter(a["pred_sentiment"] for a in arts_src)
        scores = []
        for a in arts_src:
            proba = a["sent_proba"]
            score = proba.get("positive",0) - proba.get("negative",0)
            scores.append(score)
        mean_score = sum(scores)/len(scores) if scores else 0
        sent_agg[src] = {
            "counts": dict(dist),
            "mean_score": round(mean_score, 4),
            "positive%": round(dist.get("positive",0)/len(arts_src)*100, 1),
            "negative%": round(dist.get("negative",0)/len(arts_src)*100, 1),
            "neutral%":  round(dist.get("neutral",0)/len(arts_src)*100, 1),
        }



    svd_ae = TruncatedSVD(n_components=128, random_state=RANDOM_STATE)
    X_ae   = normalize(svd_ae.fit_transform(X_arts).astype(np.float32))
    ae = NumpyAutoencoder(input_dim=128, hidden_dim=64, code_dim=16, lr=0.005, epochs=60, batch_size=8)
    ae.fit(X_ae)
    codes = ae.encode(X_ae)



    return {
        "mlp_sent": mlp_sent, 
        "np_mlp_sent": np_mlp_sent,
        "scaler_s": scaler_s, 
        "svd_ae": svd_ae,
        "codes": codes,
        "X_ae": X_ae,
        "ae": ae, 
        "sent_agg": sent_agg,
        "train_acc": train_acc_s, 
        "np_val_acc": np_sent_val_acc,
        "le_sent": le_sent
    }


def plot_training_curve(loss_curve, val_scores, fname, title):
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].plot(loss_curve, color="#1565C0", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Log-loss")
    axes[0].grid(alpha=0.3)
    if len(val_scores) > 0:
        axes[1].plot(val_scores, color="#C62828", linewidth=2)
        axes[1].set_title("Validation Accuracy")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No val data", ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle(title)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_numpy_mlp_curve(history, fname, title):
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"]  for h in history]
    val_accs = [h.get("val_acc") for h in history]
    fig, ax1 = plt.subplots(figsize=(9,4))
    ax1.plot(epochs, losses, color="#1565C0", linewidth=2, label="Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.grid(alpha=0.3)
    if any(v is not None for v in val_accs):
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_accs, color="#C62828", linewidth=2, linestyle="--", label="Val Acc")
        ax2.set_ylabel("Val Accuracy", color="#C62828")
        ax2.tick_params(axis="y", labelcolor="#C62828")
    
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper right")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_domain_distribution(arts, src_names, fname):
    fig, axes = plt.subplots(1, len(src_names), figsize=(5*len(src_names), 5))
    if len(src_names)==1: axes=[axes]
    colors = plt.cm.Set3(np.linspace(0,1,len(DOMAIN_LABELS)))
    for ax, src in zip(axes, src_names):
        src_arts = [a for a in arts if a["source"]==src]
        dist = Counter(a["pred_domain"] for a in src_arts)
        labels_ = list(dist.keys()); vals = list(dist.values())
        wedge_colors = [colors[DOMAIN_LABELS.index(l) if l in DOMAIN_LABELS else 0]
                        for l in labels_]
        ax.pie(vals, labels=labels_, autopct="%1.0f%%",
               colors=wedge_colors, startangle=140)
        ax.set_title(src)
    fig.suptitle("MLP — Розподіл тематики по виданнях")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_sim_heatmap(mat, names, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels([n[:10] for n in names])
    ax.set_yticklabels([n[:10] for n in names])
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j,i,f"{mat[i,j]:.3f}", ha="center", va="center",
                    color="white" if mat[i,j]>0.6 else "black")
    
    title = "Self-Attention Similarity між джерелами"
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()



def plot_sentiment_bars(agg, src_names, fname):
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    x=np.arange(len(src_names)); w=0.25
    axes[0].bar(x-w, [agg[s]["positive%"] for s in src_names], w,
                color="#43A047", label="Позитивна", alpha=0.87)
    axes[0].bar(x,   [agg[s]["neutral%"]  for s in src_names], w,
                color="#FB8C00", label="Нейтральна", alpha=0.87)
    axes[0].bar(x+w, [agg[s]["negative%"] for s in src_names], w,
                color="#E53935", label="Негативна", alpha=0.87)
    axes[0].set_xticks(x); axes[0].set_xticklabels(src_names, fontsize=9)
    axes[0].set_title("MLP — Тональність по джерелах")
    axes[0].set_ylabel("% статей"); axes[0].legend()

    means  = [agg[s]["mean_score"] for s in src_names]
    colors = ["#43A047" if m>0.05 else "#E53935" if m<-0.05 else "#FB8C00"
              for m in means]
    bars = axes[1].bar(src_names, means, color=colors, alpha=0.87, edgecolor="white")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    for bar,val in zip(bars,means):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     val+(0.01 if val>=0 else -0.02),
                     f"{val:+.3f}", ha="center")
    axes[1].set_title("Середній sentiment score (MLP)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_sentiment_heatmap(arts, src_names, fname):
    min_n = min(sum(1 for a in arts if a["source"]==s) for s in src_names)
    matrix = []
    for src in src_names:
        row = [a["sent_proba"].get("positive",0) - a["sent_proba"].get("negative",0)
               for a in arts if a["source"]==src][:min_n]
        matrix.append(row)
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(max(8, min_n), 3))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="P(pos)−P(neg)")
    ax.set_yticks(range(len(src_names))); ax.set_yticklabels(src_names)
    ax.set_xlabel("Стаття (індекс)")
    ax.set_title("Теплова карта тональності (MLP probabilities)")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_autoencoder(ae, codes, arts, src_names, fname):
    svd2 = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    c2d  = svd2.fit_transform(codes)
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    # scatter by source
    for src in src_names:
        mask = np.array([a["source"]==src for a in arts])
        axes[0].scatter(c2d[mask,0], c2d[mask,1], s=60,
                        label=src, alpha=0.8, edgecolors="white", lw=0.4)
    axes[0].set_title("Autoencoder латентний простір (2D SVD проекція)")
    axes[0].set_xlabel("Code-1"); axes[0].set_ylabel("Code-2")
    axes[0].legend(fontsize=8)
    # loss curve
    axes[1].plot(ae.history, color="#7B1FA2", linewidth=2)
    axes[1].fill_between(range(len(ae.history)), ae.history, alpha=0.15, color="#7B1FA2")
    axes[1].set_title("Autoencoder — MSE Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_proba_violin(arts, src_names, classes, fname):
    fig, axes = plt.subplots(1, len(src_names), figsize=(5*len(src_names), 5),
                             sharey=True)
    if len(src_names)==1: axes=[axes]
    colors_cls = {"negative":"#E53935","neutral":"#FB8C00","positive":"#43A047"}
    for ax, src in zip(axes, src_names):
        src_arts = [a for a in arts if a["source"]==src]
        data = [[a["sent_proba"].get(c,0) for a in src_arts] for c in classes]
        parts = ax.violinplot(data, positions=range(len(classes)), showmeans=True, showmedians=True)
        for pc, cls in zip(parts["bodies"], classes):
            pc.set_facecolor(colors_cls.get(cls,"#90CAF9"))
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_title(src)
        ax.set_ylabel("Probability")
    fig.suptitle("MLP Sentiment Probabilities — Violin Plot")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()










if __name__ == "__main__":
    main()