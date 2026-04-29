import os
import pandas as pd


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    df = pd.read_csv("./outputs/l3_topics.csv")

    TOPIC_NAMES = df["topic"].unique().tolist()
    UNKNOWN_LABEL = "невідомо / unknown"

    descriptions = df["description"].tolist()
    TOPIC_DOCS = {}
    for topic in TOPIC_NAMES:
        topic_descriptions = df[df["topic"] == topic]["description"].tolist()
        TOPIC_DOCS[topic] = " ".join(topic_descriptions)

    # while True:
    #     try:
    #         text = input("\n > ").strip()
    #     except Exception:
    #         break

    #     if not text or text.lower() in ("exit", "quit", "q"):
    #         break






if __name__ == "__main__":
    main()