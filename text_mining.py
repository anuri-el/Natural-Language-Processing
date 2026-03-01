import re
import csv
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

def text_filter(filename):
    with open(filename, "r", encoding="utf-8") as input_file:
        text = input_file.read()

    chars_to_replace = ["\n", ",", ".", "!", "?", ":", ";", "\"", "–", "«", "»", "\"", "$"]
    for ch in chars_to_replace:
        text = text.replace(ch, " ")
    text = re.sub(r"\d+", "", text)
    
    file = filename[:-7] + "filtered.txt"
    with open(file, "w", encoding="utf-8", newline="") as output_file:
        output_file.write(text)

    text = text.lower()
    words = text.split()
    words = remove_stop_words(words)
    words.sort()
    words_dict = Counter(words)

    return words, words_dict


def remove_stop_words(words):
    stop_words = ["і", "та", "в", "на", "не", "для", "з", "що", "це", "до", "за", "як", "у", "про", "по", "зі", "через", "проти", "під", "є", "де", "якщо", "ще", "чи", "фото", "понад", "від", "має", "після", "й", "щодо", "із", "який", "відео"]

    words = [word for word in words if word not in stop_words]
    return words


def build_wordcloud(filename):
    freq_dict = {}

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            word = row["Топ 5"]
            freq = int(row["Частота"])

            if word in freq_dict:
                freq_dict[word] += freq
            else:
                freq_dict[word] = freq

    wordcloud = WordCloud(width=1000, height=700, background_color="white").generate_from_frequencies(freq_dict)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Хмара слів")
    plt.savefig("output/wordcloud.png")
    plt.show()


def build_line_plot(filename):
    dates = []
    sums = []

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        
        last_period = None
        for row in reader:
            period = row["День"] + "_" + row["Час"]

            if period != last_period:
                dates.append(period)
                sums.append(int(row["Сума частот"]))
                last_period = period
        
        plt.figure(figsize=(12,6))
        plt.plot(dates, sums, marker="o")

        plt.xticks(rotation=45)
        plt.xlabel("Період моніторингу")
        plt.ylabel("Сума частот")
        plt.title("Динаміка суми частот")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("output/lineplot.png")
        plt.show()