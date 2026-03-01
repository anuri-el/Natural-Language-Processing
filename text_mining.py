import re
from collections import Counter

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
