import re
import os
import csv

from news_urls import pravda_urls, korrespondent_urls
from site_parsers import parser_pravda, parser_zaxid, parser_korrespondent
from text_mining import text_filter, remove_stop_words, build_wordcloud, build_line_plot
from data_analysis import get_top3_terms, build_term_time_series


def main():
    site_urls = ["https://www.pravda.com.ua/news/", "https://zaxid.net/news/", "https://ua.korrespondent.net/all/"]

    # print("-----pick a site-----")
    # site_int, site_url = get_site(site_urls)

    # if site_int == 1:
    #     news_file = parser_pravda(site_url)
    # elif site_int == 2:
    #     news_file = parser_zaxid(site_url)
    # elif site_int == 3:
    #     news_file = parser_korrespondent(site_url)
    
    # words, words_dict = text_filter(news_file)
    # print(f"words: {len(words)}")
    # print(f"unique words: {len(words_dict)}")


    # --- merge news ---
    print("-----combined-----")

    # for pravda_url, korrespondent_url in zip(pravda_urls, korrespondent_urls):
    #     news_pravda = parser_pravda(pravda_url)
    #     news_korrespondent = parser_korrespondent(korrespondent_url)
    #     news_combined = merge_news([news_pravda, news_korrespondent])

    #     words, words_dict = text_filter(news_combined)
    #     print(f"words: {len(words)}")
    #     print(f"unique words: {len(words_dict)}")
    #     # print(words_dict)

    #     freq_to_csv(news_combined, words_dict)
        
    #     print("-----")
    
    output_file = build_monitoring_table()
    build_wordcloud(output_file)
    build_line_plot(output_file)
    top3 = get_top3_terms(output_file)
    print(top3)
    term_series = build_term_time_series(output_file)
    print(term_series)


def get_site(site_urls):
    while True: 
        site = input(f"""
        Choose a site (number):
            1. {site_urls[0]}
            2. {site_urls[1]}
            3. {site_urls[2]}
        """)
        try:
            site_int = int(site)
            if site_int in range(1, len(site_urls) + 1):
                break
        except:
            print("should be a number from the ones above")

    site_url = site_urls[site_int - 1]
    return site_int, site_url


def merge_news(files):
    match = re.search(r"\d{8}", files[0])
    date = match.group()
    combined_file = f"output/news_{date}_raw.txt"
    with open(combined_file, "w", encoding="utf-8") as outfile:
        for file in files:
            with open(file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n")

    return combined_file


def freq_to_csv(newsfile, words_dict):
    newsfile = os.path.splitext(os.path.basename(newsfile))[0]
    newsfile = newsfile.split("_")[1]
    # top5 = words_dict.most_common(5)

    os.makedirs("output", exist_ok=True)

    filename = f"output/frequency_{newsfile}.csv"
    with open(filename, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["word", "frequency"])

        sorted_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)
        for word, frequency in sorted_words:
            writer.writerow([word, frequency])


def get_frequency_files(folder="output"):
    files = [file for file in os.listdir(folder) if file.startswith("frequency_")]
    files.sort(key=lambda x: re.search(r"\d{8}", x).group())
    return files


def load_frequency_files(filepath):
    freq_dict = {}
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            freq_dict[row["word"]] = int(row["frequency"])
    
    return freq_dict


def build_monitoring_table():
    files = get_frequency_files("output")

    time_labels = ["Ранок", "Обід", "Вечір"]
    output_file = "output/monitoring_table.csv"
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["День", "Час", "Топ 5", "Частота", "Сума частот", "Коментар"])

        day_number = 1
        for i in range(0, len(files), 3):
            day_files = files[i:i+3]

            for index, filename in enumerate(day_files):
                filepath = os.path.join("output", filename)

                freq_dict = load_frequency_files(filepath)

                top5 = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                total_sum = sum(freq for word, freq in top5)

                date_raw = re.search(r"\d{8}", filename).group()
                date_str = f"{date_raw[:2]}.{date_raw[2:4]}.{date_raw[4:]}"

                for word, freq in top5:
                    writer.writerow([f"{day_number} ({date_str})", time_labels[index], word, freq, total_sum, ""])
            
            day_number += 1
    return output_file


if __name__ == "__main__":
    main()