import re
import os
import csv

from news_urls import pravda_urls, korrespondent_urls
from site_parsers import parser_pravda, parser_zaxid, parser_korrespondent
from text_mining import text_filter, build_wordcloud, build_line_plot
from data_analysis import get_top3_terms, build_term_time_series, analyze_freq_sum, analyze_top3_series


def main():
    site_urls = ["https://www.pravda.com.ua/news/", "https://zaxid.net/news/", "https://ua.korrespondent.net/all/"]

    option = int(input("""
        Choose an option (number):
            1. Parse a site
            2. Full analysis
        """))
    
    if option == 1:
        site_int, site_url = get_site(site_urls)

        if site_int == 1:
            news_file = parser_pravda(site_url)
        elif site_int == 2:
            news_file = parser_zaxid(site_url)
        elif site_int == 3:
            news_file = parser_korrespondent(site_url)
        print(f"Results in {news_file}")
        
        words, words_dict = text_filter(news_file)
        print(f"Words: {len(words)}")
        print(f"Unique words: {len(words_dict)}")

    elif option == 2:
        for pravda_url, korrespondent_url in zip(pravda_urls, korrespondent_urls):
            date = re.search(r"(\d{8})", pravda_url).group(1)
            news_pravda = parser_pravda(pravda_url)
            news_korrespondent = parser_korrespondent(korrespondent_url)
            news_merged = merge_news([news_pravda, news_korrespondent])

            words, words_dict = text_filter(news_merged)
            print(f"Date: {date[:2]}-{date[2:4]}-{date[4:]}")
            print(f"Words: {len(words)}")
            print(f"Unique words: {len(words_dict)}")
            # print(words_dict)

            freq_csv = freq_to_csv(news_merged, words_dict)
            print(f"Saved to {freq_csv}")
            
            print("----------------")
    
        table_csv = build_monitoring_table()
        print(f"Table was saved to {table_csv}")
        print("----------------")

        build_wordcloud(table_csv)
        build_line_plot(table_csv)

        top3_terms, top3_freq = get_top3_terms(table_csv)
        print("Top-3 terms:")
        for index, (term, freq) in enumerate(zip(top3_terms, top3_freq)):
            print(f"{index+1}. {term}, {freq}")
        print("----------------")

        term_series = build_term_time_series(table_csv)
        print("Term series:")
        for term, freq in term_series.items():
            print(f"{term} : {freq}")

        analyze_freq_sum(table_csv)
        analyze_top3_series(term_series)


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
    file_merged = f"output/news_{date}_raw.txt"
    with open(file_merged, "w", encoding="utf-8") as outfile:
        for file in files:
            with open(file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n")

    return file_merged


def freq_to_csv(filename, words_dict):
    filename = os.path.splitext(os.path.basename(filename))[0]
    date = filename.split("_")[1]

    os.makedirs("output", exist_ok=True)

    freq_csv = f"output/frequency_{date}.csv"
    with open(freq_csv, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["word", "frequency"])

        sorted_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)
        for word, frequency in sorted_words:
            writer.writerow([word, frequency])
    
    return freq_csv


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
    table_csv = "output/monitoring_table.csv"
    with open(table_csv, "w", encoding="utf-8", newline="") as file:
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

    return table_csv


if __name__ == "__main__":
    main()