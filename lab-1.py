import re
import csv
import requests
from bs4 import BeautifulSoup
from collections import Counter

def main():
    pravda_urls = ["https://www.pravda.com.ua/news/date_08022026/",
                   "https://www.pravda.com.ua/news/date_09022026/",
                   "https://www.pravda.com.ua/news/date_10022026/",
                   "https://www.pravda.com.ua/news/date_11022026/",
                   "https://www.pravda.com.ua/news/date_12022026/",
                   "https://www.pravda.com.ua/news/date_13022026/",
                   "https://www.pravda.com.ua/news/date_14022026/",
                   "https://www.pravda.com.ua/news/date_15022026/",
                   "https://www.pravda.com.ua/news/date_16022026/",
                   "https://www.pravda.com.ua/news/date_17022026/",
                   "https://www.pravda.com.ua/news/date_18022026/",
                   "https://www.pravda.com.ua/news/date_19022026/",
                   "https://www.pravda.com.ua/news/date_20022026/",
                   "https://www.pravda.com.ua/news/date_21022026/",
                   "https://www.pravda.com.ua/news/date_22022026/",
                   "https://www.pravda.com.ua/news/date_23022026/",
                   "https://www.pravda.com.ua/news/date_24022026/",
                   "https://www.pravda.com.ua/news/date_25022026/",
                   "https://www.pravda.com.ua/news/date_26022026/",
                   "https://www.pravda.com.ua/news/date_27022026/",
                   "https://www.pravda.com.ua/news/date_28022026/" ]
    
    korrespondent_urls = ["https://ua.korrespondent.net/all/2026/february/8/", 
                          "https://ua.korrespondent.net/all/2026/february/9/",
                          "https://ua.korrespondent.net/all/2026/february/10/",
                          "https://ua.korrespondent.net/all/2026/february/11/",
                          "https://ua.korrespondent.net/all/2026/february/12/",
                          "https://ua.korrespondent.net/all/2026/february/13/",
                          "https://ua.korrespondent.net/all/2026/february/14/",
                          "https://ua.korrespondent.net/all/2026/february/15/",
                          "https://ua.korrespondent.net/all/2026/february/16/",
                          "https://ua.korrespondent.net/all/2026/february/17/",
                          "https://ua.korrespondent.net/all/2026/february/18/",
                          "https://ua.korrespondent.net/all/2026/february/19/",
                          "https://ua.korrespondent.net/all/2026/february/20/",
                          "https://ua.korrespondent.net/all/2026/february/21/",
                          "https://ua.korrespondent.net/all/2026/february/22/",
                          "https://ua.korrespondent.net/all/2026/february/23/",
                          "https://ua.korrespondent.net/all/2026/february/24/",
                          "https://ua.korrespondent.net/all/2026/february/25/",
                          "https://ua.korrespondent.net/all/2026/february/26/",
                          "https://ua.korrespondent.net/all/2026/february/27/",
                          "https://ua.korrespondent.net/all/2026/february/28/" ]
    
    site_urls = ["https://www.pravda.com.ua/news/", "https://zaxid.net/news/", "https://ua.korrespondent.net/all/"]

    # print("---pick a site---")
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
    print("---combined---")

    for pravda_url, korrespondent_url in zip(pravda_urls, korrespondent_urls):
        news_pravda = parser_pravda(pravda_url)
        news_korrespondent = parser_korrespondent(korrespondent_url)
        news_combined = combine_news([news_pravda, news_korrespondent])

        words, words_dict = text_filter(news_combined)
        print(f"words: {len(words)}")
        print(f"unique words: {len(words_dict)}")
        print(words_dict)


        top5 = words_dict.most_common(5)
        print(top5)
        save_to_csv(words_dict)
        print("---")



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


def parser_pravda(site_url): 
    response = requests.get(site_url)
    if response.status_code in range(400, 600):
        print("something went wrong. choose another site.")
        return
    site_html = response.text

    soup = BeautifulSoup(site_html, "html.parser")

    article_title_tags = soup.find_all("div", class_="article_title")

    news_txt = f"output/pravda_{site_url[-9:-1]}_raw.txt"
    with open(news_txt, "w", encoding="utf-8", newline="") as output_file:
        for article_title_tag in article_title_tags:
            article_title = article_title_tag.text.strip() +  "\n"
            output_file.write(article_title)
            # print(article_title)
    
    return news_txt


def parser_zaxid(site_url):
    response = requests.get(site_url)
    if response.status_code in range(400, 600):
        print("oops. choose another site.")
        return
    site_html = response.text

    soup = BeautifulSoup(site_html, "html.parser")
    news_title_tags = soup.find_all("div", class_="news-title")

    news_txt = "output/news_zaxid_raw.txt"
    with open(news_txt, "w", encoding="utf-8", newline="") as output_file:
        for news_title_tag in news_title_tags:
            news_title = news_title_tag.text.strip() + " \n"
            output_file.write(news_title)
            # print(news_title)
    return news_txt


def parser_korrespondent(site_url): 
    response = requests.get(site_url)
    if response.status_code in range(400, 600):
        print("something went wrong. choose another site.")
        return
    site_html = response.text

    soup = BeautifulSoup(site_html, "html.parser")

    article_title_tags = soup.find_all("div", class_="article__title")

    month = site_url.split("/")[-3]
    day = site_url.split("/")[-2]
    news_txt = f"output/korrespondent_{month}_{day}_raw.txt"
    with open(news_txt, "w", encoding="utf-8", newline="") as output_file:
        for article_title_tag in article_title_tags:
            article_title = article_title_tag.text.strip() +  "\n"
            output_file.write(article_title)
            # print(article_title)
    
    return news_txt


def combine_news(files):
    date = files[0].split("_")[-2]
    combined_file = f"output/news_{date}_raw.txt"
    with open(combined_file, "w", encoding="utf-8") as outfile:
        for file in files:
            with open(file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n")

    return combined_file


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
    stop_words = ["і", "та", "в", "на", "не", "для", "з", "що", "це", "до", "за", "як", "у", "про", "по", "зі", "через", "проти", "під", "є", "де", "якщо", "ще", "чи", "фото"]

    words = [word for word in words if word not in stop_words]
    return words


def save_to_csv(words_dict):
    filename = "output/frequency.csv"
    with open(filename, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["word", "frequency"])

        for word, frequency in words_dict.items():
            writer.writerow([word, frequency])

if __name__ == "__main__":
    main()