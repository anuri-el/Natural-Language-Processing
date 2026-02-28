import requests
from bs4 import BeautifulSoup
import csv


def main():
    site_urls = ["https://www.pravda.com.ua/news/", "https://zaxid.net/news/"]

    site_int, site_url = get_site(site_urls)

    if site_int == 1:
        news_file = parser_pravda(site_url)
    elif site_int == 2:
        news_file = parser_zaxid(site_url)
    
    words, words_dict = text_filter(news_file)
    print(f"words: {len(words)}")
    print(f"unique words: {len(words_dict)}")

    # --- merge news ---
    print("---combined---")
    news_pravda = parser_pravda(site_urls[0])
    news_zaxid = parser_zaxid(site_urls[1])
    news_combined = combine_news([news_pravda, news_zaxid])

    words, words_dict = text_filter(news_combined)
    print(f"words: {len(words)}")
    print(f"unique words: {len(words_dict)}")



def get_site(site_urls):
    while True: 
        site = input(f"""
        Choose a site (number):
            1. {site_urls[0]}
            2. {site_urls[1]}
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

    news_txt = "output/news_pravda_raw.txt"
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


def text_filter(filename):
    with open(filename, "r", encoding="utf-8") as input_file:
        text = input_file.read()

    chars_to_replace = ["\n", ",", ".", "!", "?", ":", ";", "\"", "–", "-", "«", "»", "\""]
    for ch in chars_to_replace:
        text = text.replace(ch, " ")
    
    file = filename[:-7] + "filtered.txt"
    with open(file, "w", encoding="utf-8", newline="") as output_file:
        output_file.write(text)

    text = text.lower()
    words = text.split()
    words.sort()
    words_dict = dict()
    for word in words:
        if word in words_dict:
            words_dict[word] = words_dict[word] + 1
        else:
            words_dict[word] = 1

    return words, words_dict


def combine_news(files):
    combined_file = "output/all_news_raw.txt"
    with open(combined_file, "w", encoding="utf-8") as outfile:
        for file in files:
            with open(file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n")

    return combined_file


if __name__ == "__main__":
    main()