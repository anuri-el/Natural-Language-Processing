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
    else:
        print("no")

    text_filter(news_file)


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

    article_time_tags = soup.find_all("div", class_="article_time")
    article_title_tags = soup.find_all("div", class_="article_title")
    
    news_csv = "news_pravda_raw.csv"
    with open(news_csv, "w", newline="",encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["time", "title"])
        writer.writeheader()
        for article_time_tag, article_title_tag in zip(article_time_tags, article_title_tags):
            article_time = article_time_tag.text.strip()
            article_title = article_title_tag.text.strip()
            writer.writerow({"time" : article_time, "title" : article_title})
    
    return news_csv


def parser_zaxid(site_url):
    response = requests.get(site_url)
    if response.status_code in range(400, 600):
        print("oops. choose another site.")
        return
    site_html = response.text

    soup = BeautifulSoup(site_html, "html.parser")
    news_title_tags = soup.find_all("div", class_="news-title")
    news_time_tags = soup.find_all("div", class_="time")

    news_csv = "news_zaxid_raw.csv"
    with open(news_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, ["time", "title"])
        writer.writeheader()
        for news_time_tag, news_title_tag in zip(news_time_tags, news_title_tags):
            news_time = news_time_tag.text.strip()
            news_title = news_title_tag.text.strip()
            writer.writerow({"time" : news_time, "title" : news_title})

    return news_csv


def text_filter(filename):
    text = ""
    with open(filename, encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        
        for row in reader:
            text += row["title"] + " "

    chars_to_replace = ["\n", ",", ".", "!", "?", ":", ";", "\"", "–", "-"]
    for ch in chars_to_replace:
        text = text.replace(ch, " ")
    
    file = filename[:-7] + "filtered.txt"
    with open(file, "w", newline="", encoding="utf-8") as output_file:
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

    print(f"words: {len(words)}")
    print(f"unique words: {len(words_dict)}")


if __name__ == "__main__":
    main()