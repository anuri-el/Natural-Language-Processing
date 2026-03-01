import os
import requests
from bs4 import BeautifulSoup


def parser_pravda(site_url): 
    response = requests.get(site_url)
    if response.status_code in range(400, 600):
        print("something went wrong. choose another site.")
        return
    site_html = response.text

    soup = BeautifulSoup(site_html, "html.parser")
    article_title_tags = soup.find_all("div", class_="article_title")

    os.makedirs("output", exist_ok=True)
    date = site_url[-9:-1]
    news_txt = f"output/pravda_{date}_raw.txt"
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

    os.makedirs("output", exist_ok=True)
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

    os.makedirs("output", exist_ok=True)
    month = site_url.split("/")[-3]
    day = site_url.split("/")[-2]
    news_txt = f"output/korrespondent_{month}_{day}_raw.txt"
    with open(news_txt, "w", encoding="utf-8", newline="") as output_file:
        for article_title_tag in article_title_tags:
            article_title = article_title_tag.text.strip() +  "\n"
            output_file.write(article_title)
            # print(article_title)
    return news_txt