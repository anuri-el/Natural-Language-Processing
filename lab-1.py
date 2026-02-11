import requests
from bs4 import BeautifulSoup


def main():
    site_urls = ["https://www.pravda.com.ua/news/", "https://zaxid.net/news/"]

    site_int, site_url = get_site(site_urls)

    if site_int == 1:
        news_txt = parser_pravda(site_url)
    elif site_int == 2:
        news_txt = parser_zaxid(site_url)
    else:
        print("no")

    text_filter(news_txt)


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

    article_tags = soup.find_all("div", class_="article_title")
    
    news_txt = "news_pravda_raw.txt"
    with open(news_txt, "w", encoding="utf-8") as output_file:
        for article_tag in article_tags:
            article_title = article_tag.text.strip()
            output_file.write(f"{article_title}\n")
    
    return news_txt


def parser_zaxid(site_url):
    response = requests.get(site_url)
    if response.status_code in range(400, 600):
        print("oops. choose another site.")
        return
    site_html = response.text

    soup = BeautifulSoup(site_html, "html.parser")
    news_tags = soup.find_all("div", class_="news-title")

    news_txt = "news_zaxid_raw.txt"
    with open(news_txt, "w", encoding="utf-8") as output_file:
        for news_tag in news_tags:
            news_title = news_tag.text.strip()
            output_file.write(f"{news_title}\n")

    return news_txt


def text_filter(filename):
    with open(filename, encoding="utf-8") as file:
        text = file.read()

    chars_to_replace = ["\n", ",", ".", "!", "?", ":", ";", "\"", "–", "-"]
    for ch in chars_to_replace:
        text = text.replace(ch, " ")

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