import requests
from bs4 import BeautifulSoup


def main():
    site_urls = ["https://www.pravda.com.ua/news/", "https://zaxid.net/news/"]

    site_int, site_url = get_site(site_urls)

    if site_int == 1:
        parser_pravda(site_url)
    elif site_int == 2:
        parser_zaxid(site_url)


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
    
    for article_tag in article_tags:
        articale_title = article_tag.text
        print(articale_title.strip())
    
    print()

def parser_zaxid(site_url):
    response = requests.get(site_url)
    if response.status_code in range(400, 600):
        print("oops. choose another site.")
        return
    site_html = response.text

    soup = BeautifulSoup(site_html, "html.parser")
    news_tags = soup.find_all("div", class_="news-title")
    for news_tag in news_tags:
        news_title = news_tag.text
        print(news_title.strip())



if __name__ == "__main__":
    main()