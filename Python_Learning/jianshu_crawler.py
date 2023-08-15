import requests
from bs4 import BeautifulSoup


def crawl_jianshu():
    url = "https://www.jianshu.com/"
    response = requests.get(url)
    # soup = BeautifulSoup(response.text, 'html.parser')
    soup = BeautifulSoup(response.text, 'lxml')

    articles = soup.find_all('a', class_='title')

    for article in articles:
        title = article.text.strip()
        link = url + article['href']

        print("Title:", title)
        print("Link:", link)
        print("--------------------")


crawl_jianshu()
