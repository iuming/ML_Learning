import requests
from bs4 import BeautifulSoup


def crawl_arxiv():

    url = "https://arxiv.org/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    # print(response.text)

    # soup = BeautifulSoup(response.text, 'html.parser')
    soup = BeautifulSoup(response.text, 'lxml')
    

    papers = soup.find_all('li', class_='arxiv-result')

    for paper in papers:
        title = paper.find('p', class_='title').text.strip()
        authors = paper.find('p', class_='authors').text.strip()
        abstract = paper.find('p', class_='abstract').text.strip()

        print("Title:", title)
        print("Authors:", authors)
        print("Abstract:", abstract)
        print("--------------------")


crawl_arxiv()
