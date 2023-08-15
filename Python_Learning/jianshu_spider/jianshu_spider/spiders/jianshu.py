import scrapy

class JianshuSpider(scrapy.Spider):
    name = "jianshu"
    start_urls = ["https://www.jianshu.com/"]

    def parse(self, response):
        articles = response.css('a.title')

        for article in articles:
            title = article.css('::text').get().strip()
            link = response.urljoin(article.css('::attr(href)').get())

            yield {
                'title': title,
                'link': link
            }
