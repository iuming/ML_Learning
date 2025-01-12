import scrapy
from datetime import datetime

class SinaSpider(scrapy.Spider):
    name = "sina_spider"
    allowed_domains = ["news.sina.com.cn"]
    start_urls = ["https://news.sina.com.cn/"]

    def parse(self, response):
        # 提取新闻列表页中的新闻链接
        for news_link in response.css('a::attr(href)').getall():
            if "news.sina.com.cn" in news_link and "html" in news_link:
                yield response.follow(news_link, callback=self.parse_news)

    def parse_news(self, response):
        # 提取新闻标题
        title = response.css('h1.main-title::text').get() or response.css('title::text').get()

        # 提取新闻发布时间
        date = response.css('.date::text').get() or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 提取新闻正文内容
        content = " ".join(response.css('.article p::text').getall())

        # 返回提取的数据
        yield {
            'title': title,
            'date': date,
            'content': content,
            'url': response.url
        }