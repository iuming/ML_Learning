# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class SinaNewsPipeline:
    def process_item(self, item, spider):
        return item

from openai import OpenAI

class NewsCommentaryPipeline:
    def __init__(self):
        self.client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com")

    def process_item(self, item, spider):
        prompt = f"根据以下新闻内容写一篇时评文章:\n标题: {item['title']}\n内容: {item['content']}\n"
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位资深新闻评论员。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        commentary = response.choices[0].message.content
        item['commentary'] = commentary
        return item