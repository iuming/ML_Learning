import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# 定义抓取网页的函数
def fetch_web_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # 假设我们要抓取的是页面的所有段落内容
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    else:
        print(f"Failed to retrieve content from {url}")
        return None

# 定义调用 OpenAI API 的函数
def summarize_content(content):
    client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"Please summarize the following content: {content}"},
        ],
        stream=False
    )
    return response.choices[0].message.content

# 主程序
if __name__ == "__main__":
    url = "https://iuming.github.io/2024/12/25/%E4%BA%8C%E6%AC%A1%E5%A3%B0%E6%8A%80%E6%9C%AF%E5%AE%9A%E4%BD%8D%E8%B6%85%E5%AF%BC%E8%85%94%E5%A4%B1%E8%B6%85%E4%BD%8D%E7%BD%AE/"  # 替换为你想抓取的 URL
    content = fetch_web_content(url)
    
    if content:
        summary = summarize_content(content)
        print("Summary:")
        print(summary)