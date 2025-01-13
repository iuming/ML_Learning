import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

# 邮件配置
SMTP_SERVER = "smtp.example.com"  # 替换为你的 SMTP 服务器地址
SMTP_PORT = 587  # 替换为你的 SMTP 端口
SENDER_EMAIL = "your_email@example.com"  # 替换为发件人邮箱
SENDER_PASSWORD = "your_password"  # 替换为发件人邮箱密码
RECEIVER_EMAIL = "receiver_email@example.com"  # 替换为收件人邮箱

# 读取 sina_news.json 文件
def read_news_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        news_data = json.load(file)
    return news_data

# 格式化新闻数据为邮件正文
def format_news_to_email(news_data):
    email_body = "今日新闻摘要：\n\n"
    for news in news_data:
        email_body += f"标题: {news['title']}\n"
        email_body += f"日期: {news['date']}\n"
        email_body += f"内容: {news['content'][:200]}...\n"  # 只显示前 200 个字符
        email_body += f"时评: {news['commentary'][:200]}...\n"  # 只显示前 200 个字符
        email_body += "-" * 50 + "\n"
    return email_body

# 发送邮件
def send_email(subject, body):
    # 创建邮件对象
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject

    # 添加邮件正文
    msg.attach(MIMEText(body, "plain"))

    # 连接 SMTP 服务器并发送邮件
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # 启用 TLS 加密
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败: {e}")
    finally:
        server.quit()

# 主函数
def main():
    # 读取新闻数据
    news_data = read_news_file("sina_news.json")

    # 格式化新闻数据为邮件正文
    email_body = format_news_to_email(news_data)

    # 发送邮件
    send_email("今日新闻摘要", email_body)

if __name__ == "__main__":
    main()