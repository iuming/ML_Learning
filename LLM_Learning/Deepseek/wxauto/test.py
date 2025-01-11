from wxauto import WeChat
from time import sleep
from openai import OpenAI

client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com")

wx = WeChat()

listen_list = ['小小怪下士']

# 添加监听对象
for chat in listen_list:
    wx.AddListenChat(who=chat)

while True:
    # 获取监听消息
    msgs = wx.GetListenMessage()
    
    # 对每条消息进行处理
    for chat in msgs:
        one_msgs = msgs.get(chat)
        
        if one_msgs:  # 确保消息不为空
            # 将消息内容转换为字符串
            message_content = str(one_msgs)  # 应该将消息转换为字符串
            # 调用 OpenAI API 生成回复
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message_content},  # 使用当前消息字符串
                ],
                stream=False
            )

            # 获取回复内容
            reply = response.choices[0].message.content

            who = '小小怪下士'
            # 发送回复消息
            wx.SendMsg(reply, who=who)  # 发送到当前聊天对象
            
    sleep(1)  # 等待一段时间再循环