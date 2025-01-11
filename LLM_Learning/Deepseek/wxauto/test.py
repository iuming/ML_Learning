from wxauto import WeChat
from time import sleep
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com")

# 初始化 WeChat 客户端
wx = WeChat()

# 设置要监听的聊天对象
listen_list = ['小小怪下士']

# 添加监听对象
for chat in listen_list:
    wx.AddListenChat(who=chat)

# 给每一个列表联系人添加监听设置
for i in listen_list:
    wx.AddListenChat(who = i, savepic = True)

# 无线循环 持续监听和处理微信消息
while True:
    # 获取监听到的消息
    msgs = wx.GetListenMessage()
    for chat in msgs:
        # 获取聊天窗口名(人或群名)
        who = chat.who
        # 获取特定聊天窗口的消息列表
        one_megs = msgs.get(chat)
        # 遍历聊天窗口中的每条消息列表
        for msg in one_megs:
            # 获取消息类型
            msg_type = msg.type
            # 获取消息内容
            content = msg.content
            # 如果消息类型是"friend"即是来自好友的消息
            if msg_type == "friend":
                # 调用大模型
                # message = llm.predict(content)
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": content},
                    ],
                    stream=False
                )

                message = response.choices[0].message.content
                # 发送消息给当前窗口的好友
                chat.SendMsg(message)
