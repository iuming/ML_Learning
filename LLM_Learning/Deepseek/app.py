from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-",
    base_url="https://api.deepseek.com"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # 获取用户输入
    user_input = request.json.get("message")

    # 调用 OpenAI API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ],
        stream=False
    )

    # 返回 AI 的回复
    ai_response = response.choices[0].message.content
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)