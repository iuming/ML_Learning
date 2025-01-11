import numpy as np
from openai import OpenAI

def read_data(file_name):
    """读取文件中的数据并返回一个 numpy 数组"""
    try:
        return np.loadtxt(file_name)
    except Exception as e:
        print(f"读取文件 {file_name} 时出错: {e}")
        return None

def call_llm_for_piezo_compensation(Eacc_previous, Delta_f, Piezo_previous, Eacc_now):
    """
    将数据传递给大语言模型，LLM根据输入数据生成新的Piezo驱动信号。
    """
    client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com")
    
    # 将数据转换为文本格式，作为输入传递给 LLM
    data_str = f"""
    这是上一周期的数据：
    上一周期加速梯度 (Eacc_previous): {Eacc_previous}
    上一周期失谐量 (Delta_f): {Delta_f}
    上一周期Piezo驱动信号 (Piezo_previous): {Piezo_previous}
    
    这是本周期的预计加速梯度 (Eacc_now): {Eacc_now}
    
    请根据以上数据提供本周期的Piezo补偿驱动信号 (Piezo_now)，目标是将失谐量最小化。

    只回答本周期的Piezo驱动信号数值(以换行符分隔的数值,不要省略中间参数),不要回答任何其他文本.
    """

    # 通过API调用模型来获得结果
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个帮助生成Piezo补偿驱动信号的助手。"},
                {"role": "user", "content": data_str},
            ],
            stream=False
        )

        # 从API响应中提取Piezo_now信号
        Piezo_now_str = response.choices[0].message.content

        # 解析模型输出（假设是以换行符分隔的数值）
        Piezo_now = np.array([float(val) for val in Piezo_now_str.strip().split("\n")])
        return Piezo_now

    except Exception as e:
        print(f"调用API时出错: {e}")
        return None

def write_data(file_name, data):
    """将数据写入文件"""
    try:
        np.savetxt(file_name, data)
    except Exception as e:
        print(f"写入文件 {file_name} 时出错: {e}")

def main():
    # 读取上一周期的加速梯度、失谐量和Piezo驱动信号
    Eacc_previous = read_data("Eacc_previous.txt")
    Delta_f = read_data("Delta_f.txt")
    Piezo_previous = read_data("Piezo_previous.txt")
    
    # 读取当前周期的预计加速梯度
    Eacc_now = read_data("Eacc_now.txt")
    
    if Eacc_previous is not None and Delta_f is not None and Piezo_previous is not None and Eacc_now is not None:
        # 调用LLM获取新的Piezo驱动信号
        Piezo_now = call_llm_for_piezo_compensation(Eacc_previous, Delta_f, Piezo_previous, Eacc_now)
        
        if Piezo_now is not None:
            # 输出新的 Piezo 驱动信号到 Piezo_now.txt 文件
            write_data("Piezo_now.txt", Piezo_now)
            print("Piezo 驱动信号已计算并保存到 Piezo_now.txt")
        else:
            print("无法获取Piezo补偿信号。")
    else:
        print("有文件未能正确读取，无法进行补偿。")

if __name__ == "__main__":
    main()
