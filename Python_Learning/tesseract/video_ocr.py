import cv2
import pytesseract
import pandas as pd

# 读取视频
video_path = 'testVideo.mp4'
cap = cv2.VideoCapture(video_path)

# 视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 初始化识别结果列表
results = []

# 逐帧读取视频
while cap.isOpened():
    # 获取当前帧
    ret, frame = cap.read()
    
    # 检查是否到达视频末尾
    if not ret:
        break
    
    # 获取当前帧的时间点
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    
    # 每隔1秒截取一张图片
    if current_time % 1 == 0:
        # 对图片进行文字识别
        text1 = pytesseract.image_to_string(frame[0:70, 80:740])  # 第一个区域识别
        text2 = pytesseract.image_to_string(frame[335:385, 1000:1230])  # 第二个区域识别
        
        # 将识别结果添加到列表中
        results.append([text1, text2])

# 释放视频对象
cap.release()

# 将识别结果保存为CSV文件
df = pd.DataFrame(results, columns=['区域1识别结果', '区域2识别结果'])
df.to_csv('识别结果.csv', index=False)

