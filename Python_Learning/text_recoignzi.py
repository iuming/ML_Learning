"""
Program Name: text_recoignzi
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/8 下午7:08

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/8 下午7:08: Initial Create.

"""

import cv2
import pytesseract

# 设置Tesseract OCR引擎的安装路径
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# 打开视频文件
cap = cv2.VideoCapture('your_video.mp4')

# 逐帧读取视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将每一帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用Tesseract进行文字识别
    text = pytesseract.image_to_string(gray)

    # 输出识别的文字
    print(text)

    # 显示每一帧
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
