import cv2
import pytesseract

# 打开视频文件
video_capture = cv2.VideoCapture('testVideo.mp4')

# 设置帧率
fps = video_capture.get(cv2.CAP_PROP_FPS)

# 读取视频帧
success, frame = video_capture.read()

# 每秒截取一张图片并进行文字识别
while success:
    frameId = int(round(video_capture.get(1)) - 1)  # 当前帧数（从0开始）
    current_time = frameId / fps  # 当前时间点
    if frameId % int(fps) == 0:
        # 将第一个区域转换为灰度图像
        region1_gray = cv2.cvtColor(frame[0:70, 80:740], cv2.COLOR_BGR2GRAY)  # 第一个区域转换为灰度图像
        
        # 对灰度图像进行二值化处理
        _, region1_binary = cv2.threshold(region1_gray, 200, 255, cv2.THRESH_BINARY)  # 第一个区域二值化处理
        
        # 对二值化图像进行文字识别
        text1 = pytesseract.image_to_string(region1_binary, lang='eng', config='--psm 6')  # 第一个区域OCR识别
        text1 = text1.replace("/", "_")  # 将"/"替换成"_"
        
        # 保存第二个区域的灰度图像，并使用替换后的text1作为文件名
        region2_gray = cv2.cvtColor(frame[335:385, 1000:1230], cv2.COLOR_BGR2GRAY)  # 第二个区域转换为灰度图像
        region2_filename = f"{text1.strip()}.jpg"  # 使用替换后的text1作为文件名
        cv2.imwrite(region2_filename, region2_gray)
        print(f"Saved image: {region2_filename}")

    success, frame = video_capture.read()

# 释放视频捕获对象
video_capture.release()

