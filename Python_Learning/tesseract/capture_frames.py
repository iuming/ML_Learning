import cv2

# 打开视频文件
video_capture = cv2.VideoCapture('testVideo.mp4')

# 设置帧率
fps = video_capture.get(cv2.CAP_PROP_FPS)

# 设置计数器
count = 0

# 读取视频帧
success, image = video_capture.read()

# 每秒截取一张图片
while success:
    frameId = int(round(video_capture.get(1)))  # 当前帧数
    if frameId % int(fps) == 0:
        # 保存图片
        cv2.imwrite("frame%d.jpg" % count, image)
        count += 1
    success, image = video_capture.read()

# 释放视频捕获对象
video_capture.release()

