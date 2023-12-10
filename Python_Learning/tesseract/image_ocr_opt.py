import pytesseract
from PIL import Image
import cv2
import numpy as np

# 读取图片
image_path = 'cropped_image.jpg'
image = Image.open(image_path)

# 灰度化
gray_image = image.convert('L')

# 二值化
threshold_value = 200
binary_image = gray_image.point(lambda p: 0 if p < threshold_value else 255)

# 保存二值化后的图片
binary_image_path = 'binary_image.jpg'
binary_image.save(binary_image_path)

# 使用Tesseract进行文字识别
text = pytesseract.image_to_string(binary_image, lang='eng', config='--psm 6')

# 打印识别出的文字
print(text)

