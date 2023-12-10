import pytesseract
from PIL import Image

# 打开图片文件
image = Image.open('cropped_image.jpg')

# 使用Tesseract进行文字识别
text = pytesseract.image_to_string(image)

# 打印识别出的文字
print(text)

