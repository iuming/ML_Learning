from PIL import Image

# 打开图片文件
image = Image.open('frame0.jpg')

# 定义要截取的区域
#x1, y1, x2, y2 = 80, 0, 740, 70  # 左上角和右下角坐标
x1, y1, x2, y2 = 1000, 335, 1230, 385  # 左上角和右下角坐标
# 截取图片中的指定区域
cropped_image = image.crop((x1, y1, x2, y2))

# 保存截取后的图片
cropped_image.save('cropped_image.jpg')

