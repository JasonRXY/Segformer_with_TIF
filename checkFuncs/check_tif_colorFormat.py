from PIL import Image

# 打开TIFF文件
img = Image.open(r"C:\Users\obt\Desktop\segformer-pytorch-master\VOCdevkit\VOC2007\JPEGImages\0521YuqingXuetangkan_1_49.tif")

# 输出图像模式（通常是 'RGB' 或 'BGR' 等）
print(img.mode)

# 获取并打印图像的一个像素点的颜色值
# 这可以帮助确认颜色顺序
x, y = 0, 0  # 你可以修改这些值来检查不同的像素点
print(img.getpixel((x, y)))
