# 羊咩咩
# happy happy happy
import numpy as np
import cv2
import os

# 遍历文件夹内label为19类的图
def color_dict():
    image = []
    img_path = r'D:\\my_work\\MagNet-main\\lable\\CityScapes\\gtFine\\train\\strasbourg\\'
    file_name = os.listdir(img_path)
    for i in file_name:
        if i.endswith('_color.png'):
            image.append(i)

    for i in range(len(image)):
        ImagePath = img_path + image[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        if len(unique) == 19:
            print(image[i])

color_dict()
