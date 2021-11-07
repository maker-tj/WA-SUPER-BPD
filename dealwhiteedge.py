import cv2
import numpy as np

import os
from PIL import Image


def label2color(label):
    label = label.astype(np.uint16)

    height, width = label.shape
    color3u = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(label)

    if unique_labels[-1] >= 2 ** 24:
        raise RuntimeError('Error: label overflow!')

    for i in range(len(unique_labels)):
        binary = '{:024b}'.format(unique_labels[i])
        # r g b 3*8 24
        r = int(binary[::3][::-1], 2)
        g = int(binary[1::3][::-1], 2)
        b = int(binary[2::3][::-1], 2)

        color3u[label == unique_labels[i]] = np.array([r, g, b])

    return color3u

def change():
    file_outpath = 'D:/data/VOC2012/segmentation_deal/'
    file_path = 'D:/data/VOC2012/SegmentationObject/'

    ######################################### 文件中的图像去除白色边缘为背景
    list = os.listdir(file_path)
    for filename in list:

        filename_open = file_path + filename

        im = cv2.imread(filename_open)

        img = Image.open(filename_open)  # convert("P")
        img_data = np.array(img)
        print(img_data)
        print("----------------")
        x1 = img_data == 255
        print(max(img_data[87]))
        print(img_data[87])

        categories1 = np.unique(img_data)     #225修改为0
        img_data[img_data == 255] = 0

        filename_out = file_outpath + filename


        cv2.imwrite(filename_out, label2color(img_data))

change()