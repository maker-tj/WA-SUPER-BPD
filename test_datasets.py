import cv2
import numpy as np
import os.path as osp
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.io as sio
from torch.utils.data import Dataset
import os

IMAGE_MEAN = np.array([103.939, 116.779, 123.675], dtype=np.float32)


def watershed(image):
    img = image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # 消除噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # 膨胀
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # 距离变换
    dist_transform = cv.distanceTransform(opening, 1, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 获得未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # 标记
    ret, markers1 = cv.connectedComponents(sure_fg)

    # 确保背景是1不是0
    markers = markers1 + 1

    # 未知区域标记为0
    markers[unknown == 255] = 0

    markers3 = cv.watershed(img, markers)
    img[markers3 == -1] = [0, 0, 255]

    return img

class FluxSegmentationDataset(Dataset):

    def __init__(self, dataset='PascalContext', mode='train'):
        self.dataset = dataset
        self.mode = mode
        print(self.mode)
        # 这里读取pascal context的train.txt文本
        # file_dir = 'images/' + self.dataset + '/' + self.mode + '.txt'
        file_dir = 'images/' + self.dataset + '/test.txt'


        self.random_flip = False

        if self.dataset == 'PascalContext' and mode == 'train':
            self.random_flip = False

        with open(file_dir, 'r') as f:  # 以r方式打开表示只读，以w方式打开表示只写
            self.image_names = f.read().splitlines()  # 一行行的读取图片名称


        self.dataset_length = len(self.image_names)

    def __len__(self):  # 获取数据集长度

        return self.dataset_length

    def __getitem__(self, index):  # 根据索引序号获取图片和标签

        random_int = np.random.randint(0, 2)  # 输出[0，2）之间的整数

        image_name = self.image_names[index]

        image_path = osp.join('D:\\A-work\\our_SuperBPD\\images', self.dataset, 'train', image_name)  # 原图路径

        image = cv2.imread(image_path, 1)
        image = watershed(image)

        if self.random_flip:
            if random_int:
                image = cv2.flip(image, 1)

        vis_image = image.copy()

        height, width = image.shape[:2]
        image = image.astype(np.float32)
        image -= IMAGE_MEAN
        # print("读取图片的高是：", height)
        # print("读取图片的宽是：", width)
        image = image.transpose(2, 0, 1)
        # print("图像尺寸是：",image.shape)

        if self.dataset == 'BSDS500':
            image_name = image_name.split('/')[-1]

        return image, vis_image, self.dataset_length, image_name




