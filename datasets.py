import cv2
import numpy as np
import os.path as osp
import scipy.io as sio
from torch.utils.data import Dataset
import os

IMAGE_MEAN = np.array([103.939, 116.779, 123.675], dtype=np.float32)


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

        # 读取训练数据集的label
        if self.dataset == 'PascalContext':
            label_path = osp.join('label', self.dataset, 'train', image_name)
            #print(label)
            #label = sio.loadmat(label_path)['LabelMap']
            label = cv2.imread(label_path, 0)


        elif self.dataset == 'BSDS500':
            label_path = osp.join('label', self.dataset, 'groundTruth', 'train', image_name)
            label = cv2.imread(label_path, 0)
            print("label_path:",label_path)
            #label = sio.loadmat(label_path)['LabelMap']

        elif self.dataset == 'cityscapes':
            label_path = osp.join('label', 'cityscapes/train', image_name)
            # print(label_path)
            label = cv2.imread(label_path, 0)
            # print(type(label))

        if self.random_flip:
            if random_int:
                label = cv2.flip(label, 1)
        # cv2.flip(filename, flipcode) filename：需要操作的图像 flipcode：翻转方式 1:水平翻转 0:垂直翻转 -1:水平垂直翻转

        label += 1

        gt_mask = label.astype(np.float32)

        categories = np.unique(label)

        # if 0 in categories:
        #     raise RuntimeError('invalid category')

        label = cv2.copyMakeBorder(label, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        weight_matrix = np.zeros((height + 2, width + 2), dtype=np.float32)
        direction_field = np.zeros((2, height + 2, width + 2), dtype=np.float32)

        for category in categories:
            img = (label == category).astype(np.uint8)
            weight_matrix[img > 0] = 1. / np.sqrt(img.sum())

            _, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                        labelType=cv2.DIST_LABEL_PIXEL)

            index = np.copy(labels)
            index[img > 0] = 0  # 目标区域
            place = np.argwhere(index > 0)

            nearCord = place[labels - 1, :]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, height + 2, width + 2))
            nearPixel[0, :, :] = x
            nearPixel[1, :, :] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel

            direction_field[:, img > 0] = diff[:, img > 0]

        weight_matrix = weight_matrix[1:-1, 1:-1]
        direction_field = direction_field[:, 1:-1, 1:-1]

        if self.dataset == 'BSDS500':
            image_name = image_name.split('/')[-1]

        return image, vis_image, gt_mask, direction_field, weight_matrix, self.dataset_length, image_name




