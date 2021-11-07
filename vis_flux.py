import sys
import scipy.io as sio
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import pylab as plt
from matplotlib import cm
import torch


def label2color(label):   # 定义颜色标签

    label = label.astype(np.uint16)  # astype生成的矩阵是无符号整型
    
    height, width = label.shape
    color3u = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(label)   # 除去重复标签并进行排序之后输出的标签

    if unique_labels[-1] >= 2**24:     # 标签数组的最后一位大于2**24为溢出
        raise RuntimeError('Error: label overflow!')   # 标签溢出

    for i in range(len(unique_labels)):  # 遍历每一个标签
    
        binary = '{:024b}'.format(unique_labels[i])  # 格式限定符{：b}表示二进制
        # r g b 3*8=24   2**8*3=256*3
        # 切片提取每一个标签里面的rgb，并将二进制转化为十进制
        r = int(binary[::3][::-1], 2)  # 每三个值为一个切片（三个值分别为rgb），从每个切片的第1个值开始取，从后往前排列
        g = int(binary[1::3][::-1], 2)  # 每三个值为一个切片，从每个切片的第2个值开始取，从后往前排列
        b = int(binary[2::3][::-1], 2)   # 每三个值为一个切片，从每个切片的第3个值开始取，从后往前排列

        color3u[label == unique_labels[i]] = np.array([r, g, b])

    return color3u


def vis_flux(vis_image, pred_flux, gt_flux, gt_mask, image_name, save_dir):
    # .data.cpu().numpy()：将tensor转换成numpy的格式
    vis_image = vis_image.data.cpu().numpy()[0, ...]  # 可视化图像
    pred_flux = pred_flux.data.cpu().numpy()[0, ...]   # 通过model预测的bpd

    # 对真实BPD进行归一化保证norm中间图相似
    gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1) + 1e-9)
    # np.set_printoptions(suppress=True)

    gt_flux = gt_flux.data.cpu().numpy()[0, ...]   # 真实的bpd
    gt_mask = gt_mask.data.cpu().numpy()[0, ...]  # 掩膜
    
    image_name = image_name[0]
    # print(image_name)

    norm_pred = np.sqrt(pred_flux[1,:,:]**2 + pred_flux[0,:,:]**2)   # 预测L2范数距离
    angle_pred = 180/math.pi*np.arctan2(pred_flux[1,:,:], pred_flux[0,:,:])  # 两个像素点的预测BPD的夹角

    # print(torch.from_numpy(np.sqrt(gt_flux[1,:,:]**2 + gt_flux[0,:,:]**2)).norm(p=2, dim=1).shape)
    # print(np.sqrt(gt_flux[1,:,:]**2 + gt_flux[0,:,:]**2).shape)
    norm_gt = np.sqrt(gt_flux[1, :, :] ** 2 + gt_flux[0, :, :] ** 2)
    angle_gt = 180/math.pi*np.arctan2(gt_flux[1,:,:], gt_flux[0,:,:])  # 真实角度

    fig = plt.figure(figsize=(10,6))

    ax0 = fig.add_subplot(231)
    ax0.imshow(vis_image[:,:,::-1])

    ax2 = fig.add_subplot(233)
    ax2.set_title('Angle_gt')
    ax2.set_autoscale_on(True)
    im2 = ax2.imshow(angle_gt, cmap=cm.jet)
    plt.colorbar(im2, shrink=0.5)

    ax3 = fig.add_subplot(234)
    color_mask = label2color(gt_mask)
    ax3.imshow(color_mask)

    ax4 = fig.add_subplot(235)
    ax4.set_title('Norm_pred')
    ax4.set_autoscale_on(True)
    im4 = ax4.imshow(norm_pred, cmap=cm.jet)
    plt.colorbar(im4,shrink=0.5)

    ax1 = fig.add_subplot(232)
    ax1.set_title('Norm_gt')
    ax1.set_autoscale_on(True)  # 在绘图命令上应用自动缩放
    im1 = ax1.imshow(norm_gt, cmap=cm.jet)
    # print(norm_gt)
    plt.colorbar(im4, shrink=0.5)  # 添加颜色渐变条

    ax5 = fig.add_subplot(236)
    ax5.set_title('Angle_pred')
    ax5.set_autoscale_on(True)
    im5 = ax5.imshow(angle_pred, cmap=cm.jet)
    plt.colorbar(im5, shrink=0.5)

    # plt.savefig(save_dir + image_name + '.png')  # 将图片存入固定路径
    plt.savefig('D:\\A-work\\our_SuperBPD\\images\\my_images'+ image_name + '.png')
    plt.close(fig)  # 关闭图像窗口

# label2color(np.array([[255,255,255],[255,255,255]]))
