import cv2 as cv2
import numpy as np


def watershed_demo():
    # print(src.shape)
    # 第一步： 边缘保留 滤波去噪
    blurred = cv2.pyrMeanShiftFiltering(image, 10, 100) #   src彩色图像
    # 引入了高斯核的均值漂移有滤波作用
    # 对图像进行均值漂移滤波，去除了精细纹理、颜色梯度大部分变得平坦，达到去噪效果
    # 第二步：转化为灰度图
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # 第三步：二值化
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow("binary-image", binary)  # 二值化图像

    # 第三步：morphology operation形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 构造结构
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # 连续两次开操作腐蚀（去除图像中的任何小的白噪声）;闭运算膨胀（为了去除物体上的小洞）
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 连续三次闭操作膨胀
    #cv2.imshow("mor-opt", sure_bg)   # 形态学运算图像

    # 第四步：distance transform距离变换（可以将相互连接的目标边界分割出来）
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3) # 距离变化（提取出我们确信它们是硬币的区域，提取目标区域）
    dist_output = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX) # 归一化
    dist_output1 = np.uint8(dist_output)

    # 第五步：寻找种子
    ret, surface = cv2.threshold(dist, dist.max()*0.6, 255, cv2.THRESH_BINARY)

    surface_fg = np.uint8(surface) # 将目标float类型转化为uint
    #cv2.imshow("surface-bin", surface_fg)
    unknown = cv2.subtract(sure_bg, surface_fg)  # 背景区域-前景区域=未知区域
    # 除种子以外的区域（剩下的区域是我们不知道的区域，无论是硬币还是背景.分水岭算法应该找到它）

    # 第六步：生成marker
    ret, markers = cv2.connectedComponents(surface_fg)
    # 求连通区域（创建标记：它是一个与原始图像相同大小的数组，但使用int32数据类型，并对其内部的区域进行标记.）

    # 第七步：watershed transform 分水岭变换
    markers = markers + 1  # Add one to all labels so that sure background is not 0, but 1 （目标区域标记）
    markers[unknown==255] = 0  #  mark the region of unknown with zero标记未知区域为0，为背景
    markers = cv2.watershed(image, markers=markers)
    image[markers==-1] = [120, 20, 0] # 标记边缘
    #cv2.imshow("result", image)



image = cv2.imread("images/PascalContext/train/2011_001810.jpg")
cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
#cv2.imshow("input image", image)
watershed_demo()
cv2.imwrite('images/PascalContext/train/2011_001810' + '.png', image)
cv2.waitKey(0)

cv2.destroyAllWindows()
