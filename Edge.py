from PIL import Image
import numpy as np


def edge_pixel(image_name, pred_flux):
    pred_flux = pred_flux.cpu()
    print("-------pred_flux--------")
    print(pred_flux)
    # print(pred_flux.shape)
    # 取得像素到最近边界像素的距离，为一个1,375,500的数组
    distance = pred_flux.norm(p=2, dim=1) + 1e-9
    print("-------distance--------")
    print(distance)
    # print(distance.shape)
    # image_name = image_name[0]

    _, _, height, width = pred_flux.shape

    test = [[0 for i in range(width)] for i in range(height)]  # 初始化一个维度为500*375的二维数组
    c = Image.new("RGB", (width, height))   # 初始化一个维度为长宽的黑色画布
    with open('location.txt', 'w') as file:
        for i in distance:      # 取得一个375,500的数组
            i_num = 0     # 初始化i_num为0,用来计数（height）
            for j in i:  # 取得一个375的数组
                j_num = 0  # 初始化j_num为0 用来计数（width）
                for k in j:  # 取得500单个元素
                    test[i_num][j_num] = k  # 将第i_num行，第j_num列元素存入数组
                    if k < 0.3:  # 如果为边界像素，在c上进行描点，颜色为白色
                        file.write(str(i_num) + '-' + str(j_num) + '\n')
                        # print("边缘像素的位置坐标为：", (str(i_num), str(j_num)))
                        c.putpixel([j_num, i_num], (255, 0, 0))
                    j_num += 1
                i_num += 1

    # for i in range(len(test)):
    #     for j in range(len(test[i])):
    #         if test[i][j] == 1:
    #             print("边缘像素的位置坐标为：", (str(i), str(j)))

    
    c.show()
    np.savetxt('1.txt', test)
    c.save("images/my_images/" + str(image_name) + "_Edge.png")
