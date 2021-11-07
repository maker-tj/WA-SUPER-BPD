import os

# 删除没有label对应的原图像，让原图像和label图一一对应

def delete_Img():
    path_image = r'D:\data\BSR\BSDS500\data\images\test'
    path_label = r'D:\data\BSR\BSDS500\data\groundTruth\test'

    file_image = os.listdir(path_image)

    for i in file_image:
        if (not os.path.exists(os.path.join(path_label, i[:-4] + '.mat'))):
            os.remove(os.path.join(path_image, i))

delete_Img()

