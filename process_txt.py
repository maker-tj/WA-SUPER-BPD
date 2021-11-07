from pylab import *
import matplotlib.pyplot as plt
import sys
import os
import glob
from random import shuffle

#数据集，其中train_n.txt中只包含70%正常样本集，train_a.txt中包含异常样本，test.txt中包含数量相当的正常异常样本
def writeAbnormal ():


   print("Writing files for user")
   n_dirs = os.listdir("D:/data/images/train");
   shuffle(n_dirs)
   ab_dirs = os.listdir("D:/data/images/test");
   shuffle(ab_dirs)

   #先写训练集
   text_file = open("train_n.txt", "w")
   for x in range(0, int(floor(size(n_dirs))*0.7)):
      text_file.write("%s/%s \n" % ("D:/data/images/train", n_dirs[x]))
   text_file.close()


   #再写测试集
   text_file = open("test_n.txt", "w")
   for x in range(int(floor(size(ab_dirs))*0.7),int(size(n_dirs))):
      text_file.write("%s/%s \n" % ("D:/data/images/test", n_dirs[x]))
   text_file.close()


writeAbnormal ()

