# coding:utf-8

# あるrostimeにて，そのときの加速度と画像を保存
# このとき，csv/acc.csvとbag/autonomous_series.bagを用いる

from __future__ import print_function
import numpy as np
import cv2
import sys
import rospy
from PIL import Image
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import argparse
import math
import time
import pickle
import decimal
import csv
from cv_bridge import CvBridge, CvBridgeError
import os

# csvを読み込む
with open('./../csv/acc.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    l_csv = [row for row in reader] #

time_array = [float(l_csv[i][0]) for i in range(len(l_csv))]
x_acc_array = [float(l_csv[i][1]) for i in range(len(l_csv))]
y_acc_array = [float(l_csv[i][2]) for i in range(len(l_csv))]
z_acc_array = [float(l_csv[i][3]) for i in range(len(l_csv))]
# print(l_csv[0]) # ['rostime(小数２桁)','x_acc','y_acc','z_acc']

class csv2data:
    def __init__(self):
        rospy.init_node("csv2data", anonymous=True)
        self.img_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback_img)
        self.past_index = 0
        self.num = 0
        self.rostime = 0.000
        self.N = 512
        self.count = 0

    def find_index(self, lst, target, start):  # listからtargetを探す. このときインデックスを入力
        for i,x in enumerate(lst[start:],start):
            if x >= target:
                index = i
                print("index: {0}, x: {1}, target: {2}".format(index, x, target))
                break
            else:
                index = None
        return index

    def callback_img(self, msg):
        self.rostime = round(msg.header.stamp.to_sec(),3)
        # msgの時刻と同様程度のそれをcsvから探す

        # print('self.rostime: ', self.rostime)
        index = self.find_index(time_array, self.rostime, self.past_index)
        self.past_index = index

        PNG_DIR = './../data/img'

        if not os.path.exists(PNG_DIR):
            os.makedirs(PNG_DIR)

        cv_img = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        filename = PNG_DIR+'/'+str(round(msg.header.stamp.to_sec(),3))+'_'+str(index)+'.png'
        cv2.imwrite(filename, cv_img)


if __name__ == '__main__':
    c2d = csv2data()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
        cv2.destroyAllWindows()
