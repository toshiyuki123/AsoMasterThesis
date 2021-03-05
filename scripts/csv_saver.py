# coding:utf-8

## from rosbag, save the accerelation topic and odometry topic
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt
import numpy as np
import rospy
import os
import time
import csv
import argparse
import math

parser = argparse.ArgumentParser()

# parser.add_argument('--num', type=str, default="04")
parser.add_argument('--acc_topic', type=str, default="/camera/accel/sample")
# parser.add_argument('--odom_topic', type=str, default="/ypspur_ros/odom")

args = parser.parse_args()

place = './../csv/'
# num = args.num

if os.path.exists(place) == False:
    os.mkdir(place)

# # csvファイルの存在確認: 作られていたら消去
# if os.path.exists(place+'/'+num+'.csv') == True:
#     os.remove(place+'/'+num+'.csv')
#     print("!!eliminating the past csv!!")

with open(place+'acc.csv', 'w') as f:
    print('!!start writing!!')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['rostime', 'x_acc', 'y_acc', 'z_acc'])

class csv_saver_node:
    def __init__(self):
        rospy.init_node('csv_saver_node', anonymous=True)
        self.acc_sub = rospy.Subscriber(args.acc_topic, Imu, self.callback_acc)


    def callback_acc(self, msg):
        print('!!add the data!!')
        # [rostime, acc_x, acc_y, acc_z, odom_x]
        line = [round(msg.header.stamp.to_sec(),3), round(msg.linear_acceleration.x,3),\
                round(msg.linear_acceleration.y,3), round(msg.linear_acceleration.z,3)]
        with open(place+'acc.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(line)


if __name__ == '__main__':
    csn = csv_saver_node()
    try:
        rospy.spin()
    except KeyboardInterruupt:
        print("Shutting Down")
        cv2.destroyAllWindows()
