# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import rospy
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Int16, Float32
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
from functools import partial
tf.disable_v2_behavior()

# csvを読み込む
with open('./../csv/acc.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    l_csv = [row for row in reader]

time_array = [float(l_csv[i][0]) for i in range(len(l_csv))]
z_acc_array = [float(l_csv[i][3]) for i in range(len(l_csv))]

def makeCNN(X, training):
    he_init = tf.variance_scaling_initializer()
    my_batch_norm_layer = partial(tf.layers.batch_normalization,
                                  training=training,
                                  momentum=0.9)
    my_conv2d_layer = partial(tf.layers.conv2d, kernel_initializer=he_init)
    my_dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)
    my_dconv2d_layer = partial(tf.layers.conv2d_transpose, kernel_initializer=he_init)

    conv1 = my_conv2d_layer(X, filters=32, kernel_size=(20,20), strides=(2,2), padding='valid')
    bn1 = tf.nn.relu(my_batch_norm_layer(conv1))
    conv2 = my_conv2d_layer(bn1, filters=32, kernel_size=(20,20), strides=(2,2), padding='valid')
    bn2 = tf.nn.relu(my_batch_norm_layer(conv2))
    conv3 = my_conv2d_layer(bn2, filters=32, kernel_size=(4,4), strides=(2,2), padding='valid')
    bn3 = tf.nn.relu(my_batch_norm_layer(conv3))
    conv4 = my_conv2d_layer(bn3, filters=32, kernel_size=(4,4), strides=(2,2), padding='valid')
    bn4 = tf.nn.relu(my_batch_norm_layer(conv4))
    bn4_flat = tf.reshape(bn4, shape=[-1, 4*4*32])
    fc1 = my_dense_layer(bn4_flat, 3)
    lbn1 = tf.nn.sigmoid(my_batch_norm_layer(fc1))
    fc2 = my_dense_layer(lbn1, 8)
    lbn2 = tf.nn.relu(my_batch_norm_layer(fc2))
    lbn2_flat = tf.reshape(lbn2, shape=[-1, 1, 8, 1])
    dconv1 = my_dconv2d_layer(lbn2_flat, filters=32, kernel_size=(1,5), strides=(3,2), padding='same')
    dbn1 = tf.nn.relu(my_batch_norm_layer(dconv1))
    dconv2 = my_dconv2d_layer(dbn1, filters=32, kernel_size=(2,5), strides=(1,2), padding='same')
    dbn2 = tf.nn.relu(my_batch_norm_layer(dconv2))
    dconv3 = my_dconv2d_layer(dbn2, filters=32, kernel_size=(2,6), strides=(1,2), padding='same')
    dbn3 = tf.nn.relu(my_batch_norm_layer(dconv3))
    dconv4 = my_dconv2d_layer(dbn3, filters=1, kernel_size=(2,6), strides=(1,2), padding='same')
    outputs = tf.nn.sigmoid(dconv4)

    return lbn1,outputs


class RosTensorFlow():
    def __init__(self):
        rospy.init_node("RosTensorFlow", anonymous=True)
        self.cv_bridge = CvBridge()
        self.training = tf.placeholder_with_default(False, shape=(), name='training')
        self.X = tf.placeholder(tf.float32, shape=[None, 150, 150, 1], name='X')
        self.lbn1,self.outputs = makeCNN(self.X, self.training)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.saver.restore(self.sess, "./../jupyter/model/20201119_22:40.ckpt")

        self.img_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback_img)
        self.img_pub_3channel = rospy.Publisher("/image_crop_3channel", Image, queue_size=10)
        self.img_pub = rospy.Publisher("/image_crop_1channel", Image, queue_size=10)
        self.img_pub_color = rospy.Publisher("/display_color", Image, queue_size=10)
        self.x_latent_pub = rospy.Publisher("/x_latent", Float32, queue_size=10)
        self.y_latent_pub = rospy.Publisher("/y_latent", Float32, queue_size=10)
        self.z_latent_pub = rospy.Publisher("/z_latent", Float32, queue_size=10)

        # tf
        self.tf_sub = rospy.Subscriber("/tf", TFMessage, self.callback_tf)
        self.x = 0.00
        self.y = 0.00
        self.received_once = False # まず，座標を保存，それまではカラーは保存しない

        self.N_dft = 512
        self.z = []
        self.count = 0
        self.past_index = 0
        self.num = 0
        self.rostime = 0.00

        self.cv_img = np.zeros((100,100,3), np.uint8)
        self.cv_img.fill(255)
        self.img_color = self.cv_bridge.cv2_to_imgmsg(self.cv_img, encoding="bgr8")

        self.fig = plt.figure(figsize=(7,6))
        self.ax_dft = self.fig.add_subplot(2,1,1)
        self.ax_dft.set_title("The DFT's Measurement and Prediction")
        self.ax_dft.set_ylim(0.00,0.20)
        self.ax_dft.set_xlabel('$Frequency$')
        self.ax_dft.set_ylabel('$Amplitude$')
        self.ax_scatter1 = self.fig.add_subplot(2,2,3)
        self.ax_scatter1.set_title("The Latent Space Of X-Y")
        self.ax_scatter1.set_xlim(0.00,1.00)
        self.ax_scatter1.set_ylim(0.00,1.00)
        self.ax_scatter1.set_xlabel('$X$')
        self.ax_scatter1.set_ylabel('$Y$')
        self.ax_scatter2 = self.fig.add_subplot(2,2,4)
        self.ax_scatter2.set_title("The Latent Space Of Z-Y")
        self.ax_scatter2.set_xlim(0.00,1.00)
        self.ax_scatter2.set_ylim(0.00,1.00)
        self.ax_scatter2.set_xlabel('$Z$')
        self.ax_scatter2.set_ylabel('$Y$')

        self.fq = np.linspace(0,255,self.N_dft)[129:int(self.N_dft/2)+1] # [129:257]計128個
        self.line1 = self.ax_dft.plot(self.fq,np.zeros(int(self.N_dft/4)),animated=True,c="r",label="Prediction")[0]
        self.line2 = self.ax_dft.plot(self.fq,np.zeros(int(self.N_dft/4)),animated=True,c="b",label="Measurement")[0]
        self.ax_dft.legend(loc='upper right')

        plt.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()
        self.axes = [self.ax_dft, self.ax_scatter1, self.ax_scatter2]
        self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]

        with open('x_y_color.csv','w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['x', 'y', 'color'])

    def plot(self, pred, measure, x, y, z):
        for i, (ax, background) in enumerate(zip(self.axes, self.backgrounds)):
            self.fig.canvas.restore_region(background)
            if i == 0:
                self.line1.set_ydata(pred)
                self.line2.set_ydata(measure)
                ax.draw_artist(self.line1)
                ax.draw_artist(self.line2)
            elif i == 1:
                ax.draw_artist(ax.scatter(x,y,c='blue'))
            elif i == 2:
                ax.draw_artist(ax.scatter(z,y,c='blue'))
            self.fig.canvas.blit(ax.bbox)


    def display_color(self, x, y, z):
        self.cv_img[:,:,0] = int(z*255)
        self.cv_img[:,:,1] = int(x*255)
        self.cv_img[:,:,2] = int(y*255)
        self.img_color = self.cv_bridge.cv2_to_imgmsg(self.cv_img, encoding="bgr8")
        self.img_pub_color.publish(self.img_color)

    def csv_preserve(self,x,y,z):
        R = int((255/(0.70-0.10))*(x-0.10))
        G = int((255/(0.50-0.10))*(y-0.10))
        B = int((255/(1.00-0.70))*(z-0.70))
        print("R: ",R)
        print("G: ",G)
        print("B: ",B)
        # R = int(x*255)
        # G = int(y*255)
        # B = int(z*255)
        line = [self.x, self.y, "rgb("+str(R)+","+str(G)+","+str(B)+")"]
        with open('x_y_color.csv','a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(line)

    def dft_cal(self, array):
        F = np.fft.fft(array)
        F_abs = np.abs(F)
        F_abs_amp = F_abs / self.N_dft * 2
        F_abs_amp[0] = F_abs_amp[0] / 2
        return F_abs_amp


    def find_index(self, lst, target, start):  # listからtargetを探す．このときインデックスを入力
        for i,x in enumerate(lst[start:],start):
            if float(x) >= float(target):
                index = i
                # print("x:{0}, target:{1}".format(x,target))
                break
            else:
                index = None
        return index

    def callback_tf(self, msg):
        if msg.transforms[0].header.frame_id == "world":
            self.x = msg.transforms[0].transform.translation.x
            self.y = msg.transforms[0].transform.translation.y
            self.received_once = True


    def callback_img(self, data):
        self.rostime = round(data.header.stamp.to_sec(),2)
        index = self.find_index(time_array, self.rostime, self.past_index)
        self.count += 1

        if index+self.N_dft <= len(l_csv)-1:
            self.img_pub_3channel.publish(data)
            cv_img = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
            cv_img = cv_img[165-35:315-35,245:395]
            self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_img, encoding="mono8"))

            cv_img = np.array(cv_img).astype('float32')/255
            cv_img = cv_img[np.newaxis,:,:,np.newaxis]

            # Fourier_transform
            Fz = np.fft.fft(z_acc_array[index:index+self.N_dft])
            Fz_abs = np.abs(Fz)
            Fz_abs_amp = Fz_abs / self.N_dft * 2
            Fz_abs_amp[0] = Fz_abs_amp[0] / 2
            spec = Fz_abs_amp[129:int(self.N_dft/2)+1]

            outputs_infer = self.sess.run(self.outputs,feed_dict={self.X: cv_img})

            lbn1_infer = self.sess.run(self.lbn1,feed_dict={self.X: cv_img})
            lbn1_infer_x = lbn1_infer[:,0]
            self.x_latent_pub.publish(lbn1_infer_x)
            lbn1_infer_y = lbn1_infer[:,1]
            self.y_latent_pub.publish(lbn1_infer_y)
            lbn1_infer_z = lbn1_infer[:,2]
            self.z_latent_pub.publish(lbn1_infer_z)
            # print("{0},{1},{2}".format(lbn1_infer_x,lbn1_infer_y,lbn1_infer_z))

            outputs_infer_z = outputs_infer[0,2,:,0]
            outputs_infer_z = outputs_infer_z*0.33382421481914376 + 1.1718749999722888e-05
            self.display_color(lbn1_infer_x, lbn1_infer_y, lbn1_infer_z)
            self.plot(outputs_infer_z, spec, lbn1_infer_x, lbn1_infer_y, lbn1_infer_z)

            if self.received_once:
                self.csv_preserve(lbn1_infer_x, lbn1_infer_y, lbn1_infer_z)


if __name__ == '__main__':
    rtf = RosTensorFlow()
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("!!Shutting Down!!")
