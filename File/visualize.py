#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:28:58 2017

@author: wzg
"""

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import scipy.io as sio
import glob,os
import TensorflowUtils as utils
import run as runs
from six.moves import xrange
import scipy.misc as misc
from scipy.misc import imresize
from PIL import Image

import time

root_path = './Mydata/'


height = 300
width = 300

ckpt_path = './Model/MTV4_model.ckpt'

meanfile = sio.loadmat('./Data/mats/mean300.mat')
meanvalue = meanfile['mean'] #mean value of images in training set
FLAGS = tf.flags.FLAGS

keep_probability = tf.placeholder(tf.float32, name = "keep_probabilty")
image = tf.placeholder(tf.float32, shape = [None, height, width, 3], name = "input_image")
pred_annotation, logits, pred_label = runs.inference(image, keep_probability)

    

def pre_mat(img_path,sava_path):
    saver = tf.train.Saver()
    print("a---------------------")
    with tf.Session(graph=tf.get_default_graph()) as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
            sess.run(init_op)
                    
            saver.restore(sess, ckpt_path)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            file_name = os.path.basename(img_path)
            file_name = os.path.splitext(os.path.basename(file_name))[0]
            test_img = np.float64(misc.imread(img_path))
            test_img = test_img - meanvalue
            test_img = test_img[np.newaxis, :, :]    
    
            pred = sess.run(pred_annotation, feed_dict = {image: test_img, keep_probability: 1.0})
            pred = np.squeeze(pred, axis = 3)
            sio.savemat(sava_path + file_name +'.mat', {'mask':pred[0].astype(np.uint8)})
            print(sava_path + file_name +'.mat')


# for file in vis_files:
#     file_path = os.path.basename(file)
#     file_name = os.path.splitext(os.path.basename(file_path))[0]
    
#     maskmat = sio.loadmat(file) #*
#     mask = np.float64(maskmat['mask'])#*
#     reconstructed_mask = mask.reshape((height,width)) #*
#     fig = plt.figure()
#     plt.imsave(root_path + "/pre_img/" + file_name +".jpg" ,np.uint8(reconstructed_mask))
#     plt.close()
#     print(root_path + "/pre_img/" + file_name +".jpg")

# -----------------------------------------------------
# with tf.Session(graph=tf.get_default_graph()) as sess:
#         init_op = tf.group(tf.global_variables_initializer(),
#                                tf.local_variables_initializer())
#         sess.run(init_op)
                
#         saver.restore(sess, ckpt_path)
        
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)

#         test_img = np.float64(misc.imread("./Mydata/resize_img/ドラレコ糸数那覇 13.jpg"))
#         test_img = test_img - meanvalue
#         test_img = test_img[np.newaxis, :, :]    

#         pred = sess.run(pred_annotation, feed_dict = {image: test_img, keep_probability: 1.0})
#         pred = np.squeeze(pred, axis = 3)
#         sio.savemat("./Mydata/pre_mat/" + 'pred_test.mat', {'mask':pred[0].astype(np.uint8)})


def loadmat(path):
    maskmat = sio.loadmat(path)
    mask = np.float64(maskmat['mask'])
    mask = np.uint8(mask.reshape((height,width)))
    return mask

def save_preimg(matfile,save_path):
    reconstructed_mask = [[[0]*3]*300]*300
    reconstructed_mask = np.array(reconstructed_mask)

    file_name = os.path.basename(matfile)
    file_name = os.path.splitext(os.path.basename(file_name))[0]
    
    mat = loadmat(matfile)
    
    for i,h in enumerate(mat):
        for j,label in enumerate(h):
            # print(label)
            # time.sleep(1)
            if label == 1:
                reconstructed_mask[i][j] = [0,255,255] #bule sky 
            elif label == 2:
                reconstructed_mask[i][j] = [224,255,255] #white cloud
            elif label == 3:
                reconstructed_mask[i][j] = [255,255,0] #shadow (yellow)
            elif label == 4:
                reconstructed_mask[i][j] = [128,128,128] #gray sky
            elif label == 5:
                reconstructed_mask[i][j] = [0,100,0] #black cloud (green)
    Image.fromarray(reconstructed_mask.astype(np.uint8)).save(save_path+file_name+".jpg")


files = glob.glob(root_path+"pre_mat/*")
print(files)
for file in files:
    print(file)
    save_preimg(file,root_path+"pre_img/")

