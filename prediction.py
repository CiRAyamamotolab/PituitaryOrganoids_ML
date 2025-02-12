#prediction
###Python###
#import
from __future__ import absolute_import, division, print_function, unicode_literals 
import tensorflow as tf
import os
import numpy as np
import glob
import pandas as pd
import shutil
import random
import cv2
import sys
import time
import matplotlib.pyplot as plt
from tensorflow.keras import models, optimizers, layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import VGG16, InceptionResNetV2, VGG19, Xception, InceptionV3, MobileNet, DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.preprocessing import image as images
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

###settings#############################################
dir="***"
os.chdir(dir)
print(os.getcwd())
day=15#select from [3, 9, 15, 21, 27]
image_size=300
trim_size=300
cond=[str(day), "VGG16"]#select from [Xception, VGG16, VGG19, Resnet50, InceptionV3, InceptionResnetV2, Mobilenet, Densenet121, Densenet169, Densenet201]
d=0.4#coefficient for drop out layer
########################################################

#standardizing of learning images
def image_standerdize(x, dir_name, save_name):
  for i in range(len(x)):
    #Gray scale
    img_bgr = cv2.imread(x[i])
    gamma22LUT  = np.array([pow(x/255.0 , 2.2) * 255 for x in range(256)], dtype='uint8')
    gamma045LUT = np.array([pow(x/255.0 , 1.0/2.2) * 255 for x in range(256)], dtype='uint8')
    img_bgrL = cv2.LUT(img_bgr, gamma22LUT)  # sRGB => linear (approximate value 2.2)
    img_grayL = cv2.cvtColor(img_bgrL, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.LUT(img_grayL, gamma045LUT)  # linear => sRGB
    #resize/trimming
    h, w = img_gray.shape[:2]
    if h==2048:
      img_gray=img_gray[440:1608, 440:1608]
    elif h==1200:#kyoto2
      img_gray=img_gray[128:1072, 328:1272]
    else:
      print("error:"+str(i))
    #resize
    img_gray_resized = cv2.resize(img_gray,(image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
    #trimming
    img_gray_resized_trim=img_gray_resized[(image_size-trim_size)//2:(image_size+trim_size)//2,(image_size-trim_size)//2:(image_size+trim_size)//2]
    #intensity averaging
    img_gray_st = (img_gray_resized_trim - np.mean(img_gray_resized_trim))/np.std(img_gray_resized_trim)*32+128 #average=128, SD=32
    #save image
    cv2.imwrite(os.path.join(dir_name ,save_name+'_'+str(i).zfill(5)+'.jpg'), img_gray_st)

#loading model
conv_base = VGG16(weights = "imagenet",
                include_top=False,
                input_shape=(trim_size,trim_size,3))
conv_base.summary()
last = conv_base.output
mod = Flatten()(last)
mod = Dense(128, activation='relu')(mod)
mod = Dropout(d)(mod)
mod = Dense(8, activation='relu')(mod)
mod = Dropout(d)(mod)
preds = Dense(2, activation='softmax')(mod)
model = models.Model(conv_base.input, preds)
model.summary()
conv_base.trainable = True
for layer in conv_base.layers[:15]:
    print(layer)
    layer.trainable = False
#loading constructed models
#day3
if day==3:
    model.load_weights("model_VGG16_day3.h5")#d3
#d9
if day==9:
    model.load_weights("model_VGG16_day9.h5")#d9
#d15
if day==15:
    model.load_weights("model_VGG16_day15.h5")#d15
#21
if day==21:
    model.load_weights("model_VGG16_day21.h5")#d21
#27
if day==27:
    model.load_weights("model_VGG16_day27.h5")#d27
model.summary()
#preprocessing the data
law_data_dir="law_data_prediction_"+str(day)
output_dir1="law_data_prediction_standardize"
output_dir2 = "law_data_prediction_standardize/unknown"
if not(os.path.exists(output_dir1)):
    os.mkdir(output_dir1)
if not(os.path.exists(output_dir2)):
    os.mkdir(output_dir2)
#loading the data
prediction_images = glob.glob(os.path.join(law_data_dir,"*tif"))
prediction_images.sort()
print(prediction_images)
print("prediction_images:"+str(len(prediction_images)))
#preproicessing the data
image_standerdize(prediction_images, output_dir2, "unknown")
test_data_files = sorted(glob.glob(os.path.join(output_dir1,"*","*")))
test_data_count = len(test_data_files)
print("test_data:" + str(test_data_count))
nb_test_samples = test_data_count #number of the data
batch_size = 1 
nb_category = 2 #（fail,suc）
# imagedatagenerator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        output_dir1,
        target_size=(trim_size, trim_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
# prediction
pred = model.predict_generator(
        test_generator,
        steps=nb_test_samples,
        verbose=1)
labels = ['fail', 'suc']
#result
print(prediction_images)
print("*** prediction data *****")
suc=0
fail=0
for i in range(test_data_count):
    cls = np.argmax(pred[i])
    score = np.max(pred[i])
    if pred[i][0]>pred[i][1]:
        fail=fail+1
    if pred[i][1]>pred[i][0]:
        suc=suc+1
    print(os.path.basename(test_data_files[i])+": {}  score = {:.3f}".format(labels[cls], score))
#number of fail and suc prediction
print("suc:"+str(suc)+"/"+str(suc+fail))
print("fail:"+str(fail)+"/"+str(suc+fail))
