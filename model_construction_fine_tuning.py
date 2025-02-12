#fine tuning model construction
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
cond=["VGG16"]#select from [Xception, VGG16, VGG19, Resnet50, InceptionV3, InceptionResnetV2, Mobilenet, Densenet121, Densenet169, Densenet201]
d=0.4#coefficient for drop out layer
batch_size=64
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

#preparing learning data################################################################
#making data dirs
output_dir1 = "d"+str(day)+"_"+str(image_size)+"_"+"standardize"
if not(os.path.exists(output_dir1)):
    os.mkdir(output_dir1)
output_dir2 = "d"+str(day)+"_"+str(image_size)+"_"+"standardize/train"
if not(os.path.exists(output_dir2)):
    os.mkdir(output_dir2)
output_dir3 = "d"+str(day)+"_"+str(image_size)+"_"+"standardize/train/fail"
if not(os.path.exists(output_dir3)):
    os.mkdir(output_dir3)
output_dir4 = "d"+str(day)+"_"+str(image_size)+"_"+"standardize/train/suc"
if not(os.path.exists(output_dir4)):
    os.mkdir(output_dir4)
output_dir5 = "d"+str(day)+"_"+str(image_size)+"_"+"standardize/test"
if not(os.path.exists(output_dir5)):
    os.mkdir(output_dir5)
output_dir6 = "d"+str(day)+"_"+str(image_size)+"_"+"standardize/test/fail"
if not(os.path.exists(output_dir6)):
    os.mkdir(output_dir6)
output_dir7 = "d"+str(day)+"_"+str(image_size)+"_"+"standardize/test/suc"
if not(os.path.exists(output_dir7)):
    os.mkdir(output_dir7)
#loading original images
train_fail_images = glob.glob(os.path.join(dir+'/d'+str(day)+'/fail',"*"))
print("train_fail_images:"+str(len(train_fail_images)))
train_suc_images = glob.glob(os.path.join(dir+'/d'+str(day)+'/suc',"*"))
print("train_suc_images:"+str(len(train_suc_images)))
#preprocessing original images
image_standerdize(train_fail_images, output_dir3, "train_fail")
image_standerdize(train_suc_images, output_dir4, "train_suc")
fail_images = glob.glob(os.path.join(dir+"/d"+str(day)+"_"+str(image_size)+"_"+'standardize/train/fail',"*"))
suc_images = glob.glob(os.path.join(dir+"/d"+str(day)+"_"+str(image_size)+"_"+'standardize/train/suc',"*"))
#data division into model derivation data (80%) and model validation data (20%)
test_fail_dir=dir+"/d"+str(day)+"_"+str(image_size)+"_"+"standardize/test/fail"
test_suc_dir=dir+"/d"+str(day)+"_"+str(image_size)+"_"+"standardize/test/suc"
for f in fail_images:
    if random.random() < 0.20:
        shutil.move(f, test_fail_dir)
for f in suc_images:
    if random.random() < 0.20:
        shutil.move(f, test_suc_dir)

#model construction################################################################################
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
#preparing learning data
train_dir=dir+"/d"+str(day)+"_"+str(image_size)+"_"+"standardize/train"
validation_dir=dir+"/d"+str(day)+"_"+str(image_size)+"_"+"standardize/test"
train_files = sorted(glob.glob(os.path.join(train_dir, '*', '*.*')))
train_data_count = len(train_files)
print(train_data_count)
val_files = sorted(glob.glob(os.path.join(validation_dir, '*', '*.*')))
val_data_count = len(val_files)
print(val_data_count)
#learning settings
data_augmentation=True
train_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(trim_size, trim_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical'
                                                    )
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                            target_size=(trim_size, trim_size),
                                                            batch_size=batch_size,
                                                            class_mode='categorical')
print(train_generator.class_indices)
model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-5),
            metrics=['acc'])
#early stopping
early_stopping=EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
#training
start_time = time.time()
history = model.fit_generator(train_generator,
                                steps_per_epoch=train_data_count/batch_size,
                                epochs=400,
                                validation_data=validation_generator,
                                validation_steps=val_data_count/batch_size,
                                callbacks=[early_stopping])
#saving model
model.save("model_"+cond[0]+"_"+str(start_time)+".h5", save_format='h5')
#saving graph
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1,len(acc) + 1)
fig=plt.figure()
plt.plot(epochs, acc,"bo",label="Training Acc")
plt.plot(epochs, val_acc,"b",label="Validation Acc")
plt.legend()
fig.savefig("acc"+str(start_time)+"_"+cond[0]+".pdf")
fig=plt.figure()
plt.plot(epochs,loss,"bo",label="Training Loss")
plt.plot(epochs,val_loss,"b",label="Validation Loss")
plt.legend()
fig.savefig("val"+str(start_time)+"_"+cond[0]+".pdf")
