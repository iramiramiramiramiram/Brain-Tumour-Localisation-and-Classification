import numpy as np
import pandas as pd
import os
from os import listdir
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
import imutils
import shutil
import pickle

from keras.utils import image_dataset_from_directory
from keras.models import Model,load_model
from keras.layers import Conv2D,Input,ZeroPadding2D,BatchNormalization,Flatten,Activation,Dense,MaxPooling2D

image_dir = "archive-2/Training"





def crop_brain_contour(image, plot=False):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False,
                        labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False,
                        labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()

    return new_image

destination_path = "preprocced_dir/training"
if not os.path.exists(destination_path):
    os.makedirs(destination_path)
for type_folder in os.listdir(image_dir):
    folder_path= os.path.join(image_dir,type_folder)
    # print(folder_path)
    destination_folder=os.path.join(destination_path,type_folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if type_folder != ".DS_Store":
        for img_name in os.listdir(folder_path):
            ex_img = cv2.imread(os.path.join(folder_path,img_name))
            ex_crop_img = crop_brain_contour(ex_img, False)
            cv2.imwrite(os.path.join(destination_folder,img_name),ex_crop_img)

train_dt, val_dt = image_dataset_from_directory(
    directory = image_dir,
    labels="inferred",
    batch_size = 32,
    image_size = (224, 224),
    subset = "both",
    validation_split = 0.2,
    seed = 42)

#print(train_dt) using 4570 files for training
#print(val_dt) using 1142 files for validation
class_names = train_dt.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_dt = train_dt.cache().prefetch(buffer_size=AUTOTUNE)
val_dt = val_dt.cache().prefetch(buffer_size=AUTOTUNE)
#plt.imshow(train_dt)

def build_model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1))(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((4, 4))(X)
    X = MaxPooling2D((4, 4))(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=X_input, outputs=X)

    return model


IMG_SHAPE = (224, 224, 3)
'''model = build_model(IMG_SHAPE)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])'''
'''
epochs = 50
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 20))

history_5 = model.fit(train_dt,
                        epochs=epochs,
                        steps_per_epoch=len(train_dt),
                        callbacks=[lr_scheduler],
                        verbose=1)'''
'''tf.keras.layers.RandomFlip("horizontal"),
tf.keras.layers.RandomRotation(0.25),
tf.keras.layers.RandomZoom(0.05, 0.05),
tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1)'''
model_1 = tf.keras.Sequential(layers=[
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"),
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(1, 1)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(1, 1)),

    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(1, 1)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(4, activation="softmax")
])

model_1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0014),
                metrics=["accuracy"])

epochs = 100
history_1 = model_1.fit(train_dt,
                        validation_data=val_dt,
                        epochs=epochs,
                        steps_per_epoch=len(train_dt),
                        validation_steps=len(val_dt),
                        verbose=2)

with open("model_1.pkl", "wb"):
    pickle.dump(model_1, file)

