#!/usr/bin/env python
# encoding: utf-8
'''
@author: Sunzb
@time: 2019/9/3 15:27
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import glob
import model
import scipy.misc
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')

parser = argparse.ArgumentParser()
parser.add_argument("--is_test", type=bool, default=False, help="testing of not")
a = parser.parse_args()

label_dict = {'2S1': 0, 'BMP2': 1, 'BRDM2': 2, 'BTR60': 3, 'BTR70': 4, 'D7': 5, 'T62': 6, 'T72': 7, 'ZIL131': 8,
              'ZSU23': 9}

label_dict_res = {v: k for k, v in label_dict.items()}

init_learning_rate = 0.0002
max_epochs = 800
n_classes = 10
img_width = 299
img_height = 299
batch_size = 8
capacity = 2000
logs_train_dir = './model_save'


def get_batch_img(image, label, shuffle=False):
    data_size = image.shape[0]
    num_batches_per_epoch = int(len(image) / batch_size) + 1
    # 文件名队列存放的是参与训练的文件名
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = image[shuffle_indices]
        shuffled_label = label[shuffle_indices]
    else:
        shuffled_data = image
        shuffled_label = label

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index], shuffled_label[start_index:end_index]


def get_image(file_dir):
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
        # for image in glob.glob(os.path.join(file_dir, label, "*.jpg")):
        for image in glob.glob(os.path.join(file_dir, label, "*.*")):
            image_list.append(image)
            label_list.append(label_dict[label])
    print('There are %d data' % (len(image_list)))
    size = len(image_list)
    all_images = np.zeros([size, img_height, img_width])
    for index, path in enumerate(image_list):
        all_images[index] = (scipy.misc.imresize(scipy.misc.imread(path), (img_height, img_width)).astype(
            np.float) / 255. - 0.5) * 2
    all_images = all_images[:, :, :, np.newaxis]
    all_label = np.array(label_list, dtype=np.int32)
    return all_images, all_label


def train(train_dir):
    image_list, label_list = get_image(train_dir)
    (train_x, test_x, train_y, test_y) = train_test_split(image_list, label_list, test_size=0.25, random_state=42)
    train_y = to_categorical(train_y, num_classes=n_classes)
    test_y = to_categorical(test_y, num_classes=n_classes)


    # 数据增广
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    print("[INFO] Compiling Model...")
    # network
    inception_v4 = model.create_inception_v4(img_height, img_height, nb_classes=10, load_weight=False)
    optimizer = optimizers.Adam(lr=init_learning_rate, decay=init_learning_rate / max_epochs)
    inception_v4.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print("[INFO] Training Network...")
    history = inception_v4.fit_generator(ImageDataGenerator().flow(train_x, train_y, batch_size=batch_size),
                                         steps_per_epoch=len(train_x) // batch_size,
                                         epochs=max_epochs,
                                         validation_data=(test_x, test_y))
    print("[INFO] Saving Model...")
    model_base = 'trained_model' + '.h5'
    model.save(model_base)


if __name__ == '__main__':
    train_dir = './train_set/'
    train(train_dir)
