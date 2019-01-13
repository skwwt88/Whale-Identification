# encoding=utf-8
import csv
import os
import Augmentor
import random

import cv2 as cv
import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.utils import shuffle
from augmentor import aug_pipe

from config import train_image_folder, batch_size, img_width, img_height, train_file_name, valid_file_name
import utils

class DataGenSequence(Sequence):
    def __init__(self, store_folder, usage = 'train'):
        self.usage = usage
        self.image_folder = train_image_folder
            
        train_samples_file = os.path.join(store_folder, train_file_name)
        if (self.usage != 'train'):
            train_samples_file = os.path.join(store_folder, valid_file_name)

        df = pd.read_csv(train_samples_file)
        self.train_samples = shuffle(df).reset_index(drop=True)

        self.c2id = utils.load_c2id(store_folder)
        (nb_classes, num_train_samples, num_valid_samples) = utils.load_config(store_folder)
        self.nb_classes = nb_classes

        print("sample count: {0}".format(len(self.train_samples)))
        print('nb_classes: {0}'.format(nb_classes))


    def __len__(self):
        return len(self.train_samples) // batch_size
        
    def on_epoch_end(self):
        self.train_samples = shuffle(self.train_samples).reset_index(drop=True)

    def __getitem__(self, idx):
        i = idx * batch_size
        length = min(batch_size, (len(self.train_samples) - i))
        batch_inputs = np.empty((length, img_height, img_width, 3), dtype=np.float32)
        batch_target = np.empty((length, self.nb_classes), dtype=np.float32)

        for index in range(length):
            sample = self.train_samples.iloc[i + index]
            filename = os.path.join(self.image_folder, sample['Image'])

            image = cv.imread(filename)
            image = cv.resize(image, (img_width, img_height), interpolation = cv.INTER_CUBIC)
            image = image[:, :, ::-1] # RGB
            if self.usage == 'train':
                image = aug_pipe.augment_image(image)

            batch_inputs[index] = preprocess_input(image)
            batch_target[index] = to_categorical(self.c2id[sample['Id']], self.nb_classes)

        return batch_inputs, batch_target

def revert_pre_process(x):
    return ((x + 1) * 127.5).astype(np.uint8)

if __name__ == '__main__':
    data_gen = DataGenSequence('output/test')
    item = data_gen.__getitem__(0)
    x, y = item
    print(x.shape)
    print(y.shape)

    for i in range(10):
        image = revert_pre_process(x[i])
        image = image[:, :, ::-1].astype(np.uint8)
        print(image.shape)
        print(y[i])
        cv.imwrite('images/sample_{}.jpg'.format(i), image)

    
