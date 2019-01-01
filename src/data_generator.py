# encoding=utf-8
import csv
import os
import Augmentor
import random
import utils

import cv2 as cv
import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.utils import shuffle
from augmentor import aug_pipe

from config import train_image_folder, samples_file, valid_file, batch_size, img_width, img_height, nb_classes

class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        self.image_folder = train_image_folder
            
        if self.usage == 'train':
            self.train_samples = shuffle(pd.read_csv(samples_file)).reset_index(drop=True)
        else:
            self.train_samples = shuffle(pd.read_csv(valid_file)).reset_index(drop=True)

        self.data_config = utils.load_obj('data_config')
        self.c2id = utils.load_obj('c2id')

        print(self.data_config)
        print("sample count: ", len(self.train_samples))

    def __len__(self):
        return len(self.train_samples) // batch_size
        
    def on_epoch_end(self):
        self.train_samples = shuffle(self.train_samples).reset_index(drop=True)

    def __getitem__(self, idx):
        i = idx * batch_size
        length = min(batch_size, (len(self.train_samples) - i))
        batch_inputs = np.empty((length, img_height, img_width, 3), dtype=np.float32)
        batch_target = np.empty((length, nb_classes), dtype=np.float32)

        for index in range(length):
            sample = self.train_samples.iloc[i + index]
            filename = os.path.join(self.image_folder, sample['Image'])

            image = cv.imread(filename)
            image = cv.resize(image, (img_width, img_height), interpolation = cv.INTER_CUBIC)
            image = image[:, :, ::-1] # RGB
            if self.usage == 'train':
                image = aug_pipe.augment_image(image)

            batch_inputs[index] = preprocess_input(image)
            batch_target[index] = to_categorical(self.c2id[sample['Id']], nb_classes)

        return batch_inputs, batch_target

def revert_pre_process(x):
    return ((x + 1) * 127.5).astype(np.uint8)

if __name__ == '__main__':
    data_gen = DataGenSequence('train')
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

    
