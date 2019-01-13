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

from config import train_image_folder, img_width, img_height, train_file_name, valid_file_name
import utils

class DataGenSequence(Sequence):
    def __init__(self, store_folder, batch_count = 1000, batch_size = 16, usage = 'train'):
        self.usage = usage
        self.image_folder = train_image_folder
        self.batch_count = batch_count
        self.batch_size = batch_size
            
        train_samples_file = os.path.join(store_folder, train_file_name)

        df = pd.read_csv(train_samples_file)
        groups = df.groupby(['Id'])
        id2iamge = {}
        for (id, g) in groups:
            id2iamge[id] = [image for image in g['Image'].values]

        self.id2iamge = id2iamge
        self.ids = list(id2iamge.keys())

        print('catagory count: ', len(self.ids))

    def prepare_image(self, imageid):
        filename = os.path.join(self.image_folder, imageid)
        image = cv.imread(filename)
        image = cv.resize(image, (img_width, img_height), interpolation = cv.INTER_CUBIC)
        image = image[:, :, ::-1] # RGB
        if self.usage == 'train':
            image = aug_pipe.augment_image(image)

        return preprocess_input(image)

    def __len__(self):
        return self.batch_count
        
    def on_epoch_end(self):
        print('epoch end')

    def __getitem__(self, idx):
        batch_inputs1 = np.empty((self.batch_size, img_height, img_width, 3), dtype=np.float32)
        batch_inputs2 = np.empty((self.batch_size, img_height, img_width, 3), dtype=np.float32)
        batch_target = np.empty((self.batch_size, 1), dtype=np.float32)

        for index in range(self.batch_size // 2):
            ids = np.random.choice(self.ids, 2)
            match_images = np.random.choice(self.id2iamge[ids[0]], 2)
            unmatch_image = np.random.choice(self.id2iamge[ids[1]], 1)

            image0_id0 = self.prepare_image(match_images[0])
            image1_id0 = self.prepare_image(match_images[1])
            image0_id1 = self.prepare_image(unmatch_image[0])

            batch_inputs1[index * 2] = image0_id0
            batch_inputs2[index * 2] = image1_id0
            batch_target[index * 2] = 1

            batch_inputs1[index * 2 + 1] = image0_id0
            batch_inputs2[index * 2 + 1] = image0_id1
            batch_target[index * 2 + 1] = 0

        return [batch_inputs1, batch_inputs2], batch_target    


def revert_pre_process(x):
    return ((x + 1) * 127.5).astype(np.uint8)

if __name__ == '__main__':
    data_gen = DataGenSequence('output/siamese_folder_test')
    item = data_gen.__getitem__(0)
    x, y = item
    print(x[1].shape)
    print(y.shape)

    for i in range(10):
        image = revert_pre_process(x[0][i])
        image = image[:, :, ::-1].astype(np.uint8)
        print(image.shape)
        print(y[i])
        cv.imwrite('images/sample_0_{}.jpg'.format(i), image)

        image = revert_pre_process(x[1][i])
        image = image[:, :, ::-1].astype(np.uint8)
        print(image.shape)
        print(y[i])
        cv.imwrite('images/sample_1_{}.jpg'.format(i), image)

    
