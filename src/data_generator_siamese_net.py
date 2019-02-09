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

def prepare_img(img, folder, usage = 'train'):
    filename = os.path.join(folder, img)
    image = cv.imread(filename)
    image = cv.resize(image, (img_width, img_height), interpolation = cv.INTER_CUBIC)
    image = image[:, :, ::-1] # RGB
    if usage == 'train':
        image = aug_pipe.augment_image(image)

    #return preprocess_input(image)
    return image

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

class PredDataGenSequence(Sequence):
    

    def __init__(self, store_folder, image_id, batch_size = 16):
        self.image_folder = train_image_folder
        self.batch_size = batch_size
            
        train_samples_file = os.path.join(store_folder, train_file_name)

        df = pd.read_csv(train_samples_file)
        self.predict_samples = df

        batch_inputs0 = np.empty((self.batch_size, img_height, img_width, 3), dtype=np.float32)

        image = utils.prepare_image(self.image_folder, image_id)
        for i in range(self.batch_size):
            batch_inputs0[i] = image
        
        self.batch_inputs0 = batch_inputs0

        print('catagory count: ', len(self.predict_samples))

    

    def __len__(self):
        return len(self.predict_samples)
        
    def on_epoch_end(self):
        print('epoch end')

    def __getitem__(self, idx):
        i = idx * self.batch_size
        length = min(self.batch_size, (len(self.predict_samples) - i))
        batch_inputs1 = np.empty((self.batch_size, img_height, img_width, 3), dtype=np.float32)

        for index in range(length):
            sample = self.predict_samples.iloc[i + index]
            batch_inputs1[index] = utils.prepare_image(self.image_folder, sample['Image'])

        return [self.batch_inputs0, batch_inputs1]   

class BranchPredDataGenSequence(Sequence):
    def __init__(self, samples, batch_size = 16, image_folder = train_image_folder):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.samples = samples

        print('sample count: ', len(self.samples))

    def __len__(self):
        return len(self.samples)
        
    def on_epoch_end(self):
        print('epoch end')

    def __getitem__(self, idx):
        i = idx * self.batch_size
        length = min(self.batch_size, (len(self.samples) - i))
        inputs = np.empty((length, img_height, img_width, 3), dtype=np.float32)

        for index in range(length):
            img = self.samples[i + index]
            inputs[index] = utils.prepare_image(self.image_folder, img)

        return inputs   

class HeadPredDataGenSequence(Sequence):
    def __init__(self, vecs, input_vec, batch_size = 16, image_folder = train_image_folder):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.vecs = vecs
        self.input_vec = input_vec

        print('sample count: ', len(self.vecs))

    def __len__(self):
        return len(self.vecs)
        
    def on_epoch_end(self):
        print('epoch end')

    def __getitem__(self, idx):
        i = idx * self.batch_size
        length = min(self.batch_size, (len(self.vecs) - i))
        inputs1 = np.empty((length, 1536), dtype=np.float32)
        inputs2 = np.empty((length, 1536), dtype=np.float32)

        for index in range(length):
            inputs1[index] = self.vecs[i + index]
            inputs2[index] = self.input_vec

        return [inputs1, inputs2]  

class HeadPredDataGenSequenceForNegative(Sequence):
    def __init__(self, vecs, df, count = 25600000, batch_size = 256, image_folder = train_image_folder):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.vecs = vecs
        self.df = df
        self.length = len(df)
        self.count = count

        print('sample count: ', len(self.vecs))

    def __len__(self):
        return self.count
        
    def on_epoch_end(self):
        print('epoch end')

    def __getitem__(self, idx):
        i = idx * self.batch_size
        inputs1 = np.empty((self.batch_size, 1536), dtype=np.float32)
        inputs2 = np.empty((self.batch_size, 1536), dtype=np.float32)

        for index in range(self.batch_size):
            idxs = [0, 0]
            while True:
                idxs = np.random.choice(range(self.length), 2)
                if self.df.iloc[idxs[0]]['Id'] != self.df.iloc[idxs[1]]['Id']:
                    break

            inputs1[index] = self.vecs[idxs[0]]
            inputs2[index] = self.vecs[idxs[1]]


        return [inputs1, inputs2]  

class PairDataGen(Sequence):
    def __init__(self, pairs, img2wid, usage = 'train', batch_size = 128, image_folder = train_image_folder):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.pairs = pairs
        self.usage = usage
        self.img2wid = img2wid

        print('sample count: ', len(self.pairs))

    def __len__(self):
        return ((len(self.pairs) + self.batch_size - 1) // self.batch_size)
        
    def on_epoch_end(self):
        print('epoch end')

    def __getitem__(self, idx):
        i = idx * self.batch_size
        length = min(self.batch_size, (len(self.pairs) - i))
        inputs1 = np.empty((length, img_height, img_width, 3), dtype=np.float32)
        inputs2 = np.empty((length, img_height, img_width, 3), dtype=np.float32)
        target = np.empty((length, 1), dtype=np.float32)

        for index in range(length):
            img1, img2 = self.pairs[i + index]
            inputs1[index] = prepare_img(img1, self.image_folder, self.usage)
            inputs2[index] = prepare_img(img2, self.image_folder, self.usage)
            if self.img2wid[img1] == self.img2wid[img2]:
                target[index] = 0
            else:
                target[index] = 1


        return [inputs1, inputs2], target


def revert_pre_process(x):
    return ((x + 1) * 127.5).astype(np.uint8)

if __name__ == '__main__':
    data_gen = DataGenSequence('output/siamese_folder_test')
    item = data_gen.__getitem__(0)
    x, y = item
    print(x[1].shape)
    print(y.shape)

    pred_data_gen = PredDataGenSequence('output/siamese_folder_test', '71706ea66.jpg')
    pitem = pred_data_gen.__getitem__(0)
    px = pitem
    print(px[1].shape)
    print(px[0].shape)

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

    for i in range(10):
        image = revert_pre_process(px[0][i])
        image = image[:, :, ::-1].astype(np.uint8)
        print(image.shape)
        print(y[i])
        cv.imwrite('images/sample_0_{}.jpg'.format(i), image)

        image = revert_pre_process(x[1][i])
        image = image[:, :, ::-1].astype(np.uint8)
        print(image.shape)
        print(y[i])
        cv.imwrite('images/sample_1_{}.jpg'.format(i), image)

    

    
