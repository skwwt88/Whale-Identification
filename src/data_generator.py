# encoding=utf-8
import csv
import os
import Augmentor
import random

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import Sequence
from keras.utils import to_categorical

from config import train_image_folder, train_label_file, batch_size, img_width, img_height, num_classes

image_folder = train_image_folder
annot_file = train_label_file
            
samples = []
with open(annot_file, 'r') as file:
    samples = [x for x in list(csv.reader(file))[1:] if x[1] != "new_whale"]
    # samples = [x for x in list(csv.reader(file))[1:]]
            
d_dic = {}
for s in samples:
    cid = s[1]
    if cid in d_dic:
        d_dic[cid].append(s)
    else:
        d_dic[cid] = [s]

samples = []
classes = []
for cid in d_dic.keys():
    if len(d_dic[cid]) > 10:
        new_items = [(s[0], cid) for s in d_dic[cid]]
        samples += new_items
        classes.append(cid)


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        self.image_folder = train_image_folder
        
        
        self.train_samples = samples
        if self.usage != 'train':
            self.train_samples = random.sample(samples, 200)
        self.c2id = dict((c, i) for i, c in enumerate(classes))

        print(len(self.train_samples))
        print(len(classes))
        np.random.shuffle(self.train_samples)

    def __len__(self):
        return len(self.train_samples) // batch_size
        
    def on_epoch_end(self):
        np.random.shuffle(self.train_samples)

    def __getitem__(self, idx):
        i = idx * batch_size
        length = min(batch_size, (len(self.train_samples) - i))
        batch_inputs = np.empty((length, img_height, img_width, 3), dtype=np.float32)
        batch_target = np.empty((length, num_classes), dtype=np.float32)

        for i_batch in range(length):
            sample = self.train_samples[i + i_batch]
            image_id = sample[0]
            filename = os.path.join(self.image_folder, image_id)
            class_id = sample[1]

            image = cv.imread(filename)
            image = cv.resize(image, (img_width, img_height), interpolation = cv.INTER_CUBIC)
            image = image[:, :, ::-1] # RGB

            # print(image_id, ", ", self.c2id[class_id])
            batch_inputs[i_batch] = preprocess_input(image)
            batch_target[i_batch] = to_categorical(self.c2id[class_id], num_classes)

        return batch_inputs, batch_target

def revert_pre_process(x):
    return ((x + 1) * 127.5).astype(np.uint8)

if __name__ == '__main__':
    data_gen = DataGenSequence('t1rain')
    item = data_gen.__getitem__(0)
    x, y = item
    print(x.shape)
    print(y.shape)

    for i in range(10):
        image = revert_pre_process(x[i])
        image = image[:, :, ::-1].astype(np.uint8)
        print(image.shape)
        cv.imwrite('images/sample_{}.jpg'.format(i), image)

    
