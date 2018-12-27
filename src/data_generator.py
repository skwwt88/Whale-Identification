# encoding=utf-8
import csv
import os
import Augmentor

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import Sequence
from keras.utils import to_categorical

from config import train_image_folder, train_label_file, batch_size, img_width, img_height

class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        if self.usage == 'train':
            annot_file = train_label_file
            self.image_folder = train_image_folder
        else:
            annot_file = valid_annot
            self.image_folder = valid_image_folder

        with open(annot_file, 'r') as file:
            self.samples = [x for x in list(csv.reader(file))[1:] if x[1] != "new_whale"]

        self.num_classes = list(set([x[1] for x in self.samples]))
        self.num_classes.sort()
        self.c2id = dict((c, i) for i, c in enumerate(self.num_classes))

        # print self.samples
        print len(self.num_classes)
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples) // batch_size
        
    def on_epoch_end(self):
        np.random.shuffle(self.samples)

    def __getitem__(self, idx):
        i = idx * batch_size
        length = min(batch_size, (len(self.samples) - i))
        batch_inputs = np.empty((length, img_height, img_width, 3), dtype=np.float32)
        batch_target = np.empty((length, len(self.num_classes)), dtype=np.float32)

        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            image_id = sample[0]
            filename = os.path.join(self.image_folder, image_id)
            class_id = sample[1]

            image = cv.imread(filename)
            image = cv.resize(image, (img_width, img_height), interpolation = cv.INTER_CUBIC)
            image = image[:, :, ::-1] # RGB

            batch_inputs[i_batch] = preprocess_input(image)
            batch_target[i_batch] = to_categorical(self.c2id[class_id], len(self.num_classes))

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
        cv.imwrite('images/sample_{}.jpg'.format(i), image)

    
