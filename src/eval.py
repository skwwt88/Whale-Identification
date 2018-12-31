import os
import cv2 as cv
import numpy as np
import utils
import pandas as pd

from config import test_image_folder, train_image_folder, img_height, img_width
from model import build_model
from utils import get_best_model
from keras.applications.inception_resnet_v2 import preprocess_input

test_eval = False

if __name__ == '__main__':
    best_model, epoch = get_best_model()
    model = build_model()
    model.load_weights(best_model)

    c2id = utils.load_obj('c2id')
    id2c = {v:k for k, v in c2id.items()}
    if (test_eval):
        while True:
            image_id = input("image id:")
            filename = os.path.join(train_image_folder, image_id)
            print('Start processing image: {}'.format(filename))
            image = cv.imread(filename)
            image = cv.resize(image, (img_height, img_width), cv.INTER_CUBIC)
            rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
            rgb_img = preprocess_input(rgb_img)
            preds = model.predict(rgb_img)
            prob = np.max(preds)
            class_ids = preds[0].argsort()[-3:][::-1] 
            print([id2c[cid] for cid in class_ids])
    else:
        test_image_files = [f for f in os.listdir(test_image_folder)]
        result = list()

        for image_id in test_image_files:
            filename = os.path.join(test_image_folder, image_id)
            print('Start processing image: {}'.format(filename))
            image = cv.imread(filename)
            image = cv.resize(image, (img_height, img_width), cv.INTER_CUBIC)
            rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
            rgb_img = preprocess_input(rgb_img)
            preds = model.predict(rgb_img)
            prob = np.max(preds)
            class_ids = preds[0].argsort()[-4:][::-1]
            ids = " ".join(["new_whale"] + [id2c[cid] for cid in class_ids])
            result.append((image_id, ids))
        df = pd.DataFrame(result, columns=["Image", "Id"])
        df.to_csv('../data/new_sub.csv', index=False)
        
    