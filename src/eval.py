import os
import cv2 as cv
import numpy as np
import utils
import pandas as pd

from config import test_image_folder, train_image_folder, img_height, img_width, model_name
from model import build_model
from utils import get_best_model
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model

test_eval = False


if __name__ == '__main__':
    model_name = '../models/base/model.inception_res.02-72.hdf5'
    model = build_model(utils.load_obj('eval_data'), 'res')
    model.load_weights(model_name)

    c2id = utils.load_obj('eval_c2id')
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
            p = [preds[0][i] for i in class_ids]
            print(p)
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
            p = [preds[0][i] for i in class_ids]
            print(p)
            ids = [id2c[cid] for cid in class_ids]

            if (p[0] < 0.50):
                ids.insert(0, 'new_whale')
            elif (p[1] < 0.37):
                ids.insert(1, 'new_whale')
            else:
                ids.insert(2, 'new_whale')

            ids = " ".join(ids)
            result.append((image_id, ids))
        df = pd.DataFrame(result, columns=["Image", "Id"])
        df.to_csv('../data/sub-{0}.csv'.format(model_name), index=False)
        
    