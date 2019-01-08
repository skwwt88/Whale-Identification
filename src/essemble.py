import os
import cv2 as cv
import numpy as np
import utils
import pandas as pd
import model

from config import test_image_folder, train_image_folder, img_height, img_width, model_name
from model import build_model
from utils import get_best_model
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model


models_path = ['../models/base/model.inception_res.02-72.hdf5', '../models/base/model.vgg16.hdf5']

test_eval = False

c2id = utils.load_obj('eval_c2id')
id2c = {v:k for k, v in c2id.items()}
eval_data = utils.load_obj('eval_data')

res_model = model.build_model_inception_restnet(eval_data)
res_model.load_weights('../models/base/model.inception_res.02-72.hdf5')

vgg16_model = model.build_model_vgg16(eval_data)
vgg16_model.load_weights('../models/base/model.vgg16.hdf5')

models = [res_model, vgg16_model]
weights = [0.7, 0.3]

if (test_eval):
    while True:
        image_id = input("image id:")
        filename = os.path.join(test_image_folder, image_id)
        print('Start processing image: {}'.format(filename))
        image = cv.imread(filename)
        image = cv.resize(image, (img_height, img_width), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
        rgb_img = preprocess_input(rgb_img)
        
        for m in models:
            preds = m.predict(rgb_img)
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
        print('*********************Start processing image: {}'.format(filename))
        image = cv.imread(filename)
        image = cv.resize(image, (img_height, img_width), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
        rgb_img = preprocess_input(rgb_img)

        p = np.empty(5004, dtype=np.float32)
        for i in range(5004):
            p[i] = 0
        for ii in range(2):
            preds = models[ii].predict(rgb_img)

            for iii in range(5004):
                p[iii] = max(p[iii], preds[0][iii])

            class_ids = preds[0].argsort()[-5:][::-1] 
            pp = [preds[0][i] for i in class_ids]
            print(pp)
            print([id2c[cid] for cid in class_ids])

        class_ids = p.argsort()[-4:][::-1]
        top4Pro = [p[i] for i in class_ids]
        print(top4Pro)
        print([id2c[cid] for cid in class_ids])

        ids = [id2c[cid] for cid in class_ids]

        if (top4Pro[0] < 0.65):
            ids.insert(0, 'new_whale')
        elif (top4Pro[1] < 0.65):
            ids.insert(1, 'new_whale')
        else:
            ids.insert(2, 'new_whale')

        ids = " ".join(ids)
        result.append((image_id, ids))
    
    df = pd.DataFrame(result, columns=["Image", "Id"])
    df.to_csv('../data/sub-essemble.csv', index=False)

