import utils
import os
import keras
import numpy as np
import pandas as pd
import config
import random
import time
from operator import itemgetter
from random import randint
from itertools import chain

from keras_tqdm import TQDMNotebookCallback
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.layers import Add, GlobalMaxPooling2D, Lambda, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications.xception import Xception

from config import img_height, img_width, num_channels, train_image_folder, origin_train_label_file
from utils import prepare_image
from data_generator_siamese_net import BranchPredDataGenSequence, PairDataGen

siamese_store = '../output/siamese_net/{0}'.format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
train_wid2img = {}
valid_wid2img = {}
img2wid = {}
branch_output = 256

def prepare_data():
    utils.init_output_folder(siamese_store)
    df = pd.read_csv(origin_train_label_file)

    global img2wid
    img2wid = {item[0]: item[1] for item in df.values}

    df = df[df['Id'] != 'new_whale'].groupby(['Id']).filter(lambda x: len(x) >= 2)
    wid2img = {c: set(g['Image']) for (c, g) in df.groupby(['Id'])}
    wids = {c for (c, g) in df.groupby(['Id'])}

    valid_wids = set(random.sample(list(wids), 200))
    train_wids = wids - valid_wids

    global train_wid2img
    train_wid2img = {wid: wid2img[wid] for wid in train_wids}
    
    global valid_wid2img
    valid_wid2img = {wid: wid2img[wid] for wid in valid_wids}
    utils.save_obj(valid_wid2img, os.path.join(siamese_store, 'valid_wid2img'))
    


img_shape = (img_height, img_width, num_channels)

def build_branch_model1():
    base_model = InceptionResNetV2(input_shape=img_shape, weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)

def build_branch_model():
    base_model = Xception(input_shape=img_shape, weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.20, name='Dropout1')(x)
    x = Dense(branch_output)(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')(x)
    return Model(inputs=base_model.input, outputs=x)

def build_model1(activation='sigmoid', optimizer = keras.optimizers.SGD(lr=4e-5, momentum=0.9, decay=1e-6, nesterov=True)):
    branch_model = build_branch_model()

    mid        = 32
    xa_inp     = Input(shape=branch_model.output_shape[1:])
    xb_inp     = Input(shape=branch_model.output_shape[1:])
    x1         = Lambda(lambda x : x[0]*x[1])([xa_inp, xb_inp])
    x2         = Lambda(lambda x : x[0] + x[1])([xa_inp, xb_inp])
    x3         = Lambda(lambda x : K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4         = Lambda(lambda x : K.square(x))(x3)
    x          = Concatenate()([x1, x2, x3, x4])
    x          = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x          = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x          = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x          = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x          = Flatten(name='flatten')(x)
    
    # Weighted sum implemented as a Dense layer.
    x          = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a      = Input(shape=img_shape)
    img_b      = Input(shape=img_shape)
    xa         = branch_model(img_a)
    xb         = branch_model(img_b)
    x          = head_model([xa, xb])
    model      = Model([img_a, img_b], x)
    model.compile(optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model

def build_branch_model2():
    base_model = InceptionResNetV2(input_shape=img_shape, weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')(x)
    x = Dropout(0.18, name='Dropout1')(x)
    x = Dense(256)(x)
    return Model(inputs=base_model.input, outputs=x)

def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y

def build_branch_model3(l2 = 0, activation='sigmoid'):
    kwargs = {'padding': 'same'}
    inp = Input(shape=img_shape)  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4):
        x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4):
        x = subblock(x, 128, **kwargs)

    x = GlobalMaxPooling2D()(x)  # 512

    return Model(inp, x)

def build_model(activation='sigmoid', optimizer = keras.optimizers.Adam(lr=1e-5)):
    branch_model = build_branch_model()
    xa_inp     = Input(shape=branch_model.output_shape[1:])
    xb_inp     = Input(shape=branch_model.output_shape[1:])
    x1         = Lambda(lambda x : K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x2         = Lambda(lambda x : K.square(x))(x1)
    x          = Concatenate()([x1, x2])
    x          = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    img_a      = Input(shape=img_shape)
    img_b      = Input(shape=img_shape)
    xa         = branch_model(img_a)
    xb         = branch_model(img_b)
    x          = head_model([xa, xb])
    model      = Model([img_a, img_b], x)
    model.compile(optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model

#base_model = 'output/siamese_folder_train_wide/models/model.wide.5.40.009-0.8906.hdf5'
#base_model = '../output/siamese_net/models/2019-02-08-19-10-56/model-0.8580-0.2742.hdf5'
#base_model = '../output/siamese_net/models/2019-02-09-19-01-42/model-0.8601-0.5766.hdf5'
def images2Vecs(images, branch_model):
    batch_size = 128
    datagen = BranchPredDataGenSequence(images, batch_size=batch_size)
    pred_result = branch_model.predict_generator(datagen, steps = (len(datagen) + batch_size - 1) // batch_size, verbose=1)

    result = {}
    for i in range(len(images)):
        result[images[i]] = pred_result[i]

    #utils.save_obj(result, os.path.join(siamese_store, 'img2vecs'))
    return result

def generate_train_pair(images, img2vecs, img2wid, head_model, sigma1 = 0.2, sigma2 = 0.2):
    result = []
    process_img_count = 0

    np.random.seed(int(round(time.time() * 1000)) % 100000)
    bias1 = np.random.normal(0, sigma1, 10000)
    bias2 = np.random.normal(0, sigma2, 10000)
    for img1 in images:
        img1vec = img2vecs[img1]
        inputs1 = np.empty((len(images), branch_output), dtype=np.float32)
        inputs2 = np.empty((len(images), branch_output), dtype=np.float32)
        for index in range(len(images)):
            img2vec = img2vecs[images[index]]
            inputs1[index] = img1vec
            inputs2[index] = img2vec
        
        predicts = head_model.predict_on_batch([inputs1, inputs2])
        predicts.flatten()

        possitive_pairs = []
        nagetive_pairs = []
        for index in range(len(images)):
            if img2wid[img1] == img2wid[images[index]]:
                possitive_pairs.append((img1, images[index], predicts[index] + bias2[randint(0, 9999)]))
            else:
                nagetive_pairs.append((img1, images[index], predicts[index] + bias1[randint(0, 9999)]))
        
        possitive_pairs = sorted(possitive_pairs, key=itemgetter(2), reverse=True)
        nagetive_pairs = sorted(nagetive_pairs, key=itemgetter(2))

        for index in range(min(len(possitive_pairs), len(nagetive_pairs), 5)):            
            result.append((possitive_pairs[index][0], possitive_pairs[index][1]))
            result.append((nagetive_pairs[index][0], nagetive_pairs[index][1]))

        process_img_count += 1
        if (process_img_count % 100) == 0:
            print('{0} img processed. pairs count {1}'.format(process_img_count, len(result)))

    print("{0} pairs generated.".format(len(result)))
    return result

def get_train_images(min_count = 3, max_count = 10):
    df = pd.read_csv(origin_train_label_file)
    df = df.groupby(['Id']).filter(lambda x: len(x) >= min_count and len(x) <= max_count)
    wids = [c for (c, g) in df.groupby(['Id'])]
    imgs = [item[0] for item in df.values]

    return wids, imgs
    
def valid_pairs():
    df = pd.read_csv(origin_train_label_file)
    df = df.groupby(['Id']).filter(lambda x: len(x) == 2).sort_values(['Id'])

    img1 = [item[0] for item in df.values]
    img2 = [item[0] for item in df.values]

    random.shuffle(img1)
    random.shuffle(img2)

    diff_pair = []
    for index in range(len(df)):
        if img1[index] == img2[index]:
            continue
        diff_pair.append((img1[index], img2[index]))
    random.shuffle(diff_pair)

    same_pair = []
    for index in range(len(df))[::2]:
        same_pair.append((df.iloc[index][0], df.iloc[index + 1][0]))
    random.shuffle(same_pair)

    return diff_pair[:500] + same_pair[:500]

trained_models_path = os.path.join(siamese_store, 'models/{0}'.format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))

def make_step(model, branch_model, head_model, imgs, sigma1 = 15, sigma2 = 15, new_lr = 8e-5, epoch_count = 5):
    model_names = trained_models_path + '/model-{val_binary_crossentropy:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_binary_crossentropy', verbose=1, save_best_only=True)
    
    img2vecs = images2Vecs(imgs, branch_model)
    
    train_pairs = generate_train_pair(imgs, img2vecs, img2wid, head_model, sigma1 = sigma1, sigma2 = sigma2)
    validation_pairs = valid_pairs()

    #K.set_value(model.optimizer.lr, new_lr)

    train_datagen = PairDataGen(train_pairs, img2wid, batch_size=8)
    valid_datagen = PairDataGen(validation_pairs, img2wid, usage='test', batch_size=8)
    history = model.fit_generator(
                train_datagen,
                steps_per_epoch=((len(train_pairs) + 15) // 8),
                validation_data=valid_datagen,
                validation_steps=((len(validation_pairs) + 15) // 8),
                shuffle=True,
                epochs=epoch_count,
                initial_epoch=0, 
                callbacks=[model_checkpoint, reduce_lr],
                verbose=1, 
                use_multiprocessing=False,
                max_queue_size=4, 
                workers=2
                ).history

    return epoch_count

def imgs_count(count):
    df = pd.read_csv(origin_train_label_file)
    df = df.groupby(['Id']).filter(lambda x: len(x) == count)

    return len(df)

def predict(images, img2wid, img, img2vecs, head_model):
    img1vec = img2vecs[img]
    inputs1 = np.empty((len(images), branch_output), dtype=np.float32)
    inputs2 = np.empty((len(images), branch_output), dtype=np.float32)
    for index in range(len(images)):
        img2vec = img2vecs[images[index]]
        inputs1[index] = img1vec
        inputs2[index] = img2vec
    
    predicts = head_model.predict_on_batch([inputs1, inputs2])
    predicts.flatten()

    return predicts


if __name__ == '__main__':
    prepare_data()

    model, branch_model, head_model = build_model()
    #model.load_weights(base_model)
    #model.summary()

    test = False
    sigma = [10,10,10,10,10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    l_rate = [4e-5, 4e-5, 4e-5, 4e-5, 4e-5, 4e-5, 4e-5, 4e-5, 4e-5, 4e-5, 4e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5 ]
    epoch_record = [100, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    #tmp test
    if test:
        model.load_weights(base_model)
        df = pd.read_csv(origin_train_label_file)
        df = df.groupby(['Id']).filter(lambda x: len(x) <= 1)
        imgs = [item[0] for item in df.values]
        df = pd.read_csv(origin_train_label_file)
        df = df.groupby(['Id']).filter(lambda x: len(x) == 10).sort_values(['Id'])
        img_test = []
        for index in random.sample(range(len(df) // 10), 20):
            img_test.append(df.iloc[10 * index][0])
            img_test.append(df.iloc[10 * index + 1][0])
        imgs += img_test
        vecs = images2Vecs(imgs, branch_model)
        for img in img_test:
            predict(imgs, img2wid, img, vecs, head_model)
    else:
        if not os.path.exists(trained_models_path):
            os.mkdir(trained_models_path)

        i = 0
        while True:
            train_wids = random.sample(list(train_wid2img), 10)
            imgs = list(chain.from_iterable([wid2img[wid] for wid in train_wids]))
            i += make_step(model, branch_model, head_model, imgs, sigma1=sigma[i // 10], sigma2=sigma[i // 5], new_lr=64e-5, epoch_count=epoch_record[i // 10])


    
