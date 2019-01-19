import utils
import os
import keras
import numpy as np
import pandas as pd
import config

from keras_tqdm import TQDMNotebookCallback
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.layers import Add, GlobalMaxPooling2D, Lambda, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import img_height, img_width, num_channels, predict_data_folder, train_image_folder, origin_train_label_file
from data_generator_siamese_net import DataGenSequence, PredDataGenSequence, BranchPredDataGenSequence, HeadPredDataGenSequence, HeadPredDataGenSequenceForNegative
from pre_process import data_prepare_tokonizer_test
from utils import prepare_image

img_shape = (img_height, img_width, num_channels)

def build_branch_model():
    base_model = InceptionResNetV2(input_shape=img_shape, weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)

def build_model(activation='sigmoid', optimizer = keras.optimizers.Adam(lr=64e-5)):
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

siamese_folder_test = 'output/siamese_folder_test'
def train_test():
    if not os.path.isdir(siamese_folder_test):
        utils.init_output_folder(siamese_folder_test)
        #data_prepare_tokonizer_test(siamese_folder_test)

    model, branch_model, head_model = build_model()
    base_model = '/media/tongwu/workspace/kaggle/whale/output/siamese_folder_test/models/model.001-0.7979.hdf5'
    model.load_weights(base_model)
    for i in range(0, 10):        
        trained_models_path = os.path.join(siamese_folder_test, 'models')
        model_names = trained_models_path + '/model.{0}'.format(i) + '.{epoch:03d}-{val_acc:.4f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
        callbacks = [model_checkpoint,reduce_lr]
        data_prepare_tokonizer_test(siamese_folder_test, lambda x: len(x) >= (10 + i - 3) and len(x) <= (10 + i))

        epoch = 50
        history = model.fit_generator(
            DataGenSequence(siamese_folder_test, batch_count=300),
            steps_per_epoch=300,
            validation_data=DataGenSequence(siamese_folder_test, batch_count=60, usage = 'valid'),
            validation_steps=60,
            shuffle=True,
            epochs=epoch,
            initial_epoch=0, 
            callbacks=callbacks,
            verbose=1, 
            use_multiprocessing=False
            ).history

        best_model = utils.get_best_model(siamese_folder_test, pattern='model.{0}'.format(i) + '.(?P<epoch>\d+)-(?P<val_acc>[0-9]*\.?[0-9]*).hdf5')
        model.load_weights(best_model)

siamese_folder_train_wide = 'output/siamese_folder_train_wide'
def train_wide_sample(start, end):
    if not os.path.isdir(siamese_folder_train_wide):
        utils.init_output_folder(siamese_folder_train_wide)

    model, branch_model, head_model = build_model(optimizer=keras.optimizers.Adam(lr=1e-5))
    base_model = 'output/siamese_folder_train_wide/models/model.wide.4.6.013-0.9725.hdf5'
    model.load_weights(base_model)

    trained_models_path = os.path.join(siamese_folder_train_wide, 'models')
    model_names = trained_models_path + '/model.wide.{0}.{1}'.format(start, end) + '.{epoch:03d}-{val_acc:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    callbacks = [model_checkpoint,reduce_lr]
    data_prepare_tokonizer_test(siamese_folder_train_wide, lambda x: len(x) >= start and len(x) <= end)

    epoch = 300
    history = model.fit_generator(
        DataGenSequence(siamese_folder_train_wide, batch_count=100),
        steps_per_epoch=100,
        validation_data=DataGenSequence(siamese_folder_train_wide, batch_count=100, usage = 'valid'),
        validation_steps=100,
        shuffle=True,
        epochs=epoch,
        initial_epoch=0, 
        callbacks=callbacks,
        verbose=1, 
        use_multiprocessing=False
        ).history

best_model = 'output/siamese_folder_train_wide/models/model.wide.2.1000.005-0.9600.hdf5'
def predict(image_id1, image_id2):
    model, branch_model, head_model = build_model()
    model.load_weights(best_model)

    input1 = np.empty((1, img_height, img_width, 3), dtype=np.float32)
    input1[0] = prepare_image(train_image_folder, image_id1)

    input2 = np.empty((1, img_height, img_width, 3), dtype=np.float32)
    input2[0] = prepare_image(train_image_folder, image_id2)
    result = model.predict([input1, input2])
    print(result)

branch_model_output_folder = "output/branch_model"
def branch_model_output():
    df = pd.read_csv(origin_train_label_file)
    model, branch_model, head_model = build_model()
    model.load_weights(best_model)
    df = pd.read_csv(origin_train_label_file)
    datagen = BranchPredDataGenSequence(df)

    result = branch_model.predict_generator(datagen, (len(datagen) + 15) // 16, verbose=1)

    utils.save_obj(result, os.path.join(branch_model_output_folder, 'image2vec'))

    print(result)

def predict_one(imageid, image_folder = config.train_image_folder):
    vecs = utils.load_obj(os.path.join(branch_model_output_folder, 'image2vec'))

    input_image_raw = utils.prepare_image(config.train_image_folder, imageid)
    input_image = np.empty((1, img_height, img_width, 3), dtype=np.float32)
    input_image[0] = input_image_raw

    model, branch_model, head_model = build_model()
    model.load_weights(best_model)
    input_vec = branch_model.predict(input_image)[0]
    datagen = HeadPredDataGenSequence(vecs, input_vec)
    result = head_model.predict_generator(datagen, steps = (len(datagen) + 15) // 16, verbose=1)
    utils.save_obj(result, imageid + '.result')

def best_model_with_weights(freeze = True):
    model, branch_model, head_model = build_model()
    model.load_weights(best_model)

    for layer in model.layers:
        layer.trainable = not freeze

    return model, branch_model, head_model

def best_vecs():
    return utils.load_obj(os.path.join(branch_model_output_folder, 'image2vec'))

def predict_one_test_image(imageid, branch_model, head_model, vecs):
    input_image_raw = utils.prepare_image(config.test_image_folder, imageid)
    input_image = np.empty((1, img_height, img_width, 3), dtype=np.float32)
    input_image[0] = input_image_raw

    input_vec = branch_model.predict(input_image)[0]
    datagen = HeadPredDataGenSequence(vecs, input_vec, batch_size=64)
    result = head_model.predict_generator(datagen, steps = (len(datagen) + 63) // 64, verbose=1)
    utils.save_obj(result, os.path.join('output/predicts', imageid + '.result'))

def predict_test_image():
    model, branch_model, head_model = best_model_with_weights()
    input_vec = best_vecs()

    for image in os.listdir(config.test_image_folder):
        predict_one_test_image(image, branch_model, head_model, input_vec)


precords = utils.load_obj('record')
def agg_result(x):
    x_values = x.values
    idx = np.argsort(x_values)[::-1]
    length = len(x_values)
    
    if length > 10:
        return 1 - (1 - x_values[idx[0]]) * (1 - x_values[idx[-1]])
    else:
        return x_values[idx[0]]



def predicts_for_result(file):
    result = np.array([i[0] for i in utils.load_obj(file)])
    
    df = pd.read_csv(config.origin_train_label_file)
    df['value'] = pd.Series(result)
    df = df[df['Id'] != 'new_whale']
    result = df.groupby(['Id'])['value'].apply(agg_result)
    print(result.sort_values()[::-1])
    print(result)

def negative_possibility():
    model, branch_model, head_model = best_model_with_weights()
    vecs = best_vecs()

    df = pd.read_csv(config.origin_train_label_file)
    datagen = HeadPredDataGenSequenceForNegative(vecs, df, count = 256000)
    result = head_model.predict_generator(datagen, steps = 1000, verbose=1)

    recurrance = np.zeros((100000), dtype=int)
    for p in result:
        recurrance[(int)(p * 100000)] += 1

    print(result)
    utils.save_obj(recurrance, 'recurrance')

def process_record():
    recurrance = utils.load_obj('recurrance')
    init_count = 256000
    current_count = 256000

    record = np.zeros((100000), dtype=float)
    for i in range(100000):
        record[i] = (float)(current_count) / (float)(init_count)
        current_count -= recurrance[i]

    print(record)
    utils.save_obj(record, 'record')



if __name__ == '__main__':
    #train_test()
    #train_wide_sample(2, 1000) 
    ##predict('a81155e37.jpg', 'aba74d955.jpg')
    #pd.DataFrame([1, 2, 3])

    #branch_model_output()
    #predict_one('fffcde6fe.jpg')
    #branch_model_output()

    #predict_test_image()
    predicts_for_result('output/predicts/10b1c9d01.jpg.result')

    #negative_possibility()
    #process_record()
    #agg_result(1)

    