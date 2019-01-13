import utils
import os
import keras

from keras_tqdm import TQDMNotebookCallback
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.layers import Add, GlobalMaxPooling2D, Lambda, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from config import img_height, img_width, num_channels
from data_generator_siamese_net import DataGenSequence
from pre_process import data_prepare_tokonizer_test

img_shape = (img_height, img_width, num_channels)

def build_branch_model():
    base_model = InceptionResNetV2(input_shape=img_shape, weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)

def build_model(activation='sigmoid'):
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
    model.compile(keras.optimizers.Adam(lr=64e-4), loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model

def build_model_test(activation='sigmoid'):
    branch_model = build_branch_model()

    img_a      = Input(shape=img_shape)
    img_b      = Input(shape=img_shape)
    xa         = branch_model(img_a)
    xb         = branch_model(img_b)
    x = Concatenate([xa, xb])


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
    
    x          = head_model([xa, xb])
    model      = Model([img_a, img_b], x)
    model.compile(keras.optimizers.Adam(lr=64e-4), loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model

siamese_folder_test = 'output/siamese_folder_test'
def train_test():
    if not os.path.isdir(siamese_folder_test):
        utils.init_output_folder(siamese_folder_test)
        data_prepare_tokonizer_test(siamese_folder_test)

    model, branch_model, head_model = build_model()

    history = model.fit_generator(
        DataGenSequence(siamese_folder_test, batch_count=1000),
        steps_per_epoch=1000,
        validation_data=DataGenSequence(siamese_folder_test, batch_count=50, usage = 'valid'),
        validation_steps=50,
        shuffle=True,
        epochs=1000,
        initial_epoch=0, 
        callbacks=[],
        verbose=1,
        use_multiprocessing=False
        ).history
    

if __name__ == '__main__':
    train_test()