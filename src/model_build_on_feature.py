import utils
import os
import keras
import siamese_net
import numpy as np
import pandas as pd
import config

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import img_height, img_width, num_channels
from data_generator import DataGenSequence, PredDataGenSequence
from pre_process import data_prepare_feature_extraction
from feature_model_rest import build_pre_trained_feature_model, data_prepare_feature_extraction_full

store_folder = 'output/feature_rest_full'
batch_size = 32
num_epochs = 1000
patience = 50
dropout_rate = 0.2

def build_model_from_base(base_model, nb_classes):

    x = base_model.get_output_at(0)
    x = Dropout(dropout_rate, name='Dropout')(x)
    x = Dense(nb_classes, name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)
    
    return Model(inputs=base_model.get_input_at(0), outputs=x)

def train():
    if not os.path.isdir(store_folder):
        utils.init_output_folder(store_folder)
        data_prepare_feature_extraction_full(store_folder, min_sample_count=0, valid_ratio=0.1)

    data_prepare_feature_extraction_full(store_folder, valid_ratio=0.3)
        
    (nb_classes, num_train_samples, num_valid_samples) = utils.load_config(store_folder)

    base_model = build_pre_trained_feature_model()
    #smodel, base_model, head_model = siamese_net.best_model_with_weights()
    model = build_model_from_base(base_model, nb_classes)

    trained_models_path = os.path.join(store_folder, 'models')
    model_names = trained_models_path + '/model.{epoch:02d}-{val_acc:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    callbacks = [model_checkpoint, early_stop, reduce_lr]

    sgd = keras.optimizers.SGD(lr=2.5e-3, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(
            DataGenSequence(store_folder, 'train'),
            steps_per_epoch=num_train_samples // batch_size,
            validation_data=DataGenSequence(store_folder, 'valid'),
            validation_steps=num_valid_samples // batch_size,
            shuffle=True,
            epochs=num_epochs,
            initial_epoch=0, 
            callbacks=callbacks,
            verbose=1,
            use_multiprocessing=False)

best_model = 'output/feature_rest_full/models/model.126-0.9392.hdf5'
def eval():        
    (nb_classes, num_train_samples, num_valid_samples) = utils.load_config(store_folder)

    c2id = utils.load_c2id(store_folder)
    id2cdf = pd.DataFrame.from_dict(c2id,  orient='index')
    id2cdf['Id'] = id2cdf.index
    id2cdf.index = id2cdf[0]


    files = [f for f in os.listdir(config.test_image_folder)]
    df = pd.DataFrame(files, columns=['Image'])
    
    datagen = PredDataGenSequence(df, config.test_image_folder)

    base_model = build_pre_trained_feature_model()
    #smodel, base_model, head_model = siamese_net.best_model_with_weights()
    model = build_model_from_base(base_model, nb_classes)
    model.load_weights(best_model)
    result = model.predict_generator(datagen, ((len(df) + 127) // 128), verbose = 1)

    resultStrs = [generate_str(p, id2cdf) for p in result]
    df['Id'] = pd.Series(resultStrs)
    df.to_csv("sub.csv", index = False)

    print(result)

new_pred = 0.5
new_pred1 = 0.1
def generate_str(p, id2cdf):
    idx = np.argsort(p)[::-1][0:4]

    result = [id2cdf.loc[i]['Id'] for i in idx]
    if p[idx[0]] > new_pred:
        result.insert(1, 'new_whale')
    else:
        result.insert(0, 'new_whale')

    return ' '.join(result)


if __name__ == '__main__':
    #train()
    eval()