import utils
import os
import keras

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from config import img_height, img_width, num_channels
from data_generator import DataGenSequence
from pre_process import data_prepare_feature_extraction, data_prepare_feature_extraction_full

batch_size = 32
num_epochs = 1000
patience = 50
dropout_rate = 0.2

best_model = 'output/feature_rest_full/models/model.205-0.8661.hdf5'

def build_feature_model(freeze = False):
    base_model = InceptionResNetV2(input_shape=(img_height, img_width, num_channels), weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    
    for layer in model.layers:
        layer.trainable = not freeze
    
    return model

def build_model_from_base(base_model, nb_classes):

    x = base_model.output
    x = Dropout(dropout_rate, name='Dropout')(x)
    x = Dense(nb_classes, name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)
    
    return Model(inputs=base_model.input, outputs=x)

feature_data_store_folder_test = 'output/feature_rest_test'
def train_test(from_prev_model = False):
    if not os.path.isdir(feature_data_store_folder_test):
        utils.init_output_folder(feature_data_store_folder_test)
        data_prepare_feature_extraction(feature_data_store_folder_test, min_sample_count=4)

    (nb_classes, num_train_samples, num_valid_samples) = utils.load_config(feature_data_store_folder_test)

    base_model = build_feature_model()
    base_model.name = 'base_model'
    model = build_model_from_base(base_model, nb_classes)
    if from_prev_model:
        best_model = utils.get_best_model(feature_data_store_folder_test)
        model.load_weights(best_model)

    trained_models_path = os.path.join(feature_data_store_folder_test, 'models')
    model_names = trained_models_path + '/model.{epoch:03d}-{val_acc:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_acc', patience=patience)
    callbacks = [model_checkpoint, early_stop, ]

    sgd = keras.optimizers.SGD(lr=2.5e-4, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(
            DataGenSequence(feature_data_store_folder_test, 'train'),
            steps_per_epoch=num_train_samples // batch_size,
            validation_data=DataGenSequence(feature_data_store_folder_test, 'valid'),
            validation_steps=num_valid_samples // batch_size,
            shuffle=True,
            epochs=num_epochs,
            initial_epoch=0, 
            callbacks=callbacks,
            verbose=1,
            use_multiprocessing=False)

feature_data_store_folder_full = 'output/feature_rest_full'
def train(from_prev_model = False):
    if not os.path.isdir(feature_data_store_folder_full):
        utils.init_output_folder(feature_data_store_folder_full)
        data_prepare_feature_extraction_full(feature_data_store_folder_full)
    (nb_classes, num_train_samples, num_valid_samples) = utils.load_config(feature_data_store_folder_full)

    base_model = build_feature_model()
    model = build_model_from_base(base_model, nb_classes)
    if from_prev_model:
        best_model = utils.get_best_model(feature_data_store_folder_full)
        model.load_weights(best_model)

    trained_models_path = os.path.join(feature_data_store_folder_full, 'models')
    model_names = trained_models_path + '/model.{epoch:03d}-{val_acc:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=False)
    callbacks = [model_checkpoint]

    sgd = keras.optimizers.SGD(lr=2.5e-4, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(
            DataGenSequence(feature_data_store_folder_full, 'train'),
            steps_per_epoch=num_train_samples // batch_size,
            validation_data=DataGenSequence(feature_data_store_folder_full, 'valid'),
            validation_steps=num_valid_samples // batch_size,
            shuffle=True,
            epochs=num_epochs,
            initial_epoch=0, 
            callbacks=callbacks,
            verbose=1,
            use_multiprocessing=False)

def build_pre_trained_feature_model():
    (nb_classes, num_train_samples, num_valid_samples) = utils.load_config(feature_data_store_folder_full)

    feature_model = build_feature_model(True)
    full_model = build_model_from_base(feature_model, nb_classes)
    full_model.load_weights(best_model)

    return feature_model




if __name__ == '__main__':
    train(True)

