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
from pre_process import data_prepare_feature_extraction
from feature_model_rest import build_pre_trained_feature_model

store_folder = 'output/model_build_on_feature'
batch_size = 32
num_epochs = 100
patience = 50
dropout_rate = 0.2

def build_model_from_base(base_model, nb_classes):

    x = base_model.output
    x = Dropout(dropout_rate, name='Dropout')(x)
    x = Dense(nb_classes, name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)
    
    return Model(inputs=base_model.input, outputs=x)

def train():
    data_prepare_feature_extraction(store_folder, min_sample_count=0, valid_ratio=0.1)
    (nb_classes, num_train_samples, num_valid_samples) = utils.load_config(store_folder)

    base_model = build_pre_trained_feature_model()
    model = build_model_from_base(base_model, nb_classes)

    trained_models_path = os.path.join(store_folder, 'models')
    model_names = trained_models_path + '/model.{epoch:02d}-{val_acc:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_acc', patience=patience)
    callbacks = [model_checkpoint, early_stop, ]

    sgd = keras.optimizers.SGD(lr=2.5e-4, momentum=0.9, decay=1e-6, nesterov=True)
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

if __name__ == '__main__':
    train()