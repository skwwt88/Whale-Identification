import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from config import batch_size, patience, num_epochs, verbose, model_name, dropout_rate
from data_generator import DataGenSequence
from model import build_model
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.models import Model
import utils

tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
early_stop = EarlyStopping('val_acc', patience=patience)
reduce_lr = ReduceLROnPlateau('val_acc', factor=0.5, patience=int(patience / 4), verbose=1)
trained_models_path = '../models/model.' + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.4f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
callbacks = [tensor_board, model_checkpoint, early_stop,  ]


new_model = build_model(best_model='vgg19')
new_model.load_weights('../models/base/model.vgg19.0.5663.hdf5')

'''
base_model = build_model(utils.load_obj('model.inception_res.head'))
base_model.load_weights('../models/base/model.inception_res.head.0.9544.hdf5')

for layer in base_model.layers:
        layer.trainable = False
x = base_model.layers[-5].output
x = GlobalAveragePooling2D()(x)
x = Dropout(dropout_rate, name='Dropout')(x)
x = Dense(utils.nb_classes, name='Logits')(x)
x = Activation('softmax', name='Predictions')(x)
new_model = Model(inputs=base_model.input, outputs=x)
'''

sgd = keras.optimizers.SGD(lr=2.5e-4, momentum=0.9, decay=1e-6, nesterov=True)
new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

new_model.fit_generator(
        DataGenSequence('train'),
        steps_per_epoch=utils.num_train_samples // batch_size,
        validation_data=DataGenSequence('valid'),
        validation_steps=utils.num_valid_samples // batch_size,
        shuffle=True,
        epochs=num_epochs,
        initial_epoch=0, 
        callbacks=callbacks,
        verbose=verbose,
        use_multiprocessing=False)
