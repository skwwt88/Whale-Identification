import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from config import batch_size, patience, num_train_samples, num_valid_samples, num_epochs, verbose
from data_generator import DataGenSequence
from model import build_model, build_model_VGG

tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
early_stop = EarlyStopping('val_acc', patience=patience)
reduce_lr = ReduceLROnPlateau('val_acc', factor=0.5, patience=int(patience / 4), verbose=1)
trained_models_path = '../models/model'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.4f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

new_model = build_model()
sgd = keras.optimizers.SGD(lr=2.5e-4, momentum=0.9, decay=1e-6, nesterov=True)
new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

new_model.fit_generator(
        DataGenSequence('train'),
        steps_per_epoch=num_train_samples // batch_size,
        validation_data=DataGenSequence('valid'),
        validation_steps=num_valid_samples // batch_size,
        shuffle=True,
        epochs=num_epochs,
        initial_epoch=0, 
        callbacks=callbacks,
        verbose=verbose,
        use_multiprocessing=False)
