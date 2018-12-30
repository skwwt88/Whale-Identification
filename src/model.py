from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model

from config import img_height, img_width, num_channels, nb_classes, FREEZE_LAYERS, dropout_rate


def build_model():
    base_model = InceptionResNetV2(input_shape=(img_height, img_width, num_channels), weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate, name='Dropout')(x)
    x = Dense(nb_classes, name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable = True
    for layer in model.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    return model

def build_model_VGG(with_dropout=True):
    kwargs     = {'activation':'relu', 'padding':'same'}
    conv_drop  = 0.2
    dense_drop = 0.5
    img_shape  = (img_height, img_width, num_channels)
    inp        = Input(shape=img_shape)

    x = inp

    x = Conv2D(64, (9, 9), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(dense_drop, name='Dropout')(x)
    x = Dense(nb_classes, name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)
    model = Model(inputs=inp, outputs=x)

    return model
if __name__ == '__main__':
    build_model_VGG().summary()