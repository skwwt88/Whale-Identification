from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model

import utils
from config import img_height, img_width, num_channels, FREEZE_LAYERS, dropout_rate, model_name


def build_model_inception_restnet(data_config):
    nb_classes = utils.nb_classes
    if not (data_config is None):
        nb_classes = data_config['nb_classes']

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

def build_model_inception_restnet_with_nb_classes(nb_classes):

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

def build_model_inception_restnet_pca(data_config):
    nb_classes = utils.nb_classes
    if not (data_config is None):
        nb_classes = data_config['nb_classes']

    base_model = InceptionResNetV2(input_shape=(img_height, img_width, num_channels), weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.1, name='Dropout_PCA')(x)
    x = Dense(100, name='PCA')(x)
    x = Dropout(0.1, name='Dropout')(x)
    x = Dense(nb_classes, name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)

    return model


def build_model_vgg16(data_config):
    nb_classes = utils.nb_classes
    if not (data_config is None):
        nb_classes = data_config['nb_classes']

    base_model = VGG16(input_shape=(img_height, img_width, num_channels), weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate, name='Dropout')(x)
    x = Dense(nb_classes, name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in model.layers:
        layer.trainable = True

    return model

def build_model_vgg19():
    base_model = VGG19(input_shape=(img_height, img_width, num_channels), weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate, name='Dropout')(x)
    x = Dense(utils.nb_classes, name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in model.layers:
        layer.trainable = True

    return model

def build_model(data_config = None, best_model = None):
    current_model = model_name
    if not (best_model is None):
        current_model = best_model

    if current_model == 'vgg16':
        return build_model_vgg16()
    elif current_model == 'vgg19':
        return build_model_vgg19()
    elif current_model == 'rest_pca':
        return build_model_inception_restnet_pca(data_config)
    else:
        return build_model_inception_restnet(data_config)

if __name__ == '__main__':
    build_model(utils.load_obj('model.inception_res.head')).summary()
    