import os
import utils

img_width, img_height = 299, 299
num_channels = 3

batch_size = 16
num_epochs = 100
patience = 50
verbose = 1

test_image_folder = '../data/test'
train_image_folder = '../data/train'
samples_file = '../data/samples.csv'
valid_file = '../data/valid.csv'
origin_train_label_file = '../data/train.csv'

FREEZE_LAYERS = 2
dropout_rate = 0.2

#data config
data_config = utils.load_obj('data_config')
nb_classes = data_config['nb_classes']
num_train_samples = data_config['num_train_samples']
num_valid_samples = data_config['num_valid_samples']