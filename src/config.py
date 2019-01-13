import os

img_width, img_height = 299, 299
num_channels = 3

batch_size = 16
num_epochs = 1000
patience = 50
verbose = 1

test_image_folder = '../input/humpback-whale-identification/test'
train_image_folder = '../input/humpback-whale-identification/train'
origin_train_label_file = '../input/humpback-whale-identification/train.csv'

valid_file_name = 'valid.csv'
train_file_name = 'train.csv'
config_file_name = 'config.pkl'
c2id_file_name = 'c2id.pkl'