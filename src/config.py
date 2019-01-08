import os

img_width, img_height = 299, 299
num_channels = 3

batch_size = 16
num_epochs = 1000
patience = 50
verbose = 1

test_image_folder = '../data/test'
train_image_folder = '../data/train'
samples_file = '../data/samples.csv'
valid_file = '../data/valid.csv'
origin_train_label_file = '../data/train.csv'

FREEZE_LAYERS = 2
dropout_rate = 0.2

is_test = False
model_name = 'rest_pca'#'vgg16'