import os

img_width, img_height = 299, 299
num_channels = 3

batch_size = 16
num_epochs = 100
patience = 50
verbose = 1
num_train_samples = 4354
num_valid_samples = 200

train_image_folder = '../data/train'
train_label_file = '../data/train.csv'

FREEZE_LAYERS = 2
dropout_rate = 0.2
num_classes = 227


