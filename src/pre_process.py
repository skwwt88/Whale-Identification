import pandas as pd
import numpy as np
import os
import utils
import config

from config import origin_train_label_file
from sklearn.utils import shuffle

def extract_classes(df):
    return [c for (c, g) in df.groupby(['Id'])]

def generate_c2id(classes):
    classes = np.array(classes)
    np.sort(classes)
    return dict((c, i) for (i, c) in enumerate(list(classes)))

def data_prepare_feature_extraction(store_folder, min_sample_count = 4, valid_ratio = 0.2):
    df = pd.read_csv(origin_train_label_file)
    df = df[df['Id'] != 'new_whale']
    df = df.groupby(['Id']).filter(lambda x: len(x) >= min_sample_count)
    df = shuffle(df).reset_index(drop=True)

    # print(df)
    valid_item_count = int(len(df) * valid_ratio)
    valid_df = df.loc[:valid_item_count]
    train_df = df.loc[valid_item_count:]
    
    valid_df.to_csv(os.path.join(store_folder, config.valid_file_name), index=False)
    train_df.to_csv(os.path.join(store_folder, config.train_file_name), index=False)

    classes = extract_classes(df)
    c2id = generate_c2id(classes)

    utils.save_c2id(store_folder, c2id)
    utils.save_config(store_folder, len(df.groupby(['Id'])), len(train_df), len(valid_df))

def data_prepare_feature_extraction_full(store_folder, valid_ratio = 0.2):
    df = pd.read_csv(origin_train_label_file)
    df = df[df['Id'] != 'new_whale']
    df = shuffle(df).reset_index(drop=True)

    # print(df)
    valid_item_count = int(len(df) * valid_ratio)
    valid_df = df.loc[:valid_item_count]
    train_df = df.loc[:]
    
    valid_df.to_csv(os.path.join(store_folder, config.valid_file_name), index=False)
    train_df.to_csv(os.path.join(store_folder, config.train_file_name), index=False)

    classes = extract_classes(df)
    c2id = generate_c2id(classes)

    utils.save_c2id(store_folder, c2id)
    utils.save_config(store_folder, len(df.groupby(['Id'])), len(train_df), len(valid_df))

def data_prepare_tokonizer_test(store_folder, filter=lambda x: len(x) == 10):
    df = pd.read_csv(origin_train_label_file)
    df = df[df['Id'] != 'new_whale']
    df = df.groupby(['Id']).filter(filter)
    df = shuffle(df).reset_index(drop=True)

    df.to_csv(os.path.join(store_folder, config.train_file_name), index=False)

def data_prepare_predict():
    utils.init_output_folder(config.predict_data_folder)
    df = pd.read_csv(origin_train_label_file)
    df.to_csv(os.path.join(config.predict_data_folder, config.train_file_name))
    


if __name__ == '__main__':
    #data_prepare_feature_extraction('output/test', 2)
    #data_prepare_predict()
    data_prepare_feature_extraction('output/siamese_folder_train_wide')