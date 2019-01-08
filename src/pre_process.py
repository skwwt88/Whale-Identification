import pandas as pd
import numpy as np
import random
from config import origin_train_label_file, samples_file, valid_file, is_test
from utils import save_obj
from sklearn.utils import shuffle

def extend_samples(df, mincount = 3):
    grouped = df.groupby(["Id"])
    result = grouped.filter(lambda x: len(x) >= mincount).reset_index(drop=True)
    
    for k, v in grouped:
        new_count = mincount - len(v)
        length = len(v)
        if new_count > 0:
            for i in range(mincount):
                result = result.append({'Image': v["Image"].iloc(0)[i % length], 'Id': k}, ignore_index=True)

    return result


def split_samples_for_train():
    df = pd.read_csv(origin_train_label_file)
    df = shuffle(df[df.Id != 'new_whale']).reset_index(drop=True)
    grouped = df.groupby(["Id"])
    #df = grouped.filter(lambda x: len(x) >= 10).reset_index(drop=True)
    df = df.reset_index(drop=True)
    num_all_samples = len(df)
    num_valid_samples = int(num_all_samples * 0.2)
    train_samples = df.iloc[:]
    valid_samples = df.iloc[:num_valid_samples]
    num_valid_samples = len(valid_samples)
    classes = [c for (c, g) in df.groupby(['Id'])]
    nb_classes = len(classes);

    valid_samples.to_csv(valid_file, index=False)

    train_samples = extend_samples(train_samples)
    num_train_samples = len(train_samples)
    train_samples.to_csv(samples_file, index=False)

    data_config = {'nb_classes': nb_classes, 'num_train_samples':num_train_samples, 'num_valid_samples':num_valid_samples}
    save_obj(data_config, 'data_config')
    print(data_config)

    c2id = dict((c, i) for (i, c) in enumerate(list(classes)))
    save_obj(c2id, 'c2id')

def split_samples_for_train_with_validset():
    df = pd.read_csv(origin_train_label_file)
    df = shuffle(df[df.Id != 'new_whale']).reset_index(drop=True)
    grouped = df.groupby(["Id"])
    #df = grouped.filter(lambda x: len(x) >= 10).reset_index(drop=True)
    df = df.reset_index(drop=True)
    num_all_samples = len(df)
    num_valid_samples = int(num_all_samples * 0.1)
    train_samples = df.iloc[num_valid_samples:]
    valid_samples = df.iloc[:num_valid_samples]
    num_valid_samples = len(valid_samples)
    classes = [c for (c, g) in df.groupby(['Id'])]
    nb_classes = len(classes);

    valid_samples.to_csv(valid_file, index=False)

    train_samples = extend_samples(train_samples)
    num_train_samples = len(train_samples)
    train_samples.to_csv(samples_file, index=False)

    data_config = {'nb_classes': nb_classes, 'num_train_samples':num_train_samples, 'num_valid_samples':num_valid_samples}
    save_obj(data_config, 'data_config')
    print(data_config)

    c2id = dict((c, i) for (i, c) in enumerate(list(classes)))
    save_obj(c2id, 'c2id')

def split_samples_for_test():
    df = pd.read_csv(origin_train_label_file)
    df = shuffle(df[df.Id != 'new_whale']).reset_index(drop=True)
    grouped = df.groupby(["Id"])
    df = grouped.filter(lambda x: len(x) >= 5).reset_index(drop=True)
    df = df.reset_index(drop=True)
    num_all_samples = len(df)
    num_valid_samples = int(num_all_samples * 0.2)
    train_samples = df.iloc[num_valid_samples:]
    valid_samples = df.iloc[:num_valid_samples]
    classes = [c for (c, g) in df.groupby(['Id'])]
    nb_classes = len(classes);
    valid_samples.to_csv(valid_file, index=False)
    train_samples.to_csv(samples_file, index=False)

    num_train_samples = len(train_samples)
    data_config = {'nb_classes': nb_classes, 'num_train_samples':num_train_samples, 'num_valid_samples':num_valid_samples}
    save_obj(data_config, 'data_config')
    print(data_config)

    c2id = dict((c, i) for (i, c) in enumerate(list(classes)))
    save_obj(c2id, 'c2id')

def split_samples_for_build_head_model():
    df = pd.read_csv(origin_train_label_file)
    df = shuffle(df[df.Id != 'new_whale']).reset_index(drop=True)
    grouped = df.groupby(["Id"])
    df = grouped.filter(lambda x: len(x) >= 2).reset_index(drop=True)
    df = shuffle(df).reset_index(drop=True)
    df = df.reset_index(drop=True)
    num_all_samples = len(df)
    num_valid_samples = int(num_all_samples * 0.1)
    train_samples = extend_samples(df.iloc[num_valid_samples:], 4)
    valid_samples = df.iloc[:num_valid_samples]
    classes = [c for (c, g) in df.groupby(['Id'])]
    nb_classes = len(classes)
    valid_samples.to_csv(valid_file, index=False)
    train_samples.to_csv(samples_file, index=False)

    num_train_samples = len(train_samples)
    data_config = {'nb_classes': nb_classes, 'num_train_samples':num_train_samples, 'num_valid_samples':num_valid_samples}
    save_obj(data_config, 'data_config')
    print(data_config)

    c2id = dict((c, i) for (i, c) in enumerate(list(classes)))
    save_obj(c2id, 'c2id')

if __name__ == '__main__':
    #split_samples_for_build_head_model()
    #split_samples_for_test()
    #split_samples_for_train()
    split_samples_for_train_with_validset()
'''
    if is_test:
        
    else:
        split_samples_for_train()
'''