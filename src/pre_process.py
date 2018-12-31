import pandas as pd
import numpy as np
import random
from config import origin_train_label_file, samples_file, valid_file
from utils import save_obj
from sklearn.utils import shuffle

def split_samples_for_test():
    df = pd.read_csv(origin_train_label_file)
    df = shuffle(df[df.Id != 'new_whale']).reset_index(drop=True)
    grouped = df.groupby(["Id"])
    df = grouped.filter(lambda x: len(x) >= 10).reset_index(drop=True)
    num_all_samples = len(df)
    num_valid_samples = int(num_all_samples * 0.2)
    train_samples = df.iloc[num_valid_samples:]
    valid_samples = df.iloc[:num_valid_samples]
    num_train_samples = len(train_samples)
    num_valid_samples = len(valid_samples)
    classes = [c for (c, g) in df.groupby(['Id'])]
    nb_classes = len(classes);

    valid_samples.to_csv(valid_file, index=False)
    train_samples.to_csv(samples_file, index=False)

    data_config = {'nb_classes': nb_classes, 'num_train_samples':num_train_samples, 'num_valid_samples':num_valid_samples}
    save_obj(data_config, 'data_config')
    print(data_config)

    c2id = dict((c, i) for (i, c) in enumerate(list(classes)))
    save_obj(c2id, 'c2id')

if __name__ == '__main__':
    split_samples_for_test()