import pickle
import os
import config
import numpy as np



def save_obj(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save_config(store_folder, nb_classes, num_train_samples, num_valid_samples):
    data_config = {'nb_classes': nb_classes, 'num_train_samples':num_train_samples, 'num_valid_samples':num_valid_samples}
    print(data_config)
    save_obj(data_config, os.path.join(store_folder, config.config_file_name))

def load_config(store_folder):
    data_config = load_obj(os.path.join(store_folder, config.config_file_name))
    return data_config['nb_classes'], data_config['num_train_samples'], data_config['num_valid_samples']

def save_c2id(store_folder, c2id):
    save_obj(c2id, os.path.join(store_folder, config.c2id_file_name))

def load_c2id(store_folder):
    return load_obj(os.path.join(store_folder, config.c2id_file_name))

def init_output_folder(store_folder):
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    model_path = os.path.join(store_folder, 'models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

def get_best_model(store_folder):
    import re
    pattern = 'model.(?P<epoch>\d+)-(?P<val_acc>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)

    model_path = os.path.join(store_folder, 'models')
    files = [f for f in os.listdir(model_path) if p.match(f)]
    filename = None
    epoch = None
    if len(files) > 0:
        files.sort()
        accs = [float(p.match(f).groups()[1]) for f in files]
        best_index = np.argmax(accs)

        filename = os.path.join(model_path, files[best_index])
        print('loading best model: {}'.format(filename))
        return filename

