import pickle
import os
import numpy as np
from config import model_name

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_best_model():
    import re
    pattern = 'model.' + model_name + '.(?P<epoch>\d+)-(?P<val_acc>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    files = [f for f in os.listdir('../models/') if p.match(f)]
    filename = None
    epoch = None
    if len(files) > 0:
        epoches = [p.match(f).groups()[0] for f in files]
        accs = [float(p.match(f).groups()[1]) for f in files]
        best_index = int(np.argmax(accs))
        filename = os.path.join('../models', files[best_index])
        epoch = int(epoches[best_index])
        print('loading best model: {}'.format(filename))
    return filename, epoch

def get_base_model():
    filename = os.path.join('../models/base/model.' + model_name + '.hdf5')
    return filename

data_config = load_obj('data_config')
nb_classes = data_config['nb_classes']
num_train_samples = data_config['num_train_samples']
num_valid_samples = data_config['num_valid_samples']

if __name__ == "__main__":
    test_obj = {"a":1}
    save_obj(test_obj, 'test_obj')
    test_obj = load_obj('test_obj')
    print(test_obj)

    get_best_model()
    print(get_base_model())