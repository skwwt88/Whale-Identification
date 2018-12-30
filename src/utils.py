import pickle

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    test_obj = {"a":1}
    save_obj(test_obj, 'test_obj')
    test_obj = load_obj('test_obj')
    print(test_obj)