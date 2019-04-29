import pickle
import os

def save_obj(obj, name):
    if not os.path.exists('obj/'):
        os.makedirs('obj/')
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)