import pickle
import json


def v2d(names: list):
    return {v: eval(v) for v in names}


def save_pickle(file_name: str, object):
    with open(file_name, 'wb') as f:
        pickle.dump(object, f)


def load_pickle(file_name: str):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def load_json(file: str):
    with open(file) as json_file:
        d = json.load(json_file)
    return d
