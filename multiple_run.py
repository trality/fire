from dataclasses import dataclass
import os
import sys
from main import main
import json
import random
from src.util import load_json


def list_full_dir(path: str):
    return [os.path.join(path, file) for file in os.listdir(path)]


def name(dataset, reward_set):
    rname = 'm'
    if (isinstance(reward_set, list) and len(reward_set) == 1):
        reward_set = reward_set[0]
    if not isinstance(reward_set, list):
        rname = 's_' + reward_set

    return f"{rname}_{dataset[dataset.find('/')+1:dataset.find('.')-2]}"


def temporay_main(config: dict):
    temporary_folder = f"/var/tmp/{hash(random.random())}"
    os.mkdir(temporary_folder)
    temporary_json_name = os.path.join(temporary_folder, 'config.json')

    with open(temporary_json_name, 'w') as f:
        json.dump(config, f, indent=4)
    try:
        main(temporary_json_name)
    finally:
        os.remove(temporary_json_name)
        os.rmdir(temporary_folder)
        

def run_comb(template, dataset, reward_set):
    # json template
    config = load_json(template)
    config['dataset']['path'] = dataset
    config['dataset']['name'] = name(dataset, reward_set)
    config['rewards'] = reward_set
    
    temporay_main(config)


def dir_mode():
    for config in list_full_dir(sys.argv[1]):
        main(ic(config))


def comb_mode(template, dataset):
    REWARD_SET_MULTI = ["LR", "SR", "ALR", "POWC"]
    REWARD_SINGLE = ["SR", "POWC"]
    run_comb(template, dataset, REWARD_SET_MULTI)
    for r in REWARD_SINGLE:
        run_comb(template, dataset, r)
        

if __name__ == '__main__':
    template = sys.argv[1]
    dataset = sys.argv[2]
    comb_mode(template, dataset)
