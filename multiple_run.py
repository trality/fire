from dataclasses import dataclass
import os
import sys
from main import main
import json
import random
from src.util import load_json
from icecream import ic


def list_full_dir(path: str):
    return [os.path.join(path, file) for file in os.listdir(path)]

def name(dataset, reward_set):
    rname = 'm'
    if(isinstance(reward_set, list) and len(reward_set)==1):
        reward_set = reward_set[0]
    if not isinstance(reward_set, list):
        rname = 's_' + reward_set
        
    return f"{rname}_{dataset[dataset.find('/')+1:dataset.find('.')-2]}"

def run_comb(dataset, reward_set):
    # json template
    json_file = sys.argv[1]
    config = load_json(json_file)
    config['dataset']['path'] = dataset
    config['dataset']['name'] = name(dataset, reward_set)
    config['rewards'] = reward_set
    
    #temp_name = tempfile.NamedTemporaryFile().name
    
    temp_name = f".temporary_{random.randint(100000, 1000000)}.json"
    with open(temp_name, 'w') as fp:
        json.dump(config, fp, indent=4)
        
    main(temp_name)
        
def dir_mode():
    for config in list_full_dir(sys.argv[1]):
        main(ic(config))

def comb_mode():
    DATASETS = ["datasets/eth_h.csv", "datasets/xrp_h.csv", "datasets/btc_h.csv"]
    REWARD_SET_MULTI = ["LR", "SR", "ALR", "POWC"]
    REWARD_SINGLE = ["SR", "POWC"]
    
    for d in DATASETS:
        run_comb(d, REWARD_SET_MULTI)
        for r in REWARD_SINGLE:
            run_comb(d, r)
            
"""
def comb_mode():
    DATASETS = ["datasets/aapl_d.csv", "datasets/spy_d.csv"]
    REWARD_SET_MULTI = ["LR", "SR", "ALR", "POWC"]
    REWARD_SINGLE = ["SR", "POWC"]
    
    for d in DATASETS:
        run_comb(d, REWARD_SET_MULTI)
        for r in REWARD_SINGLE:
            run_comb(d, r)
"""


               
if __name__ == '__main__':
    comb_mode()
