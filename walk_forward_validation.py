import argparse
import json
import os
import random
from joblib import Parallel, delayed
from copy import deepcopy
from main import main
from src.util import load_json
from src.data.datasets import load_train_eval_test_datasets
from src.parameters import Parameters as par


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


def get_first_fold_info(json_file: str):
    par.from_json(json_file)
    df_train, _, df_test = load_train_eval_test_datasets()
    return {'first_fold_start': df_train.index[0],
            'first_fold_end': df_test.index[-1]+1,
            'first_fold_test_size': df_test.index[-1] + 1 - df_test.index[0]}


def run_fold(first_fold_config: str, anchored: bool, fold_index: int, first_fold_info: dict):
    config = deepcopy(first_fold_config)

    shift = fold_index * first_fold_info['first_fold_test_size']
    end = first_fold_info['first_fold_end'] + shift
    start = first_fold_info['first_fold_start']

    if not anchored:
        start += shift

    config['dataset']['start'] = start
    config['dataset']['end'] = end
    config['dataset']['length'] = None
    config['dataset']['name'] += f'_fold{fold_index+1}'
    
    if anchored:
        config['dataset']['test_proportion'] = first_fold_info['first_fold_test_size']/(end-start)

    temporay_main(config)


def run_folds(json_file: str, n_folds: int, anchored: bool, jobs: int):
    first_fold_config = load_json(json_file)
    if 'path' not in first_fold_config['dataset']:
        raise ValueError('Dataset path is missing from configuration file')
    first_fold_info = get_first_fold_info(json_file)

    def process(i):
        run_fold(first_fold_config=first_fold_config, anchored=anchored,
                 fold_index=i, first_fold_info=first_fold_info)

    Parallel(n_jobs=jobs)(delayed(process)(i) for i in range(n_folds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--n_folds', default=None, type=int)
    parser.add_argument('--anchored', action=argparse.BooleanOptionalAction)
    parser.add_argument('--jobs', default=1, type=int)
    args = parser.parse_args()
    run_folds(args.config, args.n_folds, args.anchored, args.jobs)
