import argparse
import shutil
import time
import pathlib
import pandas as pd
from datetime import datetime
from tensorflow import keras

from src.data.datasets import load_train_eval_test_datasets
from src.environment.environmentRL import EnvironmentRL
from src.q_learning_agent import QLearningAgent
from src.parameters import Parameters as par
import src.constants as const


def run_experiment(path=None):

    if par.train:
        df_train, df_eval, df_test = load_train_eval_test_datasets(path)
        df_train.to_pickle(f'{path}/datasets/df_train.pkl')
        df_eval.to_pickle(f'{path}/datasets/df_eval.pkl')
        df_test.to_pickle(f'{path}/datasets/df_test.pkl')
    else:
        df_train = pd.read_pickle(f'{path}/datasets/df_train.pkl')
        df_eval = pd.read_pickle(f'{path}/datasets/df_eval.pkl')
        df_test = pd.read_pickle(f'{path}/datasets/df_test.pkl')

    env_train = EnvironmentRL(df_train, window_size=par.window_size,
                              path_plots=path, log_returns_scale="fit")
    env_eval = EnvironmentRL(df_eval, window_size=par.window_size,
                             path_plots=path, log_returns_scale=env_train.get_log_returns_scale())
    env_test = EnvironmentRL(df_test, window_size=par.window_size,
                             path_plots=path, log_returns_scale=env_train.get_log_returns_scale())

    agent = QLearningAgent(env_train, env_eval, env_test, path)

    if par.train:
        agent.learn()
        agent.model.save(f'{path}/model')
    else:
        agent.model = keras.models.load_model(f'{path}/model')
        agent.target_model = keras.models.load_model(f'{path}/model')


def get_datetime_string():
    return datetime.now().strftime("%y-%m-%d--%H-%M-%S")


def create_results_folder_structure():
    results_path = pathlib.Path(const.REFERENCE_DIRECTORY, const.EXPERIMENTS_FOLDER_NAME)
    results_path.mkdir(parents=True, exist_ok=True)
    exp_ref = f"{par.dataset.name}--{get_datetime_string()}"
    path_experiment = pathlib.Path(results_path, exp_ref)
    path_experiment.mkdir(parents=True, exist_ok=True)
    # create subfolders
    for reward in par.rewards:
        pathlib.Path(path_experiment, reward).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(path_experiment, reward, "plots").mkdir(
            parents=True, exist_ok=True)
    pathlib.Path(path_experiment, "datasets").mkdir(
        parents=True, exist_ok=True)
    return path_experiment


def main(json_file):

    start_time = time.time()
    print('New experiment created')
    par.from_json(json_file)  # set parameters
    path_experiment = create_results_folder_structure()
    shutil.copy(json_file, f'{path_experiment}')  # save a copy of parameters

    run_experiment(path=path_experiment)
    print('--- %s seconds ---' % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default=None)
    args = parser.parse_args()
    main(args.config)
