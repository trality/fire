import numpy as np
import pandas as pd

import src.data.synthetic as syndata
from src.parameters import Parameters as par
import src.constants as const


def fetch_dataset():
    if par.dataset.name not in syndata.datasets:
        print(f"Dataset not in predefined sets {syndata.datasets}")
        print("Trying to load custom dataset:")
        return load_dataset()
    else:
        return getattr(syndata, par.dataset.name)()


def load_dataset():
    df = pd.read_csv(f"{const.REFERENCE_DIRECTORY}/{par.dataset.path}")
    df = df.rename({par.dataset.column_price: 'price'}, axis='columns')
    return slice_df(df)


def slice_df(df: pd.DataFrame):
    if not par.dataset.end:
        return df
    if isinstance(par.dataset.start, str) and isinstance(par.dataset.end, str):
        return df.set_index(par.dataset.time_column or 'start').loc[par.dataset.start:par.dataset.end]
    if isinstance(par.dataset.start, int) and isinstance(par.dataset.end, int):
        if len(df) < par.dataset.end:
            raise ValueError('par.dataset.end exceeds dataset length')
        return df.iloc[par.dataset.start:par.dataset.end]
        

def split_df(df: pd.DataFrame, cut: float):
    split_point = int(len(df) * cut)
    return df.iloc[:split_point], df.iloc[split_point:]


def load_train_eval_test_datasets():
    df = fetch_dataset()

    eval_test_prop = par.dataset.eval_proportion + par.dataset.test_proportion
    train_prop = 1 - eval_test_prop
    eval_rest_prop = par.dataset.eval_proportion / eval_test_prop
    df_train, df_rest = split_df(df, train_prop)
    df_eval, df_test = split_df(df_rest, eval_rest_prop)

    assert len(df_train) > 0 and len(df_eval) > 0 and len(df_test) > 0

    return df_train, df_eval, df_test


def sample_episode(par, df: pd.DataFrame, episode: int):
    if par.episode_length >= len(df):
        raise ValueError('episode length is larger than train dataset')

    if not par.adjacent_episodes:
        episode_starting_point = np.random.choice(
            np.arange(len(df) - par.episode_length))
    else:
        episode_starting_point = (episode * par.episode_length) % len(df)

    episode_end_point = episode_starting_point + par.episode_length

    print(f"episode: {episode_starting_point}->{episode_end_point}")
    return df.iloc[episode_starting_point:episode_end_point]
