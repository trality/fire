import pandas as pd
import numpy as np

from src.parameters import Parameters as par
from src.data.util import produce_noise_one_off


# Keep a list of all implemented synthetic datasets
datasets = ["v_shape", "sin_wave", "square_wave", "noisy_sin_wave",
            "noisy_v_shape", "straight_line", "autoregressive"]


def v_shape():
    times = pd.date_range(
        '2012-10-01', periods=par.dataset.length, freq='1d')
    y = [abs(par.dataset.length / 2 - i)
         + 1 for i in range(par.dataset.length)]
    df = pd.DataFrame(index=times, data={"price": y})
    return df


def noisy_v_shape():
    times = pd.date_range(
        '2012-10-01', periods=par.dataset.length, freq='1d')
    max = np.max([abs(par.dataset.length / 2 - i)
                  + 1 for i in range(par.dataset.length)])
    y = [abs(par.dataset.length / 2 - i) + 1
         + max * np.random.normal(0, par.dataset.var)
         for i in range(par.dataset.length)]
    df = pd.DataFrame(index=times, data={"price": y})
    return df


def sin_wave():
    times = pd.date_range(
        '2012-10-01', periods=par.dataset.length, freq='1d')
    y = 2 + np.sin(np.arange(par.dataset.length)
                   / par.dataset.length * 16 * np.pi)
    df = pd.DataFrame(index=times, data={"price": y})
    return df


def square_wave():
    times = pd.date_range(
        '2012-10-01', periods=par.dataset.length, freq='1d')
    y = (np.arange(par.dataset.length) % 20 < 10)
    df = pd.DataFrame(index=times, data={"price": y})
    return df


def noisy_sin_wave():
    times = pd.date_range(
        '2012-10-01', periods=par.dataset.length, freq='1d')
    min = np.min(2 + np.sin(np.arange(par.dataset.length)
                            / par.dataset.length * 8 * np.pi))
    noise_vector = produce_noise_one_off(par)
    y = [2 + np.sin(i / par.dataset.length * 8 * np.pi)
         + min * noise_vector[i] for i in range(par.dataset.length)]
    df = pd.DataFrame(index=times, data={"price": y})
    return df


def straight_line():
    times = pd.date_range(
        '2012-10-01', periods=par.dataset.length, freq='1d')
    y = [i for i in range(par.dataset.length)]
    df = pd.DataFrame(index=times, data={"price": y})
    return df


def random_walk():
    y = np.exp(np.cumsum(np.random.normal(
        0, 0.01, par.dataset.length)))
    return pd.DataFrame({'price': y})


def autoregressive():
    times = pd.date_range(
        '2012-10-01', periods=par.dataset.length, freq='1d')
    beta = np.zeros((par.dataset.length))
    z = np.zeros((par.dataset.length))
    beta[0] = 1
    z[0] = 1

    rng = np.random.default_rng(2021)
    a = rng.standard_normal(par.dataset.length)
    b = rng.standard_normal(par.dataset.length)

    for k in range(par.dataset.length - 1):
        beta[k + 1] = 0.9 * beta[k] + b[k + 1]
    for k in range(par.dataset.length - 1):
        z[k + 1] = z[k] + beta[k] + 3 * a[k + 1]
    y = np.exp(z / (np.max(z) - np.min(z)))
    df = pd.DataFrame(index=times, data={"price": y})
    return df
