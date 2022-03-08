import numpy as np

from src.parameters import Parameters as par


def produce_noise_one_off():
    rand = np.random.default_rng(2022)
    return rand.normal(0, par.dataset.var, par.dataset.length)
