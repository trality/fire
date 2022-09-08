import math
from typing import List, Dict

from src.util import load_json


def set_max_possible_gamma(Parameters, max_possible_gamma, discount_factor_Q_learning):
    if max_possible_gamma == 'auto':
        Parameters.max_possible_gamma = discount_factor_Q_learning + \
            0.9 * (1 - discount_factor_Q_learning)
    else:
        Parameters.max_possible_gamma = 1


def set_rewards(Parameters, rewards):
    if isinstance(rewards, list):
        Parameters.rewards = rewards
    else:
        Parameters.rewards = [rewards]


def set_frequency_target_exchange(Parameters, frequency_target_exchange):
    # frequency_target_exchange is set to after each episode (-1)
    Parameters.frequency_target_exchange = frequency_target_exchange
    if frequency_target_exchange < 0 and (Parameters.episode_length is None):
        # increase default frequency if full dataset is used as episode
        Parameters.frequency_target_exchange = 4


def set_epochs_for_Q_Learning_fit(Parameters):
    if Parameters.model.epochs_for_Q_Learning_fit == 'auto':
        Parameters.model.epochs_for_Q_Learning_fit = math.ceil(
            Parameters.TRAINING_INTENSITY
            * (Parameters.model.batch_size_for_learning
               / Parameters.batch_size_replay_sampling))


class Parameters():
    TRAINING_INTENSITY = 10

    # Graphics
    dpi_res = 200

    style_train = 'solid'
    style_partial_train = (0, (3, 5, 1, 5))
    style_prog = '--'
    style_eval = 'dashdot'
    style_test = (0, (3, 5, 1, 5))

    style_train_BH = (0, (5, 5))
    style_eval_BH = 'dotted'
    style_test_BH = (0, (3, 1, 1, 1, 1, 1))

    style_train_neutral = style_train
    style_eval_neutral = style_eval

    marker_train = None
    marker_eval = None
    marker_test = None

    marker_train_neutral = "."
    marker_eval_neutral = "."

    marker_train_BH = None
    marker_eval_BH = None
    marker_test_BH = None

    color_train = 'red'
    color_eval = 'forestgreen'
    color_test = 'blue'

    # RL
    start_greedy_shift = 0
    max_eps = 1
    min_eps = 0.01
    decay = 0.01

    @staticmethod
    def set(
        frequency_q_learning: int,
        Q_learning_iterations: int,
        discount_factor_Q_learning: float,
        batch_size_replay_sampling: int,
        frequency_target_exchange: int = -1,
        dataset: Dict = {},
        model: Dict = {},
        environment: Dict = {},
        random_gamma: bool = False,
        train: bool = True,
        adjacent_episodes: bool = False,
        frequency_of_weight_sampling: int = 1,
        learning_rate_Q_learning: float = 0.7,
        replay_max_len: int = 25000,
        max_possible_gamma: str | float = 'auto',
        long_positions_only: bool = True,
        plot_frequency: int = 10,
        window_size: int = 60,
        rewards: str | List[str] = "all",
        episode_length: int | None = None,
    ):

        Parameters.dataset = DatasetParameters(**dataset)
        Parameters.model = ModelParameters(**model)
        Parameters.environment = EnvironmentParameters(**environment)

        # RL
        Parameters.window_size = window_size
        Parameters.frequency_q_learning = frequency_q_learning
        Parameters.train = train
        Parameters.frequency_of_weight_sampling = frequency_of_weight_sampling
        Parameters.random_gamma = random_gamma
        Parameters.episode_length = episode_length
        Parameters.adjacent_episodes = adjacent_episodes
        Parameters.replay_max_len = replay_max_len
        Parameters.long_positions_only = long_positions_only
        Parameters.batch_size_replay_sampling = batch_size_replay_sampling
        Parameters.min_possible_gamma = discount_factor_Q_learning * 0.9

        set_max_possible_gamma(
            Parameters, max_possible_gamma, discount_factor_Q_learning)

        Parameters.long_positions_only = long_positions_only

        Parameters.Q_learning_iterations = Q_learning_iterations
        Parameters.number_iterations = Parameters.start_greedy_shift + Parameters.Q_learning_iterations

        set_frequency_target_exchange(Parameters, frequency_target_exchange)

        Parameters.learning_rate_Q_learning = learning_rate_Q_learning
        Parameters.discount_factor_Q_learning = discount_factor_Q_learning

        set_epochs_for_Q_Learning_fit(Parameters)

        Parameters.plot_frequency = plot_frequency

        set_rewards(Parameters, rewards)

    @staticmethod
    def from_json(json: str):
        return Parameters.set(**load_json(json))


class ModelParameters():

    def __init__(
        self,
        batch_size_for_learning: int = 64,
        scale_NN: int = 1,
        l2_penalty: float = 0,
        dropout_level: float = 0,
        epochs_for_Q_Learning_fit: str = 'auto',
    ):
        self.batch_size_for_learning = batch_size_for_learning
        self.scale_NN = scale_NN
        self.l2_penalty = l2_penalty
        self.dropout_level = dropout_level
        self.epochs_for_Q_Learning_fit = epochs_for_Q_Learning_fit


class DatasetParameters():

    def __init__(
        self,
        name: str = "unkown_dataset",
        column_price: str = "close",
        time_column: str = "start",
        path: str | None = None,
        start: int | str = 0,
        end: int | str | None = None,
        length: int | None = None,
        var: float = 0.,
        eval_proportion: float = 0.2,
        test_proportion: float = 0.2,
    ):
        self.name = name
        self.column_price = column_price
        self.time_column = time_column
        self.path = path or "default_path"

        self.start = start
        self.end = end
        self.length = length
        
        if length and self.end:
            raise ValueError('You should not specify a length if start and end are given')
        if length is None and isinstance(start, int) and isinstance(end, int):
            self.length = self.end - self.start
        if self.end is None:
            self.end = self.start + self.length

        self.var = var
        self.eval_proportion = eval_proportion
        self.test_proportion = test_proportion


class EnvironmentParameters():

    def __init__(
        self,
        trade_fee_bid: float = 0.0,
        trade_fee_ask: float = 0.0,
    ):
        self.trade_fee_bid = trade_fee_bid
        self.trade_fee_ask = trade_fee_ask

