import math

from src.util import load_json


def assign_attributes(object, d: dict):
    for k in d:
        setattr(object, k, d[k])


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

    def set(model, dataset,
            frequency_q_learning, Q_learning_iterations, discount_factor_Q_learning,
            batch_size_replay_sampling, frequency_target_exchange=-1,
            environment={}, same_age_replay=True, random_gamma=False, train=True,
            adjacent_episodes=False, frequency_of_weight_sampling=1,
            learning_rate_Q_learning=0.7, replay_max_len=25000,
            max_possible_gamma='auto', long_positions_only=True,
            number_virtual_steps='auto', plot_frequency=10, window_size=60,
            rewards="all", episode_length=None, **kwargs):

        # model
        Parameters.model = ModelParameters(**model)

        # dataset
        Parameters.dataset = DatasetParameters(**dataset)

        # environment
        Parameters.environment = EnvironmentParameters(**environment)

        # RL
        Parameters.window_size = window_size
        Parameters.frequency_q_learning = frequency_q_learning
        Parameters.train = train
        Parameters.frequency_of_weight_sampling = frequency_of_weight_sampling
        Parameters.random_gamma = random_gamma
        Parameters.episode_length = episode_length
        Parameters.adjacent_episodes = adjacent_episodes
        Parameters.same_age_replay = same_age_replay
        Parameters.long_positions_only = long_positions_only
        Parameters.batch_size_replay_sampling = batch_size_replay_sampling
        Parameters.min_possible_gamma = discount_factor_Q_learning * 0.9

        Parameters.set_max_possible_gamma(
            max_possible_gamma, discount_factor_Q_learning)

        Parameters.long_positions_only = long_positions_only

        Parameters.Q_learning_iterations = Q_learning_iterations
        Parameters.number_iterations = Parameters.start_greedy_shift + Parameters.Q_learning_iterations

        Parameters.set_frequency_target_exchange(frequency_target_exchange)

        Parameters.learning_rate_Q_learning = learning_rate_Q_learning
        Parameters.discount_factor_Q_learning = discount_factor_Q_learning

        Parameters.set_epochs_for_Q_Learning_fit()

        Parameters.plot_frequency = plot_frequency

        Parameters.set_rewards(rewards)
        Parameters.set_number_virtual_steps(number_virtual_steps)
        Parameters.set_max_replay_length(replay_max_len)

        assign_attributes(Parameters, kwargs)

    def set_max_possible_gamma(max_possible_gamma, discount_factor_Q_learning):
        if max_possible_gamma == 'auto':
            Parameters.max_possible_gamma = discount_factor_Q_learning + \
                0.9 * (1 - discount_factor_Q_learning)
        else:
            Parameters.max_possible_gamma = 1

    def set_max_replay_length(replay_max_len):
        Parameters.replay_max_len = replay_max_len
        if Parameters.same_age_replay:
            Parameters.replay_max_len = Parameters.replay_max_len * \
                (Parameters.number_virtual_steps + 1)

    def set_rewards(rewards):
        if isinstance(rewards, list):
            Parameters.rewards = rewards
        else:
            Parameters.rewards = [rewards]

    def set_number_virtual_steps(number_virtual_steps):
        if number_virtual_steps == 'auto':
            Parameters.number_virtual_steps = len(Parameters.rewards) - 1

        # When there is only one reward, the augmentation is not necessary
        if len(Parameters.rewards) == 1:
            Parameters.number_virtual_steps = 0

    def set_frequency_target_exchange(frequency_target_exchange):
        # frequency_target_exchange is set to after each episode (-1)
        Parameters.frequency_target_exchange = frequency_target_exchange
        if frequency_target_exchange < 0 and (Parameters.episode_length is None):
            # increase default frequency if full dataset is used as episode
            Parameters.frequency_target_exchange = 4

    def set_epochs_for_Q_Learning_fit():
        if Parameters.model.epochs_for_Q_Learning_fit == 'auto':
            Parameters.model.epochs_for_Q_Learning_fit = math.ceil(
                Parameters.TRAINING_INTENSITY
                * (Parameters.model.batch_size_for_learning
                   / Parameters.batch_size_replay_sampling))

    @staticmethod
    def from_json(json: str):
        return Parameters.set(**load_json(json))


class ModelParameters():

    def __init__(self, batch_size_for_learning, scale_NN=1, l2_penalty=0,
                 dropout_level=0, epochs_for_Q_Learning_fit='auto',
                 **kwargs):

        self.scale_NN = scale_NN
        self.batch_size_for_learning = batch_size_for_learning
        self.l2_penalty = l2_penalty
        self.dropout_level = dropout_level
        self.epochs_for_Q_Learning_fit = epochs_for_Q_Learning_fit

        assign_attributes(self, kwargs)


class DatasetParameters():

    def __init__(self, name, path=None, column_price=None, column_time=None,
                 start_end=[0, None], length=None,
                 var=0., eval_proportion=0.2, test_proportion=0.2,
                 **kwargs):

        self.name = name
        self.path = path
        self.column_price = column_price
        self.column_time = column_time

        self.start, self.end = start_end
        self.length = length
        if length is None:
            self.length = self.end - self.start
        elif self.end is None:
            self.end = self.length

        self.eval_proportion = eval_proportion
        self.test_proportion = test_proportion
        self.var = var

        assign_attributes(self, kwargs)


class EnvironmentParameters():

    def __init__(self, trade_fee_bid=0.0, trade_fee_ask=0.0,
                 **kwargs):

        self.trade_fee_bid = trade_fee_bid
        self.trade_fee_ask = trade_fee_ask

        assign_attributes(self, kwargs)
