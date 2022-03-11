import os
import time
import tqdm
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from scipy import linalg

import src.plots as plots
from src.model import basic_model
from src.environment.environment import Actions
from src.environment.environment import Positions
from src.environment.environmentRL import EnvironmentRL
from src.data.datasets import sample_episode
from src.performance import sharpe_ratio_performance
from src.replay import Replay
from src.parameters import Parameters as par


def lookback_array(x: np.ndarray, lookback):
    return x[np.arange(lookback).reshape((1, -1))
             + np.arange(len(x) - lookback + 1).reshape((-1, 1))]


def dict_to_array(d: dict):
    return np.array(list(d.values()))


def sample_binomial(p):
    return np.random.binomial(1, p) == 1


class QLearningAgent():

    def __init__(self, env_train: EnvironmentRL, env_eval: EnvironmentRL,
                 env_test: EnvironmentRL, path: str):

        self.env_train = env_train
        self.env_eval = env_eval
        self.env_test = env_test

        self.fitted_log_returns_scale = self.env_train.get_log_returns_scale()

        self.path = path
        self.rewards = par.rewards

        self.initialize_model()
        self.initialize_replay()
        self.initialize_performance_metric()
        self.initialize_info_dataframe()
        self.initialize_best_model_info()

    def initialize_replay(self):
        self.replay = Replay(par.replay_max_len)

    def initialize_best_model_info(self):
        self.max_performance_eval_target_model = {
            r: -np.inf for r in self.rewards}
        self.best_model_info = {r: [] for r in self.rewards}

    def initialize_performance_metric(self):
        self.performance = sharpe_ratio_performance
        self.train_buy_and_hold_performance = self.performance.get_buy_and_hold(
            self.env_train)
        self.eval_buy_and_hold_performance = self.performance.get_buy_and_hold(
            self.env_eval)

    def initialize_model(self):
        self.model = self.get_model()
        self.target_model = self.get_model()
        print(self.model.summary())

    def initialize_info_dataframe(self):
        episode_index = np.arange(par.number_iterations)
        columns = (('train', 'eval'),
                   self.rewards,
                   ('average_reward', 'performance', 'average_long', 'average_short'))
        columns_index = pd.MultiIndex.from_product(columns)
        self.info_dataframe = pd.DataFrame(
            index=episode_index, columns=columns_index)

    def get_model(self):
        return basic_model(self.env_train)

    def number_of_rewards(self):
        return len(self.rewards)

    def has_multiple_rewards(self):
        return self.number_of_rewards() > 1

    def replay_length(self):
        return self.replay.size

    def replay_rewards_scaling_matrix(self):
        rewards = self.replay['vector_of_rewards']
        covariance = np.cov(rewards.transpose(), ddof=1)
        if covariance.shape == ():
            covariance = covariance.reshape(1, 1)
        return linalg.inv(linalg.sqrtm(covariance))

    def get_augmented_experiences_sample(self, batch_size):
        experiences = deepcopy(self.replay.random_sample(batch_size, replace=True))
        experiences['weights'] = self.generate_random_weights(batch_size)
        experiences['gamma'] = self.generate_random_gammas(batch_size) if par.random_gamma \
            else np.tile(par.discount_factor_Q_learning, (batch_size, 1))
        return experiences

    def rescale_experiences_sample(self, experiences: dict):
        experiences['vector_of_rewards'] = np.dot(
            experiences['vector_of_rewards'],
            self.replay_rewards_scaling_matrix()) / np.linalg.norm(experiences['weights'], axis=-1).reshape(-1, 1)
        return experiences

    def one_step_of_training(self):
        """
        One step of training (update step for Bellman equation),
        as per Deep-Q learning algorithm with memory replay
        """

        experiences = self.rescale_experiences_sample(
            self.get_augmented_experiences_sample(par.batch_size_replay_sampling))

        current_qs_list = self.model([experiences['starting_observation'],
                                      experiences['starting_position'],
                                      experiences['weights'],
                                      experiences['gamma']]).numpy()

        future_qs_list = self.target_model([experiences['new_observation'],
                                            experiences['new_position'],
                                            experiences['weights'],
                                            experiences['gamma']]).numpy()

        reward = np.sum(experiences['vector_of_rewards'] * experiences['weights'], axis=-1)

        max_future_q = reward \
            + np.where(~experiences['done'].squeeze(),
                       experiences['gamma'].squeeze() * np.max(future_qs_list, axis=-1),
                       0)

        a = experiences['action'].squeeze()
        i = np.arange(len(a))
        current_qs_list[i, a] = (1 - par.learning_rate_Q_learning) * current_qs_list[i, a] \
            + par.learning_rate_Q_learning * max_future_q

        self.model.fit(
            [experiences['starting_observation'], experiences['starting_position'],
             experiences['weights'], experiences['gamma']],
            current_qs_list,
            batch_size=par.model.batch_size_for_learning,
            verbose=0,
            epochs=par.model.epochs_for_Q_Learning_fit,
            shuffle=True)

    def my_reshape(self, obs):
        obs = np.array(obs)
        obs = obs.reshape((self.env_train.window_size, 1))
        return obs

    def get_reward_index(self, reward: str):
        return self.rewards.index(reward)

    def generate_random_weights(self, n):
        single_weights = np.random.rand(n, self.number_of_rewards())
        return single_weights / \
            (np.sum(single_weights, axis=-1).reshape(-1, 1))

    def generate_random_gammas(self, steps):
        return np.random.uniform(par.min_possible_gamma,
                                 par.max_possible_gamma, size=(steps, 1))

    def get_possible_actions(self):
        return [Actions.Hold.value, Actions.Buy.value] if par.long_positions_only else \
               [Actions.Hold.value, Actions.Buy.value, Actions.Sell.value]

    def get_possible_positions(self):
        return [Positions.Neutral, Positions.Long] if par.long_positions_only else \
               [Positions.Neutral, Positions.Long, Positions.Short]

    def sample_random_action(self, n=None):
        return np.random.choice(self.get_possible_actions(), n)

    def get_specific_reward_weights(self, reward, steps):
        """Returns a weight vector where specified reward is weighted 1, others 0"""
        reward_weights = np.zeros((steps, self.number_of_rewards()))
        reward_weights[:, self.get_reward_index(reward)] = 1
        return reward_weights

    def run_agent(self, environment: EnvironmentRL, model,
                  reward=None, training=False, epsilon=0):
        """
        Function taking currently available NN model, and computing a number of environment steps in an efficient, vectorised way
        Inputs:
        - environment
        - model: (current state -> q-values) model
        - training: if set to False, all steps in the full environment are carried. Otherwise, only those steps
            in between consecutive fitting of the model. More precisely, training = True used during training,
            training = False used during plotting
        - reward: the reward the agent is going to maximize, to be set only when training=False
        - epsilon: when training = True, epsilon is used in the RL greedy search
        """

        assert not (
            training and reward), 'you cannot specify a specific reward during training'
        assert training or reward, 'you have to specify wether it is in training mode or a specific reward'

        if not training:
            environment.reset()
            n = environment.get_last_tick() - environment.get_start_tick()
        else:
            n = par.frequency_q_learning

        start = environment.get_current_tick() - par.window_size + 1
        end = min(start + n + par.window_size - 1, environment._end_tick)

        price_signals = lookback_array(environment.signal_features[0][start:end].reshape(
            -1), par.window_size).reshape((-1, par.window_size, 1))
        steps = len(price_signals)

        gamma = self.generate_random_gammas(steps) if training and par.random_gamma else np.tile(
            par.discount_factor_Q_learning, (steps, 1))
        reward_weights = self.generate_random_weights(
            steps) if training else self.get_specific_reward_weights(reward, steps)

        get_best_action = partial(np.argmax, axis=-1)
        best_actions = {p: get_best_action(model([price_signals, np.tile(p.value, (steps, 1)), reward_weights, gamma]))
                        for p in self.get_possible_positions()}

        done = False
        i = 0
        while i < n and (not done):

            action = self.sample_random_action() if sample_binomial(
                epsilon) else best_actions[environment.get_current_position()][i]

            new_observation, done, old_position, rewards = environment.step(
                action)

            if training:
                experience = {
                    'starting_observation': price_signals[i],
                    'new_observation': self.my_reshape(new_observation),
                    'starting_position': old_position.value,
                    'new_position': environment.get_current_position().value,
                    'action': action,
                    'done': done,
                    'vector_of_rewards': dict_to_array(rewards)
                }
                self.replay.append(experience)

            i += 1

    def render_environment(
            self, environment, string, reward, model):
        self.run_agent(environment, model=model, reward=reward)
        environment.render_all(string, reward)

    def record_performance(self, episode, reward):
        for subset in ('train', 'eval'):
            env = self.get_env(subset)
            self.run_agent(env, reward=reward, model=self.target_model)
            self.info_dataframe[subset, reward,
                                'average_reward'][episode] = env.average_reward(reward)
            self.info_dataframe[subset, reward,
                                'performance'][episode] = self.performance.get(env)

    def record_average_position(self, episode, reward):
        for subset in ('train', 'eval'):
            env = self.get_env(subset)
            assert env.is_done(), 'environments should be done when calling this funcion'
            self.info_dataframe[subset, reward, 'average_long'][episode] = env.position_proportion(
                Positions.Long)
            self.info_dataframe[subset, reward, 'average_short'][episode] = env.position_proportion(
                Positions.Short)

    def training_summary_plot(self, reward):
        plots.training_summary_plot(self, reward)
        self.render_environment(
            self.env_episode, 'training', reward=reward, model=self.target_model)
        self.render_environment(
            self.env_eval, 'evaluation', reward=reward, model=self.target_model)

    def get_env(self, subset: str):
        if subset == 'train':
            return self.env_train
        if subset == 'eval':
            return self.env_eval
        if subset == 'test':
            return self.env_test
        raise ValueError(
            f"{subset} is not a valid subset, it must be train, eval or test")

    def store_best_model(self, reward):
        self.target_model.save(f'{self.path}/{reward}/best_model')

    def store_best_model_info(self, reward):
        file = os.path.join(self.path, f'{reward}/plots/best_model_info.csv')
        pd.DataFrame(self.best_model_info[reward]).to_csv(file)

    def plot_best_model_info(self, reward):
        plots.plot_best_model_info(self, reward)

    def update_best_model_info(self, reward):
        """
        Updates metrics on train, eval and test environments based on the current target model.
        This method has to be called when the current target model is the best.
        """
        info = {}
        info['episode'] = self.current_episode
        for subset in ('train', 'eval', 'test'):
            env = self.get_env(subset)
            self.run_agent(env, model=self.target_model, reward=reward)
            info[f'average_reward_{subset}'] = env.average_reward(reward)
            info[f'performance_{subset}'] = self.performance.get(env)
            info[f'performance_b&h_{subset}'] = self.performance.get_buy_and_hold(
                env)

            s = 'MULTI' if self.has_multiple_rewards() else 'SINGLE'
            env.profit_plot(file=f'{reward}/plots/best_model_{subset}',
                            title=f'{s}:\n profits best model,\n {subset} ({reward})')
            env.positions_plot(file=f'{reward}/plots/best_model_{subset}',
                               title=f'{s}:\n positions best model,\n {subset} ({reward})')

        self.best_model_info[reward].append(info)
        self.store_best_model_info(reward)
        self.plot_best_model_info(reward)

    def get_best_eval_model_metric(self, env):
        """
        Returns the value of the metric that is used to evaluate the best model on evaluation environment
        """
        assert env.is_done(), 'The environment should be done'
        return self.performance.get(env)

    def update_best_model(self, reward):
        """
        Evaluates the target model on the evaluation environment,
        if the performance improves the maximum achieved performance then:
            - maximum achieved performance is updated
            - the target model is stored
            - plot and infos about best models performances on train, eval and test are updated
            - profit and positions plots are updated according to the new best model
        """

        self.run_agent(self.env_eval, model=self.target_model, reward=reward)

        current_performance_target_model = self.get_best_eval_model_metric(
            self.env_eval)
        if current_performance_target_model > self.max_performance_eval_target_model[reward]:
            print(f"new best model found for {reward}")
            self.max_performance_eval_target_model[reward] = current_performance_target_model
            self.store_best_model(reward)
            self.update_best_model_info(reward)

    def update_best_model_for_each_reward(self):
        for r in self.rewards:
            self.update_best_model(r)

    def use_mini_episodes(self):
        return isinstance(par.episode_length, int)

    def update_target_network_weights(self):
        """ equalize target and main network weights """
        print('Copying main network weights to target network weights')
        self.target_model.set_weights(self.model.get_weights())

    def build_episode(self, episode: int):
        df = sample_episode(par, self.env_train.df,
                            episode) if self.use_mini_episodes() else self.env_train.df
        print(
            f"EPISODE {episode}: from {df.index[0]} to {df.index[-1]} ({len(df)} steps)")
        return EnvironmentRL(df, window_size=par.window_size,
                             path_plots=self.path, log_returns_scale=self.fitted_log_returns_scale)

    def learn(self):
        self.update_target_network_weights()
        eps = par.max_eps

        for episode in tqdm.tqdm(range(par.number_iterations)):
            self.current_episode = episode
            self.env_episode = self.build_episode(episode)
            episode_start_time = time.time()
            steps_to_update_target_model = 0

            while not self.env_episode.is_done():
                self.run_agent(
                    self.env_episode, model=self.model, training=True, epsilon=eps)
                if self.replay_length() > par.batch_size_replay_sampling:
                    self.one_step_of_training()

                steps_to_update_target_model += 1

                if self.env_episode.is_done() or steps_to_update_target_model == par.frequency_target_exchange:
                    self.update_target_network_weights()
                    self.update_best_model_for_each_reward()
                    steps_to_update_target_model = 0

            print("Average training reward:"
                  "{:.2g}, eps: {:.4f}, episode {} with time: {:.4f}".format(
                      self.env_episode.average_reward(), eps, episode,
                      time.time() - episode_start_time))

            if episode > par.start_greedy_shift:
                eps = self.decayed_eps(episode)

            self.log_and_plots(episode, episode_start_time)

    def decayed_eps(self, episode):
        return par.min_eps + (par.max_eps - par.min_eps) * \
            np.exp(- par.decay * (episode - par.start_greedy_shift))

    def log_and_plots(self, episode, episode_start_time):
        for rew in self.rewards:

            plotting_start_time = time.time()

            if ((episode % par.plot_frequency == 0) or (
                    episode == (par.number_iterations - 1))):
                self.record_performance(episode, rew)
                self.record_average_position(episode, rew)
                self.training_summary_plot(rew)
                self.model.save(f'{self.path}/model')
            print(
                "Time of plots/computing eval is: {}".format(time.time() - plotting_start_time))

        print("Total time for episode {}: {}".format(
            episode, time.time() - episode_start_time))
        print("\n")
