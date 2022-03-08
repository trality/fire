import numpy as np
from src.environment.environment import Environment, Actions, Positions
from src.parameters import Parameters as par


def mean_over_std(x: np.ndarray):
    std = np.std(x, ddof=1)
    mean = np.mean(x)
    return mean / std if std > 0 else 0


def over_t_ratio(x: np.ndarray):
    sum_abs = np.sum(np.abs(x))
    sum = np.sum(x)
    return sum / sum_abs if sum_abs > 0 else 0


def alr_output(x: np.ndarray):
    return np.mean(x)


def sum_dictionaries(d1: dict, d2: dict):
    assert set(d1) == set(d2), f"Different keys: {set(d1)}!={set(d2)}"
    return {k: d1[k] + d2[k] for k in d1}


def dictionary_scalar_product(d: dict, c: float):
    return {k: v * c for k, v in d.items()}


def dictionary_average(d: dict):
    return sum(d.values()) / len(d)


class EnvironmentRL(Environment):

    def __init__(self, df, window_size: int,
                 path_plots: str, log_returns_scale: float):
        self.rewards = par.rewards
        self.rewards_dictionary = self.get_rewards_dictionary()
        super().__init__(df, window_size, path_plots, log_returns_scale,
                         trade_fee_bid=par.environment.trade_fee_bid,
                         trade_fee_ask=par.environment.trade_fee_ask)

    def get_rewards_dictionary(self):
        self.check_rewards()
        return {r_name: self.ALL_REWARDS[r_name] for r_name in self.rewards}

    def check_rewards(self):
        invalid_rewards = set(self.rewards) - set(self.ALL_REWARDS)
        if len(invalid_rewards) > 0:
            raise ValueError(f"{invalid_rewards} are not valid rewards")

    def POWC(self, action):
        # Extended to "Profit Only When (Position) Closed" (POWC)
        current_price = self.add_sell_fee(self.prices[self._current_tick])
        last_trade_price = self.add_buy_fee(self.prices[self._last_trade_tick])

        if (action == Actions.Sell.value or action == Actions.Hold.value) \
           and self._position == Positions.Long:
            return np.log(current_price) - np.log(last_trade_price)

        if (action == Actions.Buy.value or action == Actions.Hold.value) \
           and self._position == Positions.Short:
            return np.log(last_trade_price) - np.log(current_price)

        return 0.

    def LR(self, action):
        return self.get_portfolio_log_returns()[-1]

    def cautious_LR(self, action, penalty=0.2):
        """
        LR but negative returns are weighted more,
        in particular they are multiplied by <penalty>+1
        """
        pv = self.LR(action)
        return pv if pv > 0 else (1 + penalty) * pv

    def SR(self, action, lookback=10):
        """
        Sharpe ratio calculated on the last <lookback> steps
        """
        returns = self.get_portfolio_log_returns()[-lookback:]
        return mean_over_std(returns)

    def SR_diff(self, action, lookback=10):
        """
        Difference between the current sharpe ratio and the one at the previous time step.
        Both are calculated using the last <lookback> steps
        """
        returns = self.get_portfolio_log_returns()[-lookback - 1:]
        current_sr = mean_over_std(returns[1:])
        previous_sr = mean_over_std(returns[:-1])
        return current_sr - previous_sr

    def OVER_t(self, action, lookback=10):
        """
        OVER_t ratio calculated on the last <lookback> steps
        """
        returns = self.get_portfolio_log_returns()[-lookback:]
        over_t = over_t_ratio(returns)
        return over_t

    def ALR(self, action, lookback=10):
        """
        ALR calculated on the last <lookback> steps
        """
        returns = self.get_portfolio_log_returns()[-lookback:]
        alr = alr_output(returns)
        return alr

    ALL_REWARDS = {
        "POWC": POWC,
        "LR": LR,
        "SR": SR,
        "SR_diff": SR_diff,
        "cautious_LR": cautious_LR,
        "ALR": ALR,
    }

    def fit_log_returns_scale(self):
        prices = self.df.loc[:, 'price'].to_numpy()
        r = np.diff(np.log(prices))
        self.log_returns_scale = np.std(r, ddof=1)

    def get_log_returns_scale(self):
        if self.log_returns_scale == "fit":
            self.fit_log_returns_scale()
        return self.log_returns_scale

    def rescale_log_returns(self, x):
        return x / self.get_log_returns_scale()

    def _process_data(self):
        prices = self.get_prices()

        log_returns = np.insert(np.diff(np.log(prices)), 0, 0)
        rescaled_log_returns = self.rescale_log_returns(log_returns)
        signal_features = np.column_stack(rescaled_log_returns)

        return prices, signal_features

    def get_reward_vector(self, action):
        return np.array([r(self, action)
                        for r in self.rewards_dictionary.values()])

    def _calculate_reward(self, action):
        return {name: float(f(self, action))
                for name, f in self.rewards_dictionary.items()}

    def _initialize_total_reward(self):
        self.total_reward = {r: 0 for r in self.rewards}

    def _update_total_reward(self, rewards):
        self.total_reward = sum_dictionaries(self.total_reward, rewards)

    def average_reward(self, reward=None):
        """ If reward is None, then an average of reward averages is returned"""
        averages = dictionary_scalar_product(
            self.total_reward, 1 / (self._current_tick - self._start_tick))
        return averages[reward] if reward else dictionary_average(averages)
