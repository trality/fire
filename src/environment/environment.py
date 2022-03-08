import numpy as np
import src.plots as plots
from src.environment.actions_and_positions import Actions, Positions


def mean_over_std(x):
    std = np.std(x, ddof=1)
    mean = np.mean(x)
    return mean / std if std > 0 else 0


class Environment():
    def __init__(self, df, window_size, path_plots, log_returns_scale,
                 trade_fee_bid=0.0, trade_fee_ask=0.0):

        self.df = df
        self.window_size = window_size
        self.log_returns_scale = log_returns_scale
        self.prices, self.signal_features = self._process_data()
        self.path_plots = path_plots
        self.trade_fee_bid = trade_fee_bid
        self.trade_fee_ask = trade_fee_ask

        self._start_tick = self.get_start_tick()
        self._end_tick = self.get_last_tick()

        self.reset()

    def get_prices(self):
        return self.df.loc[:, 'price'].to_numpy()

    def get_start_tick(self):
        return self.window_size - 1

    def get_last_tick(self):
        return len(self.prices) - 1

    def last_positions(self, number_of_positions):

        if self.history == {}:
            x = np.zeros(int(number_of_positions), dtype=int)
        else:
            len_hist = len(self.history["position"])
            if len_hist > number_of_positions - 1:
                x = np.array(
                    self.history["position"][len_hist - number_of_positions:len_hist],
                    dtype=int)
            else:
                x = np.zeros(int(number_of_positions) - len_hist, dtype=int)
                x = np.append(x, self.history["position"])
        return x

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Neutral
        self._position_history = (self.window_size * [None]) + [self._position]
        self._initialize_total_reward()
        self._total_profit = 1.
        self._first_rendering = True
        self._profits = [(self._start_tick, 1)]
        self._scaled_profits = [
            (self._start_tick, self.prices[self._start_tick])]
        self.history = {}
        self.portfolio_log_returns = np.zeros(len(self.prices))
        return self._get_observation()

    def get_current_tick(self):
        return self._current_tick

    def step(self, action):
        self._done = False
        self._current_tick += 1
        old_position = self._position

        if self._current_tick == self._end_tick:
            self._done = True

        self.update_portfolio_log_returns(action)

        rewards = self._calculate_reward(action)
        self._update_total_reward(rewards)

        self._update_profit(action)

        if self.is_tradesignal(action):
            # Update position
            if action == Actions.Hold.value:
                self._position = Positions.Neutral
            elif action == Actions.Buy.value:
                self._position = Positions.Long
            elif action == Actions.Sell.value:
                self._position = Positions.Short
            # Update last trade tick
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward=self.total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        return observation, self._done, old_position, rewards

    def is_tradesignal(self, action):
        """
        Returns True if action does not correspond to correspond to current position.
        Buy \\hat{=} Long, Sell \\hat{=} Short, Hold \\hat{=} Neutral
        """
        return not ((action == Actions.Buy.value and self._position == Positions.Long)
                    or (action == Actions.Sell.value and self._position == Positions.Short)
                    or (action == Actions.Hold.value and self._position == Positions.Neutral))

    def most_recent_return(self, action):
        """
        We support Long, Neutral and Short positions.
        Return is generated from rising prices in Long
        and falling prices in Short positions.
        The actions Sell/Buy or Hold during a Long position trigger the sell/buy-fee.
        """
        # Long positions
        if self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            if action == Actions.Sell.value or action == Actions.Hold.value:
                current_price = self.add_sell_fee(current_price)

            previous_price = self.prices[self._current_tick - 1]
            if (self._position_history[self._current_tick - 1] == Positions.Short
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_buy_fee(previous_price)

            return np.log(current_price) - np.log(previous_price)

        # Short positions
        if self._position == Positions.Short:
            current_price = self.prices[self._current_tick]
            if action == Actions.Buy.value or action == Actions.Hold.value:
                current_price = self.add_buy_fee(current_price)

            previous_price = self.prices[self._current_tick - 1]
            if (self._position_history[self._current_tick - 1] == Positions.Long
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_sell_fee(previous_price)

            return np.log(previous_price) - np.log(current_price)

        return 0

    def update_portfolio_log_returns(self, action):
        self.portfolio_log_returns[self._current_tick] = self.most_recent_return(
            action)

    def is_done(self):
        return self._done

    def _get_observation(self):
        return self.signal_features[0][(
            self._current_tick - self.window_size) + 1:self._current_tick + 1]

    def revert_history_step(self):
        for key in self.history.keys():
            self.history[key].pop()

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self):
        plots.render(self)

    def render_all(self, string, reward):
        plots.render_all(self, string, reward)

    def positions_plot(self, file: str, title: str):
        plots.positions_plot(self, file, title)

    def profit_plot(self, file: str, title: str):
        plots.profit_plot(self, file, title)

    def _is_trade(self, action: Actions):
        return ((action == Actions.Buy.value and self._position == Positions.Short)
                or (action == Actions.Sell.value and self._position == Positions.Long)
                or (action == Actions.Hold.value and self._position == Positions.Long)
                or (action == Actions.Hold.value and self._position == Positions.Short))

    def _update_profit(self, action):
        if self._is_trade(action) or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = self._total_profit / \
                    self.add_buy_fee(last_trade_price)
                self._total_profit = shares * self.add_sell_fee(current_price)
                self._profits.append((self._current_tick, self._total_profit))
                self._scaled_profits.append(
                    (self._current_tick, self._total_profit * self.prices[self._start_tick]))

            if self._position == Positions.Short:
                shares = self._total_profit / self.add_buy_fee(current_price)
                self._total_profit = shares * \
                    self.add_sell_fee(last_trade_price)
                self._profits.append((self._current_tick, self._total_profit))
                self._scaled_profits.append(
                    (self._current_tick, self._total_profit * self.prices[self._start_tick]))

    def add_buy_fee(self, price):
        return price * (1 + self.trade_fee_bid)

    def add_sell_fee(self, price):
        return price / (1 + self.trade_fee_ask)

    def position_proportion(self, position: Positions):
        # This is in decimal notation e.g. 1 = 100%
        return np.mean(np.array(self._position_history[1:])[
                       self.window_size:] == position)

    def get_total_return(self, vector=False):
        return np.exp(self.get_total_log_return(vector=vector))

    def get_portfolio_log_returns(self):
        return self.portfolio_log_returns[1:self._current_tick + 1]

    def get_total_log_return(self, vector=False):
        f = np.cumsum if vector else np.sum
        return f(self.get_portfolio_log_returns())

    def get_sharpe_ratio(self):
        return mean_over_std(self.get_portfolio_log_returns())

    def get_total_log_return_buy_and_hold(self):
        return np.log(self.add_sell_fee(
            self.prices[-1]) / self.add_buy_fee(self.prices[self._start_tick]))

    def get_sharpe_ratio_buy_and_hold(self):
        prices_with_fees = np.copy(self.prices[self._start_tick:])
        prices_with_fees[0] = self.add_buy_fee(prices_with_fees[0])
        prices_with_fees[-1] = self.add_sell_fee(prices_with_fees[-1])
        log_returns = np.diff(np.log(prices_with_fees))
        return mean_over_std(log_returns)

    def get_current_position(self):
        return self._position
