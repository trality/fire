from collections import namedtuple

Performance = namedtuple(
    'Performance', ['name', 'get', 'get_buy_and_hold', 'hline'])

sharpe_ratio_performance = Performance(
    name='Sharpe ratio',
    get=lambda environment: environment.get_sharpe_ratio(),
    get_buy_and_hold=lambda environment: environment.get_sharpe_ratio_buy_and_hold(),
    hline=0)
