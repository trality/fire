""" This module includes all plot methods used in the project """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.environment.actions_and_positions import Positions
from src.util import save_pickle
from src.parameters import Parameters as par


# Set font for plots
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 7,
    "axes.titlesize": 9,
    "figure.titlesize": 9,
    "legend.fontsize": 6,
    "lines.linewidth": 0.6,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}",
    "axes.labelpad": 0.0,
    "legend.labelspacing": 0.1
})


def _plot_position(obj, position, tick):
    color = None
    if position == Positions.Short:
        color = 'red'
    elif position == Positions.Long:
        color = 'green'
    elif position == Positions.Neutral:
        color = 'grey'
    if color:
        plt.scatter(tick, obj.prices[tick], color=color)


def render(obj):
    if obj._first_rendering:
        obj._first_rendering = False
        plt.cla()
        plt.plot(obj.prices)
        start_position = obj._position_history[obj._start_tick]
        _plot_position(obj, start_position, obj._start_tick)

    _plot_position(obj, obj._position, obj._current_tick)

    plt.suptitle(
        "Total Reward: %.6f" % obj.total_reward + ' ~ '
        + "Total Profit: %.6f" % obj._total_profit
    )

    plt.pause(0.01)
    plt.savefig("LatestFigure", dpi=par.dpi_res)


def render_all(obj, subset, reward):
    plt.figure()
    window_ticks = np.arange(len(obj._position_history))
    plt.plot(obj.prices, alpha=0.5)

    short_ticks = []
    long_ticks = []
    neutral_ticks = []
    for i, tick in enumerate(window_ticks):
        if obj._position_history[i] == Positions.Short:
            short_ticks.append(tick - 1)
        elif obj._position_history[i] == Positions.Long:
            long_ticks.append(tick - 1)
        elif obj._position_history[i] == Positions.Neutral:
            neutral_ticks.append(tick - 1)

    plt.plot(neutral_ticks, obj.prices[neutral_ticks],
             'o', color='grey', ms=3, alpha=0.1)
    plt.plot(short_ticks, obj.prices[short_ticks],
             'o', color='r', ms=3, alpha=0.8)
    plt.plot(long_ticks, obj.prices[long_ticks],
             'o', color='g', ms=3, alpha=0.8)

    plt.suptitle(
        "Generalising Across Rewards: %.6s" % (
            len(par.rewards) > 1) + ' ~ '
        + "Iterations: %i" % par.number_iterations + ' ~ '
        + "dataset: %.20s" % par.dataset.name
    )
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    if subset == 'training':
        plot_file = "{}/{}/plots/training".format(obj.path_plots, reward)
    else:
        plot_file = "{}/{}/plots/evaluation".format(obj.path_plots, reward)

    plt.savefig(plot_file, dpi=par.dpi_res)
    d = {'neutral_ticks': neutral_ticks, 'short_ticks': short_ticks,
         'long_ticks': long_ticks, 'obj.prices': obj.prices}
    save_pickle(f'{plot_file}.pkl', d)

    # plot profits
    plot_current_profits = False
    if (len(obj._profits) > 0) and plot_current_profits:

        buy_and_hold_profits = obj.prices[obj._start_tick:]
        buy_and_hold_t = np.arange(len(buy_and_hold_profits)) + obj._start_tick

        plt.figure()
        plt.plot(buy_and_hold_t, buy_and_hold_profits, label="BH", alpha=0.9)
        plt.step(*zip(*obj._scaled_profits), where='post',
                 label="RL", alpha=0.9, color='C1')
        plt.legend()

        plot_file = "{}/{}/plots/{}_AllProfits".format(
            obj.path, reward, subset)
        plt.savefig(plot_file, dpi=par.dpi_res)
        d = {'buy_and_hold_t': buy_and_hold_t, 'buy_and_hold_profits': buy_and_hold_profits,
             'obj._scaled_profits': obj._scaled_profits}
        save_pickle(f'{plot_file}.pkl', d)
    plt.close('all')


def positions_plot(obj, file: str, title: str):
    plt.figure()
    plt.tight_layout()
    plt.plot(obj.prices, alpha=0.5)
    positions = np.array(obj._position_history)[1:]
    t = np.arange(len(positions))
    short_ticks = t[positions == Positions.Short]
    long_ticks = t[positions == Positions.Long]
    neutral_ticks = t[positions == Positions.Neutral]

    plt.plot(neutral_ticks, obj.prices[neutral_ticks],
             'o', color='grey', ms=2, alpha=0.1)
    plt.plot(short_ticks, obj.prices[short_ticks],
             'o', color='r', ms=2, alpha=0.8)
    plt.plot(long_ticks, obj.prices[long_ticks],
             'o', color='g', ms=2, alpha=0.8)
    plt.title(title, wrap=True)
    plot_file = f"{obj.path_plots}/{file}_prediction"

    plt.savefig(plot_file, dpi=par.dpi_res)
    d = {'neutral_ticks': neutral_ticks, 'short_ticks': short_ticks,
         'long_ticks': long_ticks, 'obj.prices': obj.prices}
    save_pickle(f'{plot_file}.pkl', d)

    plt.close('all')


def profit_plot(obj, file: str, title: str):
    assert len(obj._profits) > 0, "No profits to plot"

    buy_and_hold_profits = obj.prices[obj._start_tick:]
    buy_and_hold_t = np.arange(len(buy_and_hold_profits)) + obj._start_tick

    fig, ax = plt.subplots()
    fig.set_size_inches(1.6687, 1.6687)
    ax.plot(buy_and_hold_t, buy_and_hold_profits,
            label="BH", alpha=0.9, linestyle='dotted')
    ax.step(*zip(*obj._scaled_profits), where='post',
            label="RL", alpha=0.9, color='C1')
    ax.legend(loc="upper left")
    plt.title(title, wrap=True)
    ax.yaxis.set_major_formatter('\\${x:1.0f}')
    plot_file = f"{obj.path_plots}/{file}_profits"
    plt.savefig(plot_file, dpi=par.dpi_res,
                bbox_inches='tight', pad_inches=+0.02)

    d = {'buy_and_hold_t': buy_and_hold_t,
         'buy_and_hold_profits': buy_and_hold_profits,
         'obj._scaled_profits': obj._scaled_profits}
    save_pickle(f'{plot_file}.pkl', d)

    plt.close("all")


def training_summary_plot(obj, reward):

    d = {
        'average_reward_train': obj.info_dataframe['train', reward, 'average_reward'].dropna(),
        'average_reward_eval': obj.info_dataframe['eval', reward, 'average_reward'].dropna(),
        'performance_train': obj.info_dataframe['train', reward, 'performance'].dropna(),
        'performance_eval': obj.info_dataframe['eval', reward, 'performance'].dropna(),

        'performance_hline': obj.performance.hline,
        'performance_name': obj.performance.name,
        'train_buy_and_hold_performance': obj.train_buy_and_hold_performance,
        'eval_buy_and_hold_performance': obj.eval_buy_and_hold_performance,

        'average_long_train': obj.info_dataframe['train', reward, 'average_long'].dropna(),
        'average_long_eval': obj.info_dataframe['eval', reward, 'average_long'].dropna(),
        'average_short_train': obj.info_dataframe['train', reward, 'average_short'].dropna(),
        'average_short_eval': obj.info_dataframe['eval', reward, 'average_short'].dropna()
    }

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(2, 4.5))
    plt.subplots_adjust(hspace=0.42)

    # Rewards
    ax1.axhline(y=0.0, color='k', linestyle='dotted', alpha=0.5)

    ax1.step(d['average_reward_eval'].index, d['average_reward_eval'], label="eval", color=par.color_eval,
             linestyle=par.style_eval, marker=par.marker_eval, where='post')
    ax1.step(d['average_reward_train'].index, d['average_reward_train'], label="train", color=par.color_train,
             linestyle=par.style_train, marker=par.marker_train, where='post')
    ax1.legend(loc='best')
    ax1.ticklabel_format(axis='y', scilimits=[-2, 2])
    s = 'MULTI' if obj.has_multiple_rewards() else 'SINGLE'
    ax1.set_title(f'{s}: average {reward}', wrap=True)
    ax1.yaxis.get_offset_text().set_x(-0.12)
    ax1.yaxis.get_offset_text().set_fontsize(6)

    # Performances
    ax2.axhline(y=obj.performance.hline, color='k',
                linestyle='dotted', alpha=0.5)

    ax2.axhline(y=obj.eval_buy_and_hold_performance, color=par.color_eval, label='eval BH',
                linestyle='--', marker=par.marker_eval_BH)
    ax2.ticklabel_format(axis='y', scilimits=[-2, 2])
    ax2.step(d['performance_train'].index, d['performance_train'], label="train", color=par.color_train,
             linestyle=par.style_train, marker=par.marker_train, where='post')
    ax2.step(d['performance_eval'].index, d['performance_eval'], label="eval", color=par.color_eval,
             linestyle=par.style_eval, marker=par.marker_eval, where='post')

    ax2.legend(loc='best')
    ax2.set_title(obj.performance.name, wrap=True)
    ax2.yaxis.get_offset_text().set_x(-0.12)
    ax2.yaxis.get_offset_text().set_fontsize(6)

    # Long proportion
    ax3.set_ylim([-0.04, 1.04])
    ax3.axhline(y=1, color='k', linestyle='dotted', alpha=0.5)
    ax3.axhline(y=0, color='k', linestyle='dotted', alpha=0.5)
    ax3.step(d['average_long_train'].index, d['average_long_train'], label="train AL", color=par.color_train,
             linestyle=par.style_train, marker=par.marker_train, where='post')
    ax3.step(d['average_long_eval'].index, d['average_long_eval'], label="eval AL", color=par.color_eval,
             linestyle=par.style_eval, marker=par.marker_eval, where='post')
    ax3.set_title(
        "Average Long (AL)" if par.long_positions_only else "Average Long/Short (AL/AS)")
    if not par.long_positions_only:
        ax3.step(d['average_short_train'].index, d['average_short_train'], label="train AS", color=par.color_train,
                 linestyle=par.style_train_neutral, marker=par.marker_train_neutral, markersize=1, where='post')
        ax3.step(d['average_short_eval'].index, d['average_short_eval'], label="eval AS", color=par.color_eval,
                 linestyle=par.style_eval_neutral, marker=par.marker_eval_neutral, markersize=1, where='post')
    ax3.legend(loc='best')
    ax3.ticklabel_format(axis='y', scilimits=[-2, 2])

    plt.xlabel("episodes")

    plot_file = "{}/{}/plots/training_rewards_target_network".format(
        obj.path, reward)

    plt.savefig(plot_file, pad_inches=+0.02,
                dpi=par.dpi_res, bbox_inches='tight')

    save_pickle(f'{plot_file}.pkl', d)

    plt.close('all')


def plot_best_model_info(obj, reward):
    """
    Does plots based on best model info
    """
    df = pd.DataFrame(obj.best_model_info[reward])
    df = df.set_index('episode')
    colors = [par.color_train, par.color_eval, par.color_test]
    linestyles = [par.style_train, par.style_eval, par.style_test]
    linestyles_BH = [par.style_train_BH,
                     par.style_eval_BH, par.style_test_BH]
    markers = [par.marker_train, par.marker_eval, par.marker_test]
    labels = ['train BH', 'eval BH', 'test BH']
    fig, ax = plt.subplots()
    d = {}
    for i, subset in enumerate(['train', 'eval', 'test']):
        c = colors[i]
        plt.plot(df[f'performance_{subset}'], label=subset, alpha=0.9, color=c,
                 linestyle=linestyles[i], marker=markers[i])
        plt.axhline(obj.performance.get_buy_and_hold(obj.get_env(subset)),
                    linestyle=linestyles_BH[i], color=c, marker=markers[i], label=labels[i])

        d[f'performance_{subset}'] = df[f'performance_{subset}']
        d[f'bh_{subset}'] = obj.performance.get_buy_and_hold(
            obj.get_env(subset))

    s = 'MULTI' if obj.has_multiple_rewards() else 'SINGLE'
    plt.title(f"{s}:\n {obj.performance.name} ({reward})", wrap=True)
    plt.legend(loc='best')
    plt.xlabel('episodes')
    fig.set_size_inches(1.6687, 1.6687)
    plot_file = f"{obj.path}/{reward}/plots/best_model"
    plt.savefig(plot_file, bbox_inches='tight',
                pad_inches=+0.02, dpi=par.dpi_res)
    save_pickle(f'{plot_file}.pkl', d)
