#!/usr/bin/env python3

from collections import OrderedDict
from itertools import combinations
import logging
import os
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from elora import Elora


# new tableau colors
# https://www.tableau.com/about/blog/2016/7/colors-upgrade-tableau-10-56782
colors = OrderedDict([
    ('blue', '#4e79a7'),
    ('orange', '#f28e2b'),
    ('green', '#59a14f'),
    ('red', '#e15759'),
    ('cyan', '#76b7b2'),
    ('purple', '#b07aa1'),
    ('brown', '#9c755f'),
    ('yellow', '#edc948'),
    ('pink', '#ff9da7'),
    ('gray', '#bab0ac')
])

default_color = '#404040'
font_size = 18

plt.rcdefaults()
plt.rcParams.update({
    'figure.figsize': (10, 6.18),
    'figure.dpi': 200,
    'figure.autolayout': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato'],
    'mathtext.fontset': 'custom',
    'mathtext.cal': 'sans',
    'font.size': font_size,
    'legend.fontsize': font_size,
    'axes.labelsize': font_size,
    'axes.titlesize': font_size,
    'axes.prop_cycle': plt.cycler('color', list(colors.values())),
    'xtick.labelsize': font_size - 2,
    'ytick.labelsize': font_size - 2,
    'lines.linewidth': 1.25,
    'lines.markeredgewidth': .1,
    'patch.linewidth': 1.25,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.facecolor': '#eaeaf2',
    'axes.linewidth': 0,
    'grid.linestyle': '-',
    'grid.linewidth': 1,
    'grid.color': '#fcfcfc',
    'savefig.facecolor': '#fcfcfc',
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'xtick.minor.size': 0,
    'ytick.minor.size': 0,
    'xtick.major.pad': 7,
    'ytick.major.pad': 7,
    'text.color': default_color,
    'axes.edgecolor': default_color,
    'axes.labelcolor': default_color,
    'xtick.color': default_color,
    'ytick.color': default_color,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.frameon': False,
    'image.interpolation': 'none',
})

plotdir = Path('_static')
plotdir.mkdir(exist_ok=True)

plot_functions = {}


def plot(f):
    """
    Plot function decorator.  Calls the function, does several generic tasks,
    and saves the figure as the function name.
    """
    def wrapper(*args, **kwargs):
        logging.info('generating plot: %s', f.__name__)
        f(*args, **kwargs)

        fig = plt.gcf()

        plotfile = plotdir / '{}.png'.format(f.__name__)
        fig.savefig(str(plotfile))
        logging.info('wrote %s', plotfile)
        plt.close(fig)

    plot_functions[f.__name__] = wrapper

    return wrapper


def figsize(relwidth=1, aspect=.618, refwidth=10):
    """
    Return figure dimensions from a relative width (to a reference width) and
    aspect ratio (default: 1/golden ratio).
    """
    width = relwidth * refwidth

    return width, width*aspect


def set_tight(fig=None, **kwargs):
    """
    Set tight_layout with a better default pad.
    """
    if fig is None:
        fig = plt.gcf()

    kwargs.setdefault('pad', .1)
    fig.set_tight_layout(kwargs)


class League:
    """
    Create a toy-model league of Poisson random variables.

    """
    locs = [100, 90, 94, 98, 102, 106, 110]
    loc1, *loc2_list = locs
    scale = 10

    def __init__(self, size=1000):

        self.times = np.arange(size).astype('datetime64[s]')
        locs1, locs2 = np.random.choice(self.locs, size=(2, size))

        self.spreads = norm.rvs(
            loc=(locs1 - locs2), scale=np.sqrt(2*self.scale**2), size=size)
        self.totals = norm.rvs(
            loc=(locs1 + locs2), scale=np.sqrt(2*self.scale**2), size=size)

        self.labels1 = locs1.astype(str)
        self.labels2 = locs2.astype(str)


league = League(10**6)


@plot
def quickstart_example(scale=1, size=100):
    """
    Create time series of comparison data by pairing and
    substracting 100 different Poisson distributions

    """
    loc_values = np.random.randint(80, 110, 100)
    locs1, locs2 = map(np.array, zip(*combinations(loc_values, 2)))
    labels1, labels2 = [loc.astype(str) for loc in [locs1, locs2]]
    spreads = norm.rvs(loc=(locs1 - locs2), scale=np.sqrt(2*scale**2))
    times = np.arange(spreads.size).astype('datetime64[s]')

    # train the model on the list of comparisons
    elora = Elora(times, labels1, labels2, spreads)
    elora.fit(.05, False)

    # predicted and true (analytic) comparison values
    pred_times = np.repeat(elora.last_update_time, times.size)
    pred = elora.mean(pred_times, labels1, labels2)
    true = norm.rvs(loc=(locs1 - locs2), scale=np.sqrt(2*scale**2))

    # plot predicted means versus true means
    plt.scatter(pred, true)
    plt.plot([-30, 30], [-30, 30], color='k')
    plt.xlabel('predicted mean')
    plt.ylabel('true mean')


@plot
def validate_spreads():
    """
    Prior spread predictions at every value of the line.

    """
    fig, (ax_prior, ax2_post) = plt.subplots(
        nrows=2, figsize=figsize(aspect=1.2))

    # train margin-dependent Elo model
    lines = np.arange(-49.5, 50.5)

    elora = Elora(league.times, league.labels1, league.labels2, league.spreads)
    elora.fit(1e-4, False, scale=np.sqrt(2*league.scale**2))

    # exact prior distribution
    sf = norm.sf(lines, loc=elora.median_value, scale=np.sqrt(2*10**2))
    ax_prior.plot(lines, sf, color='k')

    # label names
    label1 = str(league.loc1)
    label2_list = [str(loc2) for loc2 in league.loc2_list]

    plot_args = [
        (ax_prior, elora.first_update_time, 'prior'),
        (ax2_post, elora.last_update_time, 'posterior'),
    ]

    for ax, time, title in plot_args:
        for n, label2 in enumerate(label2_list):

            sf = elora.sf(lines, time, label1, label2)
            label = r'$\mu_2={}$'.format(label2)

            if ax.is_first_row():
                ax.plot(lines[n::6], sf[n::6], 'o', zorder=2, label=label)

            if ax.is_last_row():
                ax.plot(lines, sf, 'o', zorder=2, label=label)

                sf = norm.sf(lines, loc=int(label1) - int(label2),
                             scale=np.sqrt(2*league.scale**2))
                ax.plot(lines, sf, color='k')

            leg = ax.legend(title=r'$\mu_1 = {}$'.format(label1),
                            handletextpad=.2, loc=1)
            leg._legend_box.align = 'right'

            lines = np.floor(lines)
            ax.set_xticks(lines[::10])
            ax.set_xlim(lines.min(), lines.max())

            if ax.is_last_row():
                ax.set_xlabel('line $=$ scored $-$ allowed')

            ax.set_ylabel('probability to cover line')

            ax.annotate(title, xy=(.05, .05),
                        xycoords='axes fraction', fontsize=24)

    set_tight(h_pad=1)


@plot
def validate_totals():
    """
    Prior spread predictions at every value of the line.

    """
    fig, (ax_prior, ax2_post) = plt.subplots(
        nrows=2, figsize=figsize(aspect=1.2))

    # train margin-dependent Elo model
    lines = np.arange(149.5, 250.5)
    elora = Elora(league.times, league.labels1, league.labels2, league.totals)
    elora.fit(1e-4, True, scale=np.sqrt(2*league.scale**2))

    # exact prior distribution
    sf = norm.sf(lines, loc=elora.median_value, scale=np.sqrt(2*10**2))
    ax_prior.plot(lines, sf, color='k')

    # label names
    label1 = str(league.loc1)
    label2_list = [str(loc2) for loc2 in league.loc2_list]

    plot_args = [
        (ax_prior, elora.first_update_time, 'prior'),
        (ax2_post, elora.last_update_time, 'posterior'),
    ]

    for ax, time, title in plot_args:
        for n, label2 in enumerate(label2_list):

            sf = elora.sf(lines, time, label1, label2)
            label = r'$\mu_2={}$'.format(label2)

            if ax.is_first_row():
                ax.plot(lines[n::6], sf[n::6], 'o', zorder=2, label=label)

            if ax.is_last_row():
                ax.plot(lines, sf, 'o', zorder=2, label=label)

                sf = norm.sf(lines, loc=int(label1) + int(label2),
                             scale=np.sqrt(2*league.scale**2))
                ax.plot(lines, sf, color='k')

            leg = ax.legend(title=r'$\mu_1 = {}$'.format(label1),
                            handletextpad=.2, loc=1)
            leg._legend_box.align = 'right'

            lines = np.floor(lines)
            ax.set_xticks(lines[::10])
            ax.set_xlim(lines.min(), lines.max())

            if ax.is_last_row():
                ax.set_xlabel('line $=$ scored $+$ allowed')

            ax.set_ylabel('probability to cover line')

            ax.annotate(title, xy=(.05, .05),
                        xycoords='axes fraction', fontsize=24)

    set_tight(h_pad=1)


@plot
def convergence():
    """
    Test rating convergence at single value of the line.

    """
    fig, axes = plt.subplots(nrows=2, figsize=figsize(aspect=1.2))

    # label names
    label1 = str(league.loc1)
    label2_list = [str(loc2) for loc2 in league.loc2_list]

    # point spread and point total subplots
    subplots = [
        (False, 0, league.spreads, 'probability spread > 0'),
        (True, 200, league.totals, 'probability total > 200'),
    ]

    for ax, (commutes, line, values, ylabel) in zip(axes, subplots):

        # train margin-dependent Elo model
        elora = Elora(league.times, league.labels1, league.labels2, values)
        elora.fit(1e-4, commutes, scale=np.sqrt(2*league.scale**2))

        for label2 in label2_list:

            # evaluation times and labels
            times = np.arange(league.times.size)[::1000]
            labels1 = times.size * [label1]
            labels2 = times.size * [label2]

            # observed win probability
            prob = elora.sf(line, times, labels1, labels2)
            ax.plot(times, prob)

            # true (analytic) win probability
            if ax.is_first_row():
                prob = norm.sf(
                    line, loc=int(label1) - int(label2),
                    scale=np.sqrt(2*league.scale**2))
                ax.axhline(prob, color='k')
            else:
                prob = norm.sf(
                    line, loc=int(label1) + int(label2),
                    scale=np.sqrt(2*league.scale**2))
                ax.axhline(prob, color='k')

        # axes labels
        if ax.is_last_row():
            ax.set_xlabel('Iterations')
        ax.set_ylabel(ylabel)

    set_tight(w_pad=.5)


def main():
    import argparse

    logging.basicConfig(
            format='[%(levelname)s][%(module)s] %(message)s',
            level=os.getenv('LOGLEVEL', 'info').upper()
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('plots', nargs='*')
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        if args.plots:
            for i in args.plots:
                if i.endswith('.pdf'):
                    i = i[:-4]
                if i in plot_functions:
                    plot_functions[i]()
                else:
                    print('unknown plot:', i)
        else:
            for f in plot_functions.values():
                f()


if __name__ == "__main__":
    main()
