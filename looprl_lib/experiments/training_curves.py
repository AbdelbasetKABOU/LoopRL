"""
Generating the training curves
"""

from typing import Literal

import matplotlib as mpl  # type: ignore

mpl.use('Agg')
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from looprl_lib.training.agent import solver, teacher
from looprl_lib.training.dashboard import stats_table
from looprl_lib.training.session import read_params, set_cur_session_dir


def setup_matplotlib() -> None:
    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    SMALL_SIZE = 7
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the figure title


def rewards_data(
    session_dirs: list[str],
    agent_name: Literal['solver', 'teacher'],
):
    data: list[pd.Series] = []
    for session_dir in session_dirs:
        set_cur_session_dir(session_dir)
        ps = read_params()
        assert ps is not None
        agent = teacher(ps) if agent_name == 'teacher' else solver(ps)
        stats = stats_table(agent)
        data.append(stats['success'])
    table = pd.concat(data, axis=1)
    mean = table.mean(axis=1)
    std = table.std(axis=1)
    return mean, std


def generate_training_curves(
    session_dirs: list[str],
    agent_name: Literal['solver', 'teacher'],
    output: str
):
    mean, std = rewards_data(session_dirs, agent_name)
    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    xs = np.array(range(mean.shape[0]))
    if len(session_dirs) > 1:
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
    ax.grid(linewidth=0.5)
    ax.plot(xs, mean, linewidth=1.0)
    ax.set(
        #title=(agent_name.title() + " Training Curve"),
        ylim=(-1, 1),
        xlabel="Iteration number",
        ylabel="Average reward")
    fig.savefig(output + ".pdf", bbox_inches="tight")
    #fig.savefig(output + ".png", dpi=300)


if __name__ == '__main__':
    setup_matplotlib()
    sessions = ["../sessions/final", "../sessions/final2"]
    generate_training_curves(
        sessions, "teacher", "out/teacher_training")
    generate_training_curves(
        sessions, "solver", "out/solver_training")
