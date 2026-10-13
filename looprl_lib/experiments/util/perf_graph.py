from typing import Optional, Sequence

import matplotlib.pyplot as plt  # type: ignore
import numpy as np


def dummy_data(n, mean, stdev, lb):
    data = np.round(stdev * np.random.randn(n) + mean).astype(int)
    return np.clip(data, lb, None).tolist()


def plot_perf_curves(
    data: dict[str, Sequence[Optional[int]]],
    min_steps: int,
    max_steps: int,
    title: str = "",
    log_scale: bool = True,
    switch_axes: bool = False,
    min_problems: int = 0,
):
    n = max(len(points) for points in data.values())
    fig, ax = plt.subplots()
    for name, points in data.items():
        xs = list(range(max_steps + 1))
        ys = [sum(1 if p is not None and p <= x else 0 for p in points)
              for x in xs]
        if switch_axes:
            xs, ys = ys, xs
        ax.plot(xs, ys, label=name)
    xlabel = 'Search budget'
    ylabel = 'Number of solved problems'
    if log_scale:
        min_steps = max(1, min_steps)
    xlim = (min_steps, None)
    ylim = (min_problems, n)
    if not switch_axes:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        ax.legend(loc='lower right')
        if log_scale: ax.set_xscale('log')
    else:
        ax.set(title=title, xlabel=ylabel, ylabel=xlabel, xlim=ylim, ylim=xlim)
        ax.legend(loc='upper left')
        if log_scale: ax.set_yscale('log')
    ax.grid(which='both', color='0.85')
    return fig


if __name__ == '__main__':
    n = 2000
    baseline_data = dummy_data(n, 100, 70, 20)
    trained_data = dummy_data(n, 30, 20, 20)
    fig = plot_perf_curves(
        {'baseline': baseline_data, 'trained': trained_data},
        min_steps=1,
        max_steps=200)
    fig.savefig("test_curve.png", dpi=300)
