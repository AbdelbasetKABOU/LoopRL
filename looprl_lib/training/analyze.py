"""
Utilities to inspect and tune a Looprl agent.
"""

import argparse
import os
from typing import Any

import sexpdata  # type: ignore
from ansi.colour.fg import yellow  # type: ignore
from looprl_lib.params import Params, ParamsDiff
from looprl_lib.training import dashboard
from looprl_lib.training.agent import teacher


def generate_teacher_samples(
    params_diff: ParamsDiff,
    net_file: str,
    target_dir: str,
    num_probs: int,
    gumbel_exploration: bool = True,
    save_failures: bool = True
) -> dict[str, float]:
    ps = Params.from_diff(params_diff)
    agent = teacher(ps)
    agent.log = lambda s: print(s, end="\n\n")
    return agent.gen_samples(
        net_file, target_dir, num_probs, gumbel_exploration, save_failures)


def mcts_grid_search(
    net_file: str,
    dir: str,
    num_probs: int = 2000,
    gumbel_exploration: bool = True
):
    res: dict[str, float] = {}
    for value_scale in [0.3, 0.1, 3.0]:
        for fpu_red in [0, 0.15]:
            for reset_tree in [False, True]:
                name = (
                    f"vscale={value_scale}__"+
                    f"fpu={fpu_red}__"+
                    f"reset={reset_tree}")
                print("\n", yellow(f"Running experiment {name}"), "\n")
                subdir = os.path.join(dir, name)
                diff = {
                    '::mcts.value_scale': value_scale,
                    '::mcts.fpu_red': fpu_red,
                    '::mcts.reset_tree': reset_tree }
                summary = generate_teacher_samples(
                    diff, net_file, subdir, num_probs, gumbel_exploration)
                res[name] = summary['rewards']
    pairs = list(res.items())
    pairs.sort(reverse=True, key=lambda x: x[1])
    print("\n", yellow(f"Results {name}"), "\n")
    for k, r in pairs:
        print(f"{k:60s} {r:.3f}")
    return res


def inspect_problems_dir(problems_dir: str) -> None:
    agent = teacher(Params())
    dashboard.viz_problems_in_dir(agent, problems_dir)


def sexp_to_nested_tuples(obj: Any):
    if isinstance(obj, list):
        return tuple(sexp_to_nested_tuples(x) for x in obj)
    elif isinstance(obj, sexpdata.Symbol):
        return obj.value()
    else: return obj


def sexp_to_set(sexp: str):
    return set(sexp_to_nested_tuples(sexpdata.parse(sexp)[0]))


def spec_matches(spec_pat: str, spec):
    pat_sexp = sexp_to_set(spec_pat)
    spec_sexp = sexp_to_set(spec)
    return pat_sexp.issubset(spec_sexp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='looprl-inspect',
        description='Looprl Inspection Utilities.')
    subparsers = parser.add_subparsers(help='', dest='command')
    problem = subparsers.add_parser("viz-problems")
    problem.add_argument("dir", type=str)
    args = parser.parse_args()
    if args.command == 'viz-problems':
        inspect_problems_dir(args.dir)
