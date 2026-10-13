"""
Running all experiments for the NeurIPS 2022 paper.
"""

import os

import ansi.colour.fg as fg  # type: ignore
from looprl_lib.experiments.search_budget import \
    generate_and_print_no_search_table
from looprl_lib.experiments.training_curves import (generate_training_curves,
                                                    setup_matplotlib)
from looprl_lib.params import STD_PARAMS, ParamsDiff
from looprl_lib.training.loop import run_session

RUNS_DIR = "runs"
OUT_DIR = "out"
TRAINED_TEACHER_DIR = "full"
UNTRAINED_TEACHER_DIR = "no-teacher"


def header(s: str):
    pad = "\n" * 3
    line = "-" * 80
    print(pad + fg.boldred(f"{line}\n{s}\n{line}\n") + pad)


def run_all_experiments(dir: str, num_runs: int, params: ParamsDiff):
    # Run training sessions
    for i in range(num_runs):
        run_dir = os.path.join(dir, RUNS_DIR, str(i))
        header(f"Run {i}: Full training session")
        run_session(os.path.join(run_dir, TRAINED_TEACHER_DIR), params)
        header(f"Run {i}: Untrained teacher")
        run_session(
            os.path.join(run_dir, UNTRAINED_TEACHER_DIR),
            {**params, 'teacher.agent.num_iters': 0})
    # Generate training figures
    outdir = os.path.join(dir, OUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    header(f"Generating training curves")
    setup_matplotlib()
    sessions = [
        os.path.join(dir, RUNS_DIR, str(i), TRAINED_TEACHER_DIR)
        for i in range(num_runs)]
    generate_training_curves(
        sessions, "teacher", os.path.join(outdir, "teacher_training"))
    generate_training_curves(
        sessions, "solver", os.path.join(outdir, "solver_training"))
    # Generate Code2inv table
    header(f"Generating the Code2inv table")
    generate_and_print_no_search_table(
        outdir=outdir,
        sessions=[(
            os.path.join(dir, RUNS_DIR, str(i), TRAINED_TEACHER_DIR),
            os.path.join(dir, RUNS_DIR, str(i), UNTRAINED_TEACHER_DIR))
            for i in range(num_runs)])


if __name__ == '__main__':
    params: ParamsDiff = {}
    # params = STD_PARAMS['toy'].copy()
    run_all_experiments(dir="neurips", num_runs=2, params=params)
