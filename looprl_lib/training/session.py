"""
Session management utilities

# Directory structure:
- session
  - log.txt, stage.txt, params.json, params_diff.json, time.json
  - solver_problems.txt
  - solver_problems
  - teacher, solver
      - pre, 0, 1, 2...
          - train_data, validation_data:
              - stats.json, problems, samples
          - training
              - net.pt, epochs.json, steps.json
"""

import os
import shutil
import timeit
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

import ansi.colour.fg as fg  # type: ignore
import ansi.colour.fx as fx  # type: ignore
from looprl_lib.params import Params
from looprl_lib.net_util import NETWORK_FILE

# High level agent dirs and files
TEACHER_DIR = "teacher"
SOLVER_DIR = "solver"
LOG_FILE = "log.txt"
PARAMS_FILE = "params.json"
PARAMS_DIFF_FILE = "params_diff.json"
TIME_FILE = "time.json"
STAGE_FILE = "stage.txt"
SOLVER_PROBLEMS_FILE = "problems.txt"
SOLVER_PROBLEMS_DIR = "problems"


# Inside an agent dir (add iteration directories)
PRETRAINING_DIR = "pre"

# Inside an iteration dir
TRAINING_DIR = "training"
TRAIN_DATA_DIR = "data_train"
VALIDATION_DATA_DIR = "data_valid"

# Inside data dir
SAMPLES_DIR = "samples"
PROBLEMS_DIR = "problems"
STATS_FILE = "stats.json"

# Inside training dir
# Files defined in net_util.py


_cur_session_dir: Optional[str] = None


def set_cur_session_dir(d: str) -> None:
    global _cur_session_dir
    _cur_session_dir = d


def cur_session_dir() -> str:
    msg = "Please call set_cur_session_dir"
    assert _cur_session_dir is not None, msg
    os.makedirs(_cur_session_dir, exist_ok=True)
    return _cur_session_dir


def write_params(params: Params) -> None:
    file = os.path.join(cur_session_dir(), PARAMS_FILE)
    with open(file, "w") as f:
        json = params.to_json(indent=4)  #type: ignore
        assert params == Params.from_json(json)  #type: ignore
        f.write(json)


def read_params() -> Optional[Params]:
    file = os.path.join(cur_session_dir(), PARAMS_FILE)
    if not os.path.isfile(file):
        return None
    with open(file, "r") as f:
        return Params.from_json(f.read())  #type: ignore


def create_dir_if_needed(path: str):
    os.makedirs(path, exist_ok=True)


def subdir(*components: Union[str, int]) -> str:
    path = os.path.join(cur_session_dir(), *[str(c) for c in components])
    os.makedirs(path, exist_ok=True)
    return path


def file(*components: Union[str, int]) -> str:
    *dirs, file = components
    return os.path.join(subdir(*dirs), str(file))


def log(msg: str, style=None) -> None:
    if style is None: style = lambda s: s
    styles = {
        'bold': fx.bold,
        'red': fg.red,
        'yellow': fg.yellow,
        'boldyellow': fg.boldyellow,
        'header': fg.boldyellow }
    if style == 'header':
        msg = "\n\n\n" + msg
    if style in styles:
        style = styles[style]
    print(style(msg), end="\n\n")
    session_dir = cur_session_dir()
    os.makedirs(session_dir, exist_ok=True)
    file = os.path.join(session_dir, LOG_FILE)
    with open(file, 'a') as f:
        f.write(msg + "\n\n")


T = TypeVar('T')

def timed(f: Callable[[], T]) -> tuple[T, float]:
    t0 = timeit.default_timer()
    y = f()
    t1 = timeit.default_timer()
    return y, (t1-t0)


def completed_iteration_numbers(agent_dir: str) -> Iterable[int]:
    for d in os.listdir(agent_dir):
        sub = os.path.join(agent_dir, d)
        if os.path.isdir(sub) and d.isnumeric():
            if os.path.isfile(os.path.join(sub, TRAINING_DIR, NETWORK_FILE)):
                yield int(d)


def completed_iteration_dirs(agent_dir: str) -> Iterable[str]:
    for n in completed_iteration_numbers(agent_dir):
        yield os.path.join(agent_dir, str(n))


def remove_session_problems_and_samples(session_dir: str) -> None:
    """
    Remove all problems and samples from a session to make
    it lighter on disk while keeping enough info for future analysis.
    """
    def rmdir(dir: str) -> None:
        if os.path.isdir(dir):
            print(f"Removing: {dir}")
            shutil.rmtree(dir)
    def in_dir(dir: str) -> None:
        rmdir(os.path.join(dir, PROBLEMS_DIR))
        rmdir(os.path.join(dir, SAMPLES_DIR))
    def for_agent(agent_dir: str) -> None:
        dir = os.path.join(session_dir, agent_dir)
        for sub in completed_iteration_dirs(dir):
            in_dir(os.path.join(sub, TRAIN_DATA_DIR))
            in_dir(os.path.join(sub, VALIDATION_DATA_DIR))
    in_dir(os.path.join(session_dir, SOLVER_PROBLEMS_DIR))
    for_agent(TEACHER_DIR)
    for_agent(SOLVER_DIR)
