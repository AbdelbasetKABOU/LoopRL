"""
The Looprl Training Loop.

To run a dummy training session, just launch:
    python -m looprl_lib.training.loop -f -p toy -d ./sessions/test
"""

import argparse
import json
import os
import shutil
import timeit
from typing import Any, Callable, Iterator, Literal, Optional, Union

import ray
from looprl_lib.training.agent import (Agent, restart_ray, solver,
                                       solver_problems_generation_step,
                                       teacher)

from ..params import STD_PARAMS, Params, ParamsDiff
from .session import (PARAMS_DIFF_FILE, PARAMS_FILE, SOLVER_PROBLEMS_FILE,
                      STAGE_FILE, TIME_FILE, cur_session_dir, file, log,
                      read_params, set_cur_session_dir, write_params)

#####
## Training Stages
#####


IterNum = Union[Literal['pre'], int]


# A stage indicates the next thing to do
Stage = Union[
    tuple[Literal['done']],
    tuple[Literal['generate-problems']],
    tuple[Literal['teacher', 'solver'], Literal['pretrain']],
    tuple[Literal['teacher', 'solver'], int, Literal['generate', 'update']]]


StageProcedure = Callable[[], None]


def parse_int_or_str(s: str):
    try: return int(s)
    except ValueError: return s


def parse_stage(s: str) -> Stage:
    return tuple([parse_int_or_str(e) for e in s.split(":")])  #type: ignore


def show_stage(s: Stage) -> str:
    return ":".join(str(e) for e in s)


def iter_stages_for(agent: Agent) -> Iterator[tuple[Stage, StageProcedure]]:
    name: Literal['solver', 'teacher'] = agent.name  #type: ignore
    yield ((name, 'pretrain'), lambda: agent.pretrain())
    for i in range(agent.params.num_iters):
        yield ((name, i, 'generate'),
            lambda i=i: agent.gen_iter_samples(i))  #type: ignore
        yield ((name, i, 'update'),
            lambda i=i: agent.update_network(i))  #type: ignore


def iter_stages(ps: Params) -> Iterator[tuple[Stage, StageProcedure]]:
    teacher_agent = teacher(ps)
    solver_agent = solver(ps)
    yield from iter_stages_for(teacher_agent)
    yield (('generate-problems',), lambda: solver_problems_generation_step(ps))
    yield from iter_stages_for(solver_agent)
    yield (('done',), lambda: None)


def stages_dict(ps: Params) -> dict[Stage, StageProcedure]:
    return {s: proc for s, proc in iter_stages(ps)}


def initial_stage() -> Stage:
    return ('teacher', 'pretrain')


def next_stage(cur: Stage, params: Params) -> Optional[Stage]:
    found_cur = False
    for s, _ in iter_stages(params):
        if found_cur: return s
        if s == cur: found_cur = True
    return None


def read_training_stage() -> Stage:
    file = os.path.join(cur_session_dir(), STAGE_FILE)
    if not os.path.isfile(file):
        return initial_stage()
    with open(file, "r") as f:
        return parse_stage(f.read())


def write_training_stage(stage: Stage) -> None:
    file = os.path.join(cur_session_dir(), STAGE_FILE)
    with open(file, "w") as f:
        f.write(show_stage(stage))


def test_enumerate_stages(params: Params) -> None:
    for s, _ in iter_stages(params):
        print(show_stage(s))


def log_stage_duration(stage: Stage, duration: float) -> None:
    file = os.path.join(cur_session_dir(), TIME_FILE)
    times: dict[str, float] = {}
    if os.path.isfile(file):
        with open(file, "r") as f:
            times = json.load(f)
    times[show_stage(stage)] = duration
    with open(file, "w") as f:
        json.dump(times, f)


#####
## Session manipulation
#####


def write_params_diff(params_diff: ParamsDiff) -> None:
    with open(file(PARAMS_DIFF_FILE), 'w') as f:
        json.dump(params_diff, f)


def init_session(session_dir: str, params_diff: ParamsDiff) -> None:
    set_cur_session_dir(session_dir)
    write_params(Params.from_diff(params_diff))
    write_params_diff(params_diff)


def delete_session(session_dir: str) -> None:
    # To avoid accidental deletes
    assert "session" in session_dir
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir)


#####
## Session execution
#####


def execute_stage(stage: Stage, params: Params) -> None:
    restart_ray()
    log(f"Executing stage {show_stage(stage)}", 'header')
    stages_dict(params)[stage]()


def resume_session(session_dir: str) -> None:
    set_cur_session_dir(session_dir)
    params = read_params()
    assert params is not None, "No params file found."
    stage: Optional[Stage] = read_training_stage()
    while stage is not None:
        write_training_stage(stage)
        t0 = timeit.default_timer()
        execute_stage(stage, params)
        t1 = timeit.default_timer()
        log_stage_duration(stage, t1-t0)
        stage = next_stage(stage, params)


def run_session(
    session_dir: str,
    params_diff: ParamsDiff,
    force_new: bool = False
) -> None:
    if force_new:
        delete_session(session_dir)
    init_session(session_dir, params_diff)
    resume_session(session_dir)


def run_solver_only_session(
    session_dir: str,
    teacher_dir: str,
    params_diff: ParamsDiff,
):
    delete_session(session_dir)
    init_session(session_dir, params_diff)
    shutil.copyfile(
        os.path.join(teacher_dir, SOLVER_PROBLEMS_FILE),
        file(SOLVER_PROBLEMS_FILE))
    with open(file(STAGE_FILE), 'w') as f:
        f.write(show_stage(('solver', 'pretrain')))
    resume_session(session_dir)


def overwrite_params(session_dir: str, params_diff: ParamsDiff):
    params = Params.from_diff(params_diff)
    with open(os.path.join(session_dir, PARAMS_DIFF_FILE), 'w') as f:
        json.dump(params_diff, f)
    with open(os.path.join(session_dir, PARAMS_FILE), 'w') as f:
        serialized = params.to_json(indent=4)  #type: ignore
        f.write(serialized)


def overwrite_stage(session_dir: str, stage: str):
    parse_stage(stage)
    with open(os.path.join(session_dir, STAGE_FILE), 'w') as f:
        f.write(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='looprl-training',
        description='The Looprl Training Loop.')
    parser.add_argument('-f', '--force-new', action='store_true')
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('-p', '--preset', type=str)
    parser.add_argument('params_diff', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.preset is None:
        diff: dict[str, Any] = {}
    else:
        assert args.preset in STD_PARAMS, f"Unrecognized preset: {args.preset}"
        diff = STD_PARAMS[args.preset].copy()
    for kv in args.params_diff:
        k, v = kv.split("=")
        diff[k] = json.loads(v)
    ray.init()
    run_session(args.dir, diff, args.force_new)
