"""
Search budget computation for different solving methods
"""


import math
import os
from dataclasses import replace
from typing import Any, Callable, Optional

import looprl
import numpy as np
import pandas as pd  # type: ignore
import ray
from looprl import Prog
from looprl_lib import inference
from looprl_lib.env_wrapper import (ChoiceState, FinalState, OutcomeType,
                                    StateWrapper, init_solver)
from looprl_lib.events import EventsSpec
from looprl_lib.examples import code2inv, iter_code2inv
from looprl_lib.experiments.util.cache_util import cached
from looprl_lib.inference import ServerHandle, pmap_with_oracle
from looprl_lib.mcts import init_mcts
from looprl_lib.net_util import NETWORK_FILE
from looprl_lib.params import Params
from looprl_lib.training.agent import generate_rng_seed, start_or_restart_ray
from looprl_lib.training.session import (SOLVER_DIR, TRAINING_DIR,
                                         completed_iteration_numbers)
from looprl_lib.util import apply_temperature

CACHE_DIR = "cached"
TEX_TABLE_FILE = "code2inv.tex"
BEST_KNOWN_CACHE_FILE = os.path.join(CACHE_DIR, "best_known")
PROBS_DATA_CACHE_DIR = os.path.join(CACHE_DIR, "probs_data")

BASELINE_LABEL = "Baseline"
UNTRAINED_TEACHER_LABEL = "Network (untrained teacher)"
TRAINED_TEACHER_LABEL = "Network (trained teacher)"


async def single_attempt(
    state: StateWrapper,
    rng: np.random.Generator,
    tau: float = 1.
) -> Optional[float]:
    while True:
        status = await state.status()
        if isinstance(status, FinalState):
            if status.outcome_type == OutcomeType.SUCCESS:
                return state.final_reward
            else:
                return None
        assert isinstance(status, ChoiceState)
        p = status.oracle_output.policy
        p = apply_temperature(p, tau)
        action_ids = list(range(len(state.actions)))
        i = rng.choice(action_ids, p=p)
        state = state.select(i)


async def best_known_reward(
    prog: Prog,
    params: Params
) -> float:
    mcts_params = replace(params.solver.agent.mcts,
        num_considered_actions=16,
        num_simulations=4096,
        max_tree_size=None)
    state = init_solver(prog, params.solver.agent, oracle=None)
    mcts = await init_mcts(state, mcts_params)
    await mcts.explore()
    return mcts.success_value if mcts.success_value is not None else -1.0


async def successes_vec(
    params: Params,
    best_known: dict[int, float],
    rng_seed: list[int],
    oracle: Optional[Callable],
    tau: float = 1.,
    eps: float = 0.01
) -> np.ndarray:
    res: list[int] = []
    rng = np.random.default_rng(seed=rng_seed)
    for i, p in iter_code2inv():
        state = init_solver(p, params.solver.agent, oracle=oracle)
        reward = await single_attempt(state, rng, tau=tau)
        if reward is None:
            reward = -1.0
        success = reward + eps >= best_known[i]
        res.append(1 if success else 0)
    return np.array(res)


def start_inference_server(net_file: Optional[str], params: Params):
    if net_file is None:
        return ServerHandle(None,
            probe_size=params.solver.agent.search.max_probe_size,
            action_size=params.solver.agent.search.max_action_size,
            espec=EventsSpec(looprl.solver_spec))
    else:
        return inference.start_inference_server(
            net_file,
            params.solver.agent.network,
            params.solver.agent.encoding.tensorizer_config,
            probe_size=params.solver.agent.search.max_probe_size,
            action_size=params.solver.agent.search.max_action_size,
            max_cuda_memory_fraction=params.max_cuda_memory_fraction,
            espec=EventsSpec(looprl.solver_spec))


def produce_best_known_rewards(
    params: Params,
    num_workers: int = 10
) -> dict[int, float]:
    print("Producing best known rewards.")
    async def solve(i, _):
        prog = code2inv(i)
        return await best_known_reward(prog, params)
    probs = [i for i, _ in iter_code2inv()]
    server = start_inference_server(None, params)
    start_or_restart_ray()
    res = pmap_with_oracle(solve, probs, server, num_workers=num_workers)
    return {i: r for (i, r) in zip(probs, res)}


def successes_probabilities(
    params: Params,
    best_known: dict[int, float],
    net_file: Optional[str],
    num_workers: int = 400,
    tau: float = 1.0,
    n: int = 1000,
) -> np.ndarray:
    print(f"Computing success probabilities for: {net_file}")
    seeds = [generate_rng_seed() for _ in range(n)]
    async def solve(seed, oracle):
        return await successes_vec(params, best_known, seed, oracle, tau=tau)
    start_or_restart_ray()
    server = start_inference_server(net_file, params)
    vecs = pmap_with_oracle(solve, seeds, server, num_workers=num_workers)
    return np.stack(vecs).mean(axis=0)


def last_net_file(session_dir: str):
    solver_dir = os.path.join(session_dir, SOLVER_DIR)
    net_file = None
    for i in range(100):
        cand = os.path.join(solver_dir, str(i), TRAINING_DIR, NETWORK_FILE)
        if os.path.isfile(cand):
            net_file = cand
        else:
            assert net_file is not None
            return net_file


def percent_solved_expectation(probs: np.ndarray, num_attempts: int):
    n = len(probs)
    return np.sum(1 - (1 - probs) ** num_attempts) * 100 / n


def viz_probs(data: dict[str, np.ndarray]):
    table = pd.DataFrame(
        data,
        index=[i for i, _ in iter_code2inv()])
    print(table.to_string())


def scores_table(probs: dict[str, np.ndarray], budgets: list[int]):
    scores = {
        k: [percent_solved_expectation(p, b) for b in budgets]
        for k, p in probs.items()}
    return pd.DataFrame(scores, index=budgets)


def scores_table_to_latex(scores: pd.DataFrame, std: pd.DataFrame):
    n = scores.shape[0]
    budgets_strs = scores.columns
    print(r"\begin{tabular}{l" + "c" * n + "}")
    print(r"  \toprule")
    print(r"  & \multicolumn{" + str(n) + r"}{c}{Search budget} \\")
    print(r"  \cmidrule(r){2-" + str(n+1) + r"}")
    print(r"  Configuration & " + " & ".join(budgets_strs) + r" \\")
    print(r"  \midrule")
    for col in scores.columns:
        col_scores = scores[col]
        col_std = std[col]
        scores_strs: list[str] = []
        for m, s in zip(col_scores, col_std):
            score_str = f"{m:.1f}"
            if not math.isnan(s):
                score_str += f" \\pm {s:.1f}"
            scores_strs.append(score_str)
        print(r"  " + col + " & " + " & ".join(scores_strs) + r" \\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")


def net_at_iter(session_dir: str, it_num: int) -> str:
    return os.path.join(
        session_dir, SOLVER_DIR, str(it_num), TRAINING_DIR, NETWORK_FILE)


def search_all_nets(
    outdir: str,
    session_dir: str,
    tau: float = 1.0,
    n: int = 1000,
    it_nums: list[int] = [1, 5, 10, 15 , 19],
    budgets: list[int] = [1, 2, 3, 4, 5, 10, 50, 100]
):
    params = Params()
    ray.init()
    best_known = cached(os.path.join(outdir, BEST_KNOWN_CACHE_FILE), lambda:
        produce_best_known_rewards(params))
    data: dict[str, Any] = {}
    for i in it_nums:
        net_file = net_at_iter(session_dir, i)
        print(f"Using net file at: {net_file}")
        probs = cached(f"{outdir}/net_{i}.json", lambda:
        successes_probabilities(
            params, best_known, net_file,
            n=n, tau=tau))
        data[str(i)] = probs
    table = scores_table(data, budgets)
    print(table)
    return table


def generate_table(
    outdir: str,
    network: str,
    n: int = 10_000,
    budgets: list[int] = [1, 2, 3, 5]
) -> pd.DataFrame:
    params = Params()
    best_known = cached(os.path.join(outdir, BEST_KNOWN_CACHE_FILE), lambda:
        produce_best_known_rewards(params))
    data: dict[str, Any] = {}
    for title, cache_file, net_file in [
        ('Baseline', 'baseline.json', None),
        ('Network', 'net.json', network)]:
        print(f"Using net file at: {net_file}")
        probs = cached(os.path.join(outdir, cache_file), lambda:
            successes_probabilities(
                params, best_known, net_file, n=n, tau=1.0))
        data[title] = probs
    return scores_table(data, budgets)


def generate_tables(
    outdir: str,
    networks: list[str],
    n: int = 10_000,
    budgets: list[int] = [1, 2, 3, 5]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for i, net in enumerate(networks):
        sub_outdir = os.path.join(outdir, str(i))
        table = generate_table(sub_outdir, net, n, budgets)
        tables.append(table)
    full = pd.concat(tables, axis=0).groupby(level=0)
    return full.mean(), full.std()


def find_best_net(
    outdir: str,
    session_dir: str
):
    print(f"Finding the best network in: {session_dir}")
    params = Params()
    best_known = cached(os.path.join(outdir, BEST_KNOWN_CACHE_FILE), lambda:
        produce_best_known_rewards(params))
    best_i = 0
    best_score = 0
    for i in completed_iteration_numbers(os.path.join(session_dir, SOLVER_DIR)):
        net_file = net_at_iter(session_dir, i)
        probs = successes_probabilities(
            params, best_known, net_file,
            n=1, tau=0.)
        score = percent_solved_expectation(probs, 1)
        print(f"{i:2d}: {score:.2f}%")
        if score > best_score:
            best_i = i
            best_score = score
    return best_i


def generate_no_search_probs_data(
    outdir: str,
    trained_teacher_session_dirs: str,
    untrained_teacher_session_dirs: str
):
    params = Params()
    best_known = cached(os.path.join(outdir, BEST_KNOWN_CACHE_FILE), lambda:
        produce_best_known_rewards(params))
    data: dict[str, np.ndarray] = {}
    # Baseline
    probs = successes_probabilities(
        params, best_known, None, n=10_000, tau=1.0)
    data[BASELINE_LABEL] = probs
    # Trained and untrained teachers
    for label, dir in [
        (UNTRAINED_TEACHER_LABEL, untrained_teacher_session_dirs),
        (TRAINED_TEACHER_LABEL, trained_teacher_session_dirs)]:
        best_net_it = find_best_net(outdir, dir)
        print(f"Using network from iteration {best_net_it}")
        net_file = net_at_iter(dir, best_net_it)
        probs = successes_probabilities(
            params, best_known, net_file, n=1, tau=0.)
        data[label] = probs
    return data


def generate_no_search_single_table(
    outdir: str,
    cached_name: str,
    sessions: tuple[str, str],
):
    data = cached(os.path.join(outdir, cached_name), lambda:
        generate_no_search_probs_data(outdir, sessions[0], sessions[1]))
    table = scores_table(data, budgets=[1])
    return table.transpose()[1]


def tex_no_search_table(
    scores: pd.DataFrame,
    std: pd.DataFrame,
) -> str:
    ls: list[str] = []
    ls.append(r"\begin{tabular}{lc}")
    ls.append(r"  \toprule")
    ls.append(r"  Policy & \% Problems solved \\")
    ls.append(r"  \midrule")
    for col in [BASELINE_LABEL, UNTRAINED_TEACHER_LABEL, TRAINED_TEACHER_LABEL]:
        col_score = scores[col]
        col_std = std[col]
        score_str = f"{col_score:.1f}"
        if not math.isnan(col_std):
            score_str += f" \\pm {col_std:.1f}"
        ls.append(r"  " + col + " & ${" + score_str + r"}$ \\")
    ls.append(r"  \bottomrule")
    ls.append(r"\end{tabular}")
    return "\n".join(ls)


def generate_and_print_no_search_table(
    outdir: str,
    sessions: list[tuple[str, str]]
):
    tables = [
        generate_no_search_single_table(
            outdir, os.path.join(PROBS_DATA_CACHE_DIR, str(i)), ss)
        for i, ss in enumerate(sessions)]
    full = pd.concat(tables, axis=0).groupby(level=0)
    tex = tex_no_search_table(full.mean(), full.std())
    print(tex)
    with open(os.path.join(outdir, TEX_TABLE_FILE), 'w') as f:
        f.write(tex)


if __name__ == '__main__':
    generate_and_print_no_search_table(
        outdir="out",
        sessions=[
            ("../sessions/final", "../sessions/no-teacher"),
            ("../sessions/final2", "../sessions/no-teacher2")])
