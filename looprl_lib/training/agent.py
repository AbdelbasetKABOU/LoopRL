"""
Training a Looprl AlphaZero agent
"""

import json
import os
import random
import sys
import traceback
from dataclasses import dataclass, replace
from os.path import join
from typing import Any, Awaitable, Callable, Optional, cast

import looprl
import looprl_lib
import numpy as np
import ray
from looprl import AgentSpec, CamlRng, Prog, TeacherResult
from looprl_lib.disk_recorder import DiskRecorder, indexed_elements_in_dir
from looprl_lib.env_wrapper import (ChoiceState, FinalState, Oracle,
                                    StateWrapper, init_solver, init_teacher)
from looprl_lib.events import EventsSpec
from looprl_lib.inference import ServerHandle, pmap_with_oracle
from looprl_lib.net_util import (NETWORK_FILE, init_and_save_network,
                                 update_network)
from looprl_lib.params import AgentParams, Params
from looprl_lib.pretraining import PretrainingDataset
from looprl_lib.samples import (SOLVER_UNSERIALIZER, TEACHER_UNSERIALIZER,
                                SamplesDataset, Unserializer, sample_to_string)
from looprl_lib.self_play import (Trace, aggregate_trace_summaries,
                                  generate_samples, self_play_episode,
                                  trace_summary_stats)
from looprl_lib.training.session import (PRETRAINING_DIR, PROBLEMS_DIR,
                                         SAMPLES_DIR, SOLVER_DIR,
                                         SOLVER_PROBLEMS_DIR,
                                         SOLVER_PROBLEMS_FILE, STATS_FILE,
                                         TEACHER_DIR, TRAIN_DATA_DIR,
                                         TRAINING_DIR, VALIDATION_DATA_DIR,
                                         create_dir_if_needed, file, log,
                                         subdir)
from torch.utils.data import ConcatDataset

# Create a problem to be solved
# The integer argument corresponds to a problem id
ProblemFactory = Callable[
    [int, CamlRng, np.random.Generator, AgentParams, Oracle],
    Awaitable[StateWrapper]]

# Compute a dictionary summarizing a solution to be saved
# The Any argument corresponds to a success value or None
SolutionExporter = Callable[[StateWrapper, Any, Trace], dict[str, Any]]


@dataclass
class Agent:
    name: str
    dir: str
    spec: AgentSpec
    params: AgentParams
    max_cuda_memory_fraction: Optional[float]
    unserialize: Unserializer
    make_problem_generator: Callable[[int], ProblemFactory]
    export_solution: SolutionExporter
    log: Callable[[str], None] = lambda msg: log(msg)

    #####
    ##### Session agnostic methods
    #####

    def start_inference_server(self, net_file: Optional[str]) -> ServerHandle:
        probe_size = self.params.search.max_probe_size
        action_size = self.params.search.max_action_size
        espec = EventsSpec(self.spec)
        if net_file is not None:
            self.log(f"Starting an inference server with: {net_file}")
            return looprl_lib.inference.start_inference_server(
                net_file, self.params.network,
                self.params.encoding.tensorizer_config,
                probe_size, action_size, espec,
                max_cuda_memory_fraction=self.max_cuda_memory_fraction,
                num_waited_processes=self.params.num_waited_processes)
        else:
            return ServerHandle(None, probe_size, action_size, espec)

    def gen_samples(
        self,
        net_file: Optional[str],
        target_dir: str,
        num_probs: int,
        gumbel_exploration: bool,
        save_failures: bool
    ) -> dict[str, float]:
        restart_ray()  # needed before instantiating an inference server
        server = self.start_inference_server(net_file)
        samples_dir = join(target_dir, SAMPLES_DIR)
        create_dir_if_needed(samples_dir)
        problems_dir = join(target_dir, PROBLEMS_DIR)
        create_dir_if_needed(problems_dir)
        samples_recorder = DiskRecorder.remote(samples_dir)  #type: ignore
        problems_recorder = DiskRecorder.remote(problems_dir)  #type: ignore
        inputs = [(i, generate_rng_seed()) for i in range(num_probs)]
        espec = EventsSpec(self.spec)
        num_workers = self.params.num_workers
        make_problem = self.make_problem_generator(num_probs)
        async def solve_print_exn(input: tuple[int, list[int]], oracle: Oracle):
            # Necessary to avoid silent failures
            try: return await solve(input, oracle)
            except Exception:
                traceback.print_exc()
                assert False
        async def solve(input: tuple[int, list[int]], oracle: Oracle):
            i, seed = input
            camlrng = CamlRng(seed=seed)
            nprng = np.random.default_rng(seed=seed)
            init_state = await make_problem(
                i, camlrng, nprng, self.params, oracle)
            mcts = self.params.mcts
            if self.params.use_biases_for_initial_policy and net_file is None:
                mcts = replace(mcts, bias_eps=1.0)
            trace, success_value = await self_play_episode(
                init_state, mcts, nprng,
                gumbel_exploration=gumbel_exploration)
            if save_failures or success_value is not None:
                exported = json.dumps(
                    self.export_solution(init_state, success_value, trace))
                problem_id = ray.get(
                    problems_recorder.push.remote([exported]))[0]
            else:
                problem_id = None
            samples = [
                sample_to_string(s)
                for s in generate_samples(trace, problem_id, espec)]
            ray.get(samples_recorder.push.remote(samples))
            return trace_summary_stats(trace, espec.agent_spec)
        summaries = pmap_with_oracle(
            solve_print_exn, inputs, server, num_workers=num_workers,
            num_processes=self.params.num_processes)
        # Log and save statistics
        with open(join(target_dir, STATS_FILE), 'w') as f:
            json.dump(summaries, f)  # type: ignore
        summary = aggregate_trace_summaries(summaries)
        slines = [f"{k:40s} {v:.2f}" for k, v in summary.items()]
        self.log("Iteration statistics:")
        self.log("\n".join(slines))
        return summary

    #####
    ##### Session related methods
    #####

    def prev_network_file(self, it_num: int) -> str:
        it_dir = it_num - 1 if it_num > 0 else PRETRAINING_DIR
        return file(self.dir, str(it_dir), TRAINING_DIR, NETWORK_FILE)

    def final_network_file(self) -> str:
        return self.prev_network_file(self.params.num_iters)

    def start_inference_server_for_iter(self, it_num: int) -> ServerHandle:
        net_file = self.prev_network_file(it_num) if it_num > 0 else None
        return self.start_inference_server(net_file)

    def pretrain(self) -> None:
        net_params = self.params.network
        tconf=self.params.encoding.tensorizer_config
        if not self.params.pretraining.enable_pretraining:
            netfile = file(
                self.dir, PRETRAINING_DIR, TRAINING_DIR, NETWORK_FILE)
            init_and_save_network(net_params, tconf, self.spec, netfile)
            return
        num_samples = self.params.pretraining.num_samples
        num_vsamples = self.params.pretraining.num_validation_samples
        def make_dataset(num_samples: int):
            return PretrainingDataset(
                size=num_samples,
                encoding=self.params.encoding,
                agent_spec=self.spec,
                true_false=False,
                randomize_uids=False)
        update_network(
            train_set=make_dataset(num_samples),
            validation_set=make_dataset(num_vsamples),
            in_net_file=None,
            training_dir=subdir(self.dir, PRETRAINING_DIR),
            net_params=net_params,
            tconf=tconf,
            train_params=self.params.pretraining.training,
            agent_spec=self.spec,
            max_cuda_memory_fraction=self.max_cuda_memory_fraction,
            log=log)

    def gen_iter_samples(self, it_num: int) -> None:
        net_file = self.prev_network_file(it_num) if it_num > 0 else None
        def gen(data_dir: str, num_problems: int, save_failures: bool):
            self.gen_samples(
                target_dir=subdir(self.dir, it_num, data_dir),
                net_file=net_file,
                num_probs=num_problems,
                gumbel_exploration=True,
                save_failures=save_failures)
        # We don't save failures in TRAIN_DATA_DIR because these
        # are used as challenge problems for the solver.
        gen(TRAIN_DATA_DIR, self.params.num_problems_per_iter, False)
        gen(VALIDATION_DATA_DIR, self.params.num_validation_problems, True)

    def update_network(self, it_num: int):
        def make_dataset(i: int, data_dir: str):
            samples_dir = subdir(self.dir, i, data_dir, SAMPLES_DIR)
            log(f"Loading samples from {samples_dir}")
            return SamplesDataset(
                unserialize=self.unserialize,
                encoding=self.params.encoding,
                probe_size=self.params.search.max_probe_size,
                action_size=self.params.search.max_action_size,
                randomize_uids=False,
                dir=samples_dir)
        def make_concatenated_dataset(data_dir: str):
            if it_num < len(self.params.training_window):
                window_size = self.params.training_window[it_num]
            else:
                window_size = self.params.training_window[-1]
            window_size = min(window_size, it_num + 1)
            start = it_num - window_size + 1
            end = it_num + 1
            datasets = [make_dataset(i, data_dir) for i in range(start, end)]
            return ConcatDataset(datasets)
        train_set = make_concatenated_dataset(TRAIN_DATA_DIR)
        valid_set = make_concatenated_dataset(VALIDATION_DATA_DIR)
        update_network(
            train_set=train_set,
            validation_set=valid_set,
            in_net_file=self.prev_network_file(it_num),
            training_dir=subdir(self.dir, it_num, TRAINING_DIR),
            net_params=self.params.network,
            tconf=self.params.encoding.tensorizer_config,
            train_params=self.params.training,
            agent_spec=self.spec,
            max_cuda_memory_fraction=self.max_cuda_memory_fraction,
            log=log)


def restart_ray() -> None:
    """
    There is currently a bug in ray where the GPU does
    not get released when an actor is destroyed. Thus, one
    should call `restart_ray` between each instantiation of
    a GPU actor.
    """
    ray.shutdown()
    ray.init()


def start_or_restart_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()
    ray.init()


def generate_rng_seed() -> list[int]:
    return np.random.randint(0, 1<<32, size=4).tolist()


async def make_teacher_problem(
    id: int,
    camlrng: CamlRng,
    nprng: np.random.Generator,
    params: AgentParams,
    oracle: Oracle
) -> StateWrapper:
    initial_state = init_teacher(params, camlrng, oracle)
    while True:
        state = initial_state
        while True:
            status = await state.status()
            if isinstance(status, FinalState):
                break
            elif isinstance(status, ChoiceState):
                if state.is_chance_node:
                    p = state.bias_distribution
                    assert p is not None
                    i = nprng.choice(list(range(len(p))), p=p)
                    state = state.select(i)
                else:
                    return state
            else:
                assert False


def export_teacher_solution(
    initial_state: StateWrapper,
    success_value: Any,
    trace: Trace
):
    res: dict[str, Any] = {}
    if success_value is not None:
        success_dict = cast(TeacherResult, success_value)
        res['problem'] = str(success_dict['problem'])
        res['nonprocessed'] = str(success_dict['nonprocessed'])
    res['spec'] = initial_state.probe.meta()['spec']
    res['spec_sexp'] = initial_state.probe.meta()['spec_sexp']
    res['outcome'] = trace.outcome
    res['events'] = trace.events
    return res


def teacher(params: Params) -> Agent:
    return Agent(
        'teacher', TEACHER_DIR, looprl.teacher_spec,
        params.teacher.agent, params.max_cuda_memory_fraction,
        TEACHER_UNSERIALIZER,
        lambda n: make_teacher_problem, export_teacher_solution)


def write_solver_problems_file(params: Params) -> None:
    num_iters = params.teacher.agent.num_iters
    num_taken = params.num_teacher_iters_used_by_solver
    dirs = [subdir(SOLVER_PROBLEMS_DIR, PROBLEMS_DIR)]
    for i in range(max(0, num_iters-num_taken), num_iters):
        dirs.append(subdir(TEACHER_DIR, i, TRAIN_DATA_DIR, PROBLEMS_DIR))
    with open(file(SOLVER_PROBLEMS_FILE), 'w') as f:
        f.write("\n".join(dirs))


def generate_problems_inventory() -> list[str]:
    with open(file(SOLVER_PROBLEMS_FILE), 'r') as f:
        sources = [s.strip() for s in f.readlines()]
    inventory: list[str] = []
    for dir in sources:
        assert os.path.isdir(dir), f"Not a directory: {dir}"
        files = indexed_elements_in_dir(dir)
        if not files:
            print(f"WARNING: no problem stored in {dir}", file=sys.stderr)
        inventory += [join(dir, str(f)) for f in files]
    return inventory


def make_solver_problem_factory(
    problem_files: list[str],
    num_problems: int,
) -> ProblemFactory:
    problem_files = problem_files.copy()
    n = len(problem_files)
    problems = [
        problem_files[random.randint(0, n-1)]
        for _ in range(num_problems)]
    async def make_problem(
        id: int,
        camlrng: CamlRng,
        nprng: np.random.Generator,
        params: AgentParams,
        oracle: Oracle
    ):
        file = problems[id]
        with open(file, 'r') as f:
            problem = Prog(json.load(f)['problem'])
        return init_solver(problem, params, oracle)
    return make_problem


def make_default_solver_problem_factory(num_problems: int) -> ProblemFactory:
    problem_files = generate_problems_inventory()
    return make_solver_problem_factory(problem_files, num_problems)


def export_solver_solution(
    initial_state: StateWrapper,
    success_value: Any,
    trace: Trace
):
    res: dict[str, Any] = {}
    if success_value is not None:
        success_prog = cast(Prog, success_value)
        res['solved'] = str(success_prog)
    res['outcome'] = trace.outcome
    res['events'] = trace.events
    return res


def solver(params: Params) -> Agent:
    return Agent(
        'solver', SOLVER_DIR, looprl.solver_spec,
        params.solver.agent, params.max_cuda_memory_fraction,
        SOLVER_UNSERIALIZER,
        make_default_solver_problem_factory, export_solver_solution)


def solver_problems_generation_step(ps: Params) -> None:
    ps = ps.update({'::mcts.fpu_red': 0.1})
    agent = teacher(ps)
    net_file = agent.final_network_file()
    agent.gen_samples(
        net_file=net_file,
        target_dir=subdir(SOLVER_PROBLEMS_DIR),
        num_probs=ps.extra_teacher_problems,
        gumbel_exploration=False,
        save_failures=False)
    write_solver_problems_file(ps)
