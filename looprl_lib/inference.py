"""
An efficient inference server to batch network inference
requests across worker threads.
"""

import asyncio
import json
import math
import timeit
import traceback
from dataclasses import dataclass, field
from typing import (Awaitable, Callable, List, Optional, Sequence, TypeVar,
                    Union)

import numpy as np
import psutil  # type: ignore
import ray
import torch
from looprl import AgentSpec, TensorizerConfig
from ray.actor import ActorHandle

from looprl_lib.events import EventsSpec

from .env_wrapper import Oracle, OracleOutput, dummy_oracle
from .net_util import make_network
from .params import NetworkParams
from .progress_bar import ProgressBar, ProgressBarActor
from .tensors import ChoicesBatch, ChoiceTensors, to_device, to_torch


def evaluate_batch(
    net,
    batch: ChoicesBatch,
    stats: Optional[list['InferenceEvent']] = None
) -> list[OracleOutput]:
    t0 = timeit.default_timer()
    batch = to_torch(batch)
    batch = to_device(batch, net.get_device())
    t1 = timeit.default_timer()
    with torch.no_grad():
        try:
            vpreds_mat, ascores = net(batch)
        except Exception as e:
            size = batch.probes.batch_size
            actions = batch.actions.batch_size
            print(f"On a batch of size {size} with {actions} actions:")
            raise e
    t2 = timeit.default_timer()
    vpreds_mat = vpreds_mat.to(device='cpu').numpy()
    vpreds = [vpreds_mat[i,:] for i in range(vpreds_mat.shape[0])]
    ascores = ascores.to(device='cpu').numpy()
    t3 = timeit.default_timer()
    ascores = split_seq(ascores, batch.num_actions)
    assert len(vpreds) == len(ascores)
    t4 = timeit.default_timer()
    if stats is not None:
        stats.append(InferenceEvent(f"Make batch", "Make batch", t0, t1))
        stats.append(InferenceEvent(f"{len(vpreds)}", "GPU run", t1, t2))
        stats.append(InferenceEvent(f"To CPU", "To CPU", t2, t3))
        stats.append(InferenceEvent(f"Actions", "Get Action Scores", t3, t4))
    return [OracleOutput(v, p) for v, p in zip(vpreds, ascores)]


T = TypeVar('T')
def split_seq(
    xs: Sequence[T],
    sizes: Union[np.ndarray, Sequence[int]]
) -> list[Sequence[T]]:
    idx = np.concatenate((np.array([0]), np.cumsum(sizes)))
    return [xs[idx[i]:idx[i+1]] for i in range(len(sizes))]


def make_nonbatched_network_oracle(
    net_state_dict: dict,
    net_params: NetworkParams,
    tconf: TensorizerConfig,
    agent_spec: AgentSpec
) -> Oracle:
    net = make_network(net_params, tconf, agent_spec).to(device='cpu')
    net.train(mode=False)
    net.load_state_dict(net_state_dict)
    async def eval(query: ChoiceTensors) -> OracleOutput:
        nonlocal net
        return evaluate_batch(net, ChoicesBatch.make([query]))[0]
    return eval


#####
## Dispatcher / server
#####


class InferenceServer:

    def __init__(self, num_waited: Optional[int] = None) -> None:
        self.num_waited = num_waited
        self.num_clients = 0
        self.pending: list[ChoicesBatch] = []
        self.answers: list[Sequence[OracleOutput]] = []
        self.full = asyncio.Event()
        self.stats: InferenceStats = InferenceStats()

    def load_network(
        self,
        net_params: NetworkParams,
        tconf: TensorizerConfig,
        agent_spec: AgentSpec,
        net_state_dict: Optional[dict],
        max_cuda_memory_fraction: Optional[float],
        device: str
    ) -> None:
        if max_cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(max_cuda_memory_fraction)
        self.net = make_network(net_params, tconf, agent_spec).to(device=device)
        self.net.train(mode=False)
        if net_state_dict is not None:
            self.net.load_state_dict(net_state_dict)

    def eval_batch(
        self,
        queries: list[ChoicesBatch]
    ) -> Sequence[Sequence[OracleOutput]]:
        if not queries:
            return []
        batch = ChoicesBatch.concatenate(queries)
        res = evaluate_batch(self.net, batch, self.stats.inference_events)
        return split_seq(res, [q.batch_size for q in queries])

    def try_eval_batch(self) -> None:
        batch_size = (
            self.num_clients if self.num_waited is None
            else min(self.num_waited, self.num_clients))
        if len(self.pending) >= batch_size:
            answers = self.answers
            pending = self.pending
            full = self.full
            self.pending = []
            self.answers = []
            self.full = asyncio.Event()
            answers[:] = self.eval_batch(pending)
            full.set()

    def add_client(self) -> None:
        self.num_clients += 1

    def remove_client(self) -> None:
        self.num_clients -= 1
        assert self.num_clients >= 0
        self.try_eval_batch()

    async def infer(
        self,
        query: ChoicesBatch,
        worker_id: Optional[int] = None
    ) -> Sequence[OracleOutput]:
        time_recv = timeit.default_timer()
        i = len(self.pending)
        self.pending.append(query)
        answers, full = self.answers, self.full
        self.try_eval_batch()
        await full.wait()
        # Collect stats
        time_ans = timeit.default_timer()
        if worker_id is not None:
            self.stats.worker_events.append(
                WorkerEvent('Query received', worker_id, time_recv))
            self.stats.worker_events.append(
                WorkerEvent('Answer returned', worker_id, time_ans))
        return answers[i]

    def get_stats(self) -> 'InferenceStats':
        return self.stats


#####
## Local servers
#####


@dataclass
class LocalServer:
    server: ActorHandle
    process_id: int
    probe_size: int
    action_size: int

    def __post_init__(self) -> None:
        self.events: list[WorkerEvent] = []
        self.num_clients = 0
        self.pending: list[ChoiceTensors] = []
        self.answers: list[OracleOutput] = []
        self.full = asyncio.Event()

    def add_client(self) -> None:
        self.num_clients += 1

    def add_clients(self, num: int) -> None:
        assert num >= 0
        self.num_clients += num

    def remove_client(self) -> None:
        self.num_clients -= 1
        assert self.num_clients >= 0
        self.try_eval_batch()

    def eval_batch(
        self,
        queries: list[ChoiceTensors]
    ) -> Sequence[OracleOutput]:
        if not queries:
            return []
        batch = ChoicesBatch.make(queries, self.probe_size, self.action_size)
        t0 = timeit.default_timer()
        res = ray.get(self.server.infer.remote(ray.put(batch), self.process_id))
        t1 = timeit.default_timer()
        self.events.append(WorkerEvent('Query sent', self.process_id, t0))
        self.events.append(WorkerEvent('Answer received', self.process_id, t1))
        return res

    def try_eval_batch(self) -> None:
        if len(self.pending) >= self.num_clients:
            answers = self.answers
            pending = self.pending
            full = self.full
            self.pending = []
            self.answers = []
            self.full = asyncio.Event()
            answers[:] = self.eval_batch(pending)
            full.set()

    async def infer(self, query: ChoiceTensors) -> OracleOutput:
        i = len(self.pending)
        self.pending.append(query)
        answers, full = self.answers, self.full
        self.try_eval_batch()
        await full.wait()
        return answers[i]


def make_oracle(server: LocalServer) -> Oracle:
    async def eval(query: ChoiceTensors):
        res = await server.infer(query)
        return res
    return eval


def iter_with_oracle(
    f: Callable[[T, Oracle], Awaitable[None]],
    xs: Sequence[T],
    server: ActorHandle,
    num_threads: int,
    process_id: int,
    probe_size: int,
    action_size: int
) -> list['WorkerEvent']:
    async def task():
        local_server = LocalServer(
            server, process_id, probe_size, action_size)
        oracle = make_oracle(local_server)
        ray.get(server.add_client.remote())
        local_server.add_clients(num_threads)
        await async_map(
            lambda x: f(x, oracle), xs,
            num_threads,
            worker_done=local_server.remove_client)
        ray.get(server.remove_client.remote())
        return local_server.events
    return asyncio.run(task())


def iter_with_dummy_oracle(
    f: Callable[[T, Oracle], Awaitable[None]],
    xs: Sequence[T],
    espec: EventsSpec
) -> list['WorkerEvent']:
    for x in xs:
        asyncio.run(f(x, dummy_oracle(espec)))
    return []


#####
## Async utils
#####


# To avoid silent failures, we write a wrapper for 'asyncio.create_task'
# that prints an exception on failure
# https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/
def create_task(coroutine: Awaitable[T]) -> asyncio.Task[T]:
    def handle_task_result(task):
        try: task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            traceback.print_exc()
    task = asyncio.create_task(coroutine)  # type: ignore
    task.add_done_callback(handle_task_result)
    return task


async def async_map(f, xs, num_threads, worker_done=None):
    q = asyncio.Queue()
    res = [None] * len(xs)
    for (i, x) in enumerate(xs): await q.put((i, x))
    async def task():
        while True:
            try:
                i, x = q.get_nowait()
                y = await f(x)
                res[i] = y
            except asyncio.QueueEmpty:
                break
        if worker_done is not None:
            worker_done()
    tasks = [create_task(task()) for _ in range(num_threads)]
    await asyncio.gather(*tasks)
    return res


#####
## Profiling tools
#####


@dataclass
class InferenceEvent:
    """
    Generated for every GPU call.
    """
    name: str
    cat: str
    start_time: float
    end_time: float


@dataclass
class WorkerEvent:
    name: str
    worker_id: int
    time: float


def profiler_entry(name, cat, ph, tid, ts):
    return {
        'name': name, 'cat': cat, 'ph': ph,
        'pid': 0, 'tid': tid, 'ts': int(1e6 * ts)}


@dataclass
class InferenceStats:
    worker_events: List[WorkerEvent] = field(default_factory=list)
    inference_events: List[InferenceEvent] = field(default_factory=list)

    def dump_profile_file(self, filename: str):
        """
        Create a JSON file to be visualized with chrome://tracing
        """
        entries: List = []
        def entry(*args): entries.append(profiler_entry(*args))
        for we in self.worker_events:
            entry(we.name, we.name, 'i', we.worker_id + 1, we.time)
        for ie in self.inference_events:
            entry(ie.name, ie.cat, 'B', 0, ie.start_time)
            entry(ie.name, ie.cat, 'E', 0, ie.end_time)
        with open(filename, "w") as io:
            json.dump(entries, io)


#####
## High-level utilities
#####


@dataclass
class ServerHandle:
    server: Optional[ActorHandle]
    probe_size: int
    action_size: int
    espec: EventsSpec


def start_inference_server(
    network_file: str,
    network_params: NetworkParams,
    tconf: TensorizerConfig,
    probe_size: int,
    action_size: int,
    espec: EventsSpec,
    max_cuda_memory_fraction: Optional[float],
    num_waited_processes: Optional[int] = None
) -> ServerHandle:
    num_gpus = 1 if torch.cuda.is_available() else 0
    device = 'cuda' if num_gpus > 0 else 'cpu'
    Server = ray.remote(
        num_cpus=1, num_gpus=num_gpus)(InferenceServer)  #type: ignore
    server = Server.remote(num_waited_processes)
    net_weights = torch.load(network_file)
    ray.get(server.load_network.remote(
        net_params=network_params,
        tconf=tconf, agent_spec=espec.agent_spec,
        net_state_dict=net_weights, device=device,
        max_cuda_memory_fraction=max_cuda_memory_fraction))
    return ServerHandle(server, probe_size, action_size, espec)


R = TypeVar('R')


def pmap_with_oracle(
    f: Callable[[T, Oracle], Awaitable[R]],
    xs: Sequence[T],
    server: ServerHandle,
    num_workers: int,
    num_processes: Optional[int] = None
) -> list[R]:
    if num_processes is None:
        num_processes = psutil.cpu_count(logical=False)
    threads_per_process = math.ceil(num_workers / num_processes)
    ProgressActor = ray.remote(num_cpus=1)(ProgressBarActor)  #type: ignore
    progress = ProgressActor.remote()
    num_problems = len(xs)
    pbar = ProgressBar(progress, num_problems)

    @ray.remote(num_cpus=1)  #type: ignore
    def process(
        ixs_subset: Sequence[tuple[int, T]],
        process_id: int
    ) -> Sequence[tuple[int, R]]:
        res = []
        async def single(ix, oracle):
            i, x = ix
            y = await f(x, oracle)
            res.append((i, y))
            ray.get(progress.tick.remote())
        if server.server is not None:
            iter_with_oracle(
                single, ixs_subset, server.server,
                num_threads=threads_per_process,
                process_id=process_id,
                probe_size=server.probe_size,
                action_size=server.action_size)
        else:
            iter_with_dummy_oracle(single, ixs_subset, espec=server.espec)
        return res

    ixs_subsets = np.array_split(
        np.array(list(enumerate(xs)), dtype=object), num_processes)
    procs = [
        process.remote(xs_subset, process_id=i)
        for i, xs_subset in enumerate(ixs_subsets)]
    pbar.print_until_done()
    iyss: Sequence[Sequence[tuple[int, R]]] = ray.get(procs)
    iys = [iy for iys in iyss for iy in iys]
    iys.sort(key=lambda iy: iy[0])
    return [y for _, y in iys]
