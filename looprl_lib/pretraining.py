"""
A simple arithmetic task used for pretraining.
"""

from dataclasses import dataclass, field
from typing import Iterator

import looprl
import numpy as np
import psutil  # type: ignore
import torch
from looprl import AgentSpec
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm  # type: ignore

from looprl_lib.models import num_value_targets
from looprl_lib.net_util import make_network
from looprl_lib.params import EncodingParams, NetworkParams
from looprl_lib.samples import (Sample, SamplesBatch,
                                convert_and_collate_samples, tensorize_sample,
                                to_device)

# A runtime error will be raised if a pretraining sample
# does not respect those limits.
MAX_PROBE_SIZE = 20
MAX_ACTION_SIZE = 10


@dataclass
class PretrainingDataset(IterableDataset):
    size: int
    encoding: EncodingParams
    agent_spec: AgentSpec
    true_false: bool
    randomize_uids: bool
    keep_graphable: bool = False
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __iter__(self) -> Iterator[SamplesBatch]:
        info = get_worker_info()
        n = self.size // info.num_workers
        if info.id == 0:
            n += self.size % info.num_workers
        rng = looprl.CamlRng()
        sampler = looprl.pretraining_tasks_sampler(
            rng, true_false=self.true_false)
        for i in range(n):
            assums, concl, non_concl = sampler()
            sample = Sample(
                probe=assums,
                actions=[concl, non_concl],
                value_target=([0.] * num_value_targets(self.agent_spec)),
                policy_target=[1., 0.],
                problem_id=i)
            tensorized = tensorize_sample(
                sample, self.encoding,
                probe_size=MAX_PROBE_SIZE,
                action_size=MAX_ACTION_SIZE,
                randomize_uids=self.randomize_uids,
                keep_graphable=self.keep_graphable,
                rng=self.rng)
            yield tensorized

    def __len__(self) -> int:
        return self.size


def eval_accuracy_and_analyze_mistakes(
    net_file: str,
    net_params: NetworkParams,
    encoding: EncodingParams,
    agent_spec: AgentSpec,
    true_false: bool,
    randomize_uids: bool,
    num_samples: int = 1024 * 50,
    batch_size: int = 1024,
    mistake_threshold: float = 0.1,
    max_num_mistakes: int = 100
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = psutil.cpu_count(logical=True)
    net = make_network(net_params, encoding.tensorizer_config, agent_spec)
    net.load_state_dict(torch.load(net_file))
    dataset = PretrainingDataset(
        size=num_samples,
        encoding=encoding,
        agent_spec=agent_spec,
        true_false=true_false,
        randomize_uids=randomize_uids,
        keep_graphable=True)
    data = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        collate_fn=convert_and_collate_samples)
    net.train(mode=False)
    net.to(device=device)
    acc_hist: list[float] = []
    mistakes: list[tuple[float, dict]] = []
    with torch.no_grad():
        for batch in tqdm(data):
            batch = to_device(batch, device)
            _, P = net(batch['choice'])
            p_correct = P[0::2]
            acc = (p_correct >= 0.5).sum().item() / len(p_correct)
            acc_hist.append(acc)
            if len(mistakes) < max_num_mistakes:
                idx = torch.nonzero(p_correct < mistake_threshold)
                for i in idx.cpu():
                    mistakes.append(
                        (p_correct[i].item(), batch['graphable'][i]))
    print(f"Accuracy: {np.mean(acc_hist):.3f}")
    print("")
    print("Mistakes Examples:")
    for p, g in mistakes:
        probe = g['probe']
        concl = g['actions'][0]
        non_concl = g['actions'][1]
        print(f"  [{p:.2f}] {probe:40s} {concl:15s} {non_concl:15s}")
