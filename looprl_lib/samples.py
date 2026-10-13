import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, Union

import looprl
import numpy as np
import torch
from looprl import Graphable
from torch import Tensor
from torch.utils.data import Dataset

from looprl_lib.disk_recorder import indexed_elements_in_dir
from looprl_lib.params import EncodingParams

from .tensors import ChoicesBatch, shuffle_uids, tensorize_choice, to_torch


@dataclass
class Sample:
    probe: Graphable
    actions: list[Graphable]
    value_target: list[float]
    policy_target: list[float]
    problem_id: Optional[int]


T = TypeVar('T', bound=Union[torch.Tensor, np.ndarray])


@dataclass
class SamplesBatch(Generic[T]):
    choice: ChoicesBatch
    value_target: T  # 2D tensor
    policy_target: T  # 1D tensor
    extra: list[dict[str, Any]]


def tensorize_sample(
    sample: Sample,
    encoding: EncodingParams,
    probe_size: int,
    action_size: int,
    randomize_uids: bool = False,
    keep_graphable: bool = False,
    rng: Optional[np.random.Generator] = None
) -> SamplesBatch[np.ndarray]:
    choice = tensorize_choice(sample.probe, sample.actions, encoding)
    tconf = encoding.tensorizer_config
    if randomize_uids:
        assert rng is not None
        choice = shuffle_uids(choice, tconf, rng, pshuffle=0.5)
    choice_padded = ChoicesBatch.make(
        [choice], probe_size, action_size, allow_empty=True, tconf=tconf)
    extra: dict[str, Any] = {}
    if keep_graphable:
        extra['graphable'] = {
            'probe': str(sample.probe),
            'actions': [str(a) for a in sample.actions]}
    value_target = np.array(sample.value_target, dtype=np.float32)
    value_target = value_target.reshape(1, -1)
    policy_target = np.array(sample.policy_target, dtype=np.float32)
    return SamplesBatch(choice_padded, value_target, policy_target, [extra])


def convert_and_collate_samples(
    samples: Sequence[SamplesBatch[np.ndarray]]
) -> SamplesBatch[torch.Tensor]:
    """
    Convert all samples to torch tensors and collate them
    into a single batch.
    """
    choice = to_torch(ChoicesBatch.concatenate([s.choice for s in samples]))
    value_target = torch.cat(
        [torch.from_numpy(s.value_target) for s in samples], dim=0)
    policy_target = torch.cat(
        [torch.from_numpy(s.policy_target) for s in samples], dim=0)
    extra = [e for s in samples for e in s.extra]
    return SamplesBatch(choice, value_target, policy_target, extra)


#####
## Utilities for saving and loading samples
#####


@dataclass
class Unserializer:
    probe: Callable[[str], looprl.Graphable]
    action: Callable[[str], looprl.Graphable]


# Introducing explicit functions is important.
# Otherwise, the unserializer would capture a PyCapsule
# that is not serializable by ray.
def unserialize_teacher_probe(sexp):
    return looprl.unserialize_teacher_probe(sexp)
def unserialize_teacher_action(sexp):
    return looprl.unserialize_teacher_action(sexp)
def unserialize_solver_probe(sexp):
    return looprl.unserialize_solver_probe(sexp)
def unserialize_solver_action(sexp):
    return looprl.unserialize_solver_action(sexp)


TEACHER_UNSERIALIZER = Unserializer(
    unserialize_teacher_probe,
    unserialize_teacher_action)


SOLVER_UNSERIALIZER = Unserializer(
    unserialize_solver_probe,
    unserialize_solver_action)


def sample_to_string(s: Sample) -> str:
    return json.dumps({
        'probe': s.probe.serialize(),
        'actions': [a.serialize() for a in s.actions],
        'problem_id': s.problem_id,
        'value_target': s.value_target,
        'policy_target': s.policy_target})


def load_sample(unserialize: Unserializer, path: str) -> Sample:
    with open(path, 'r') as f:
        sample = json.load(f)
    return Sample(
        probe=unserialize.probe(sample['probe']),
        actions=[unserialize.action(a) for a in sample['actions']],
        problem_id=sample['problem_id'],
        value_target=sample['value_target'],
        policy_target=sample['policy_target'])


#####
## Datasets
#####


@dataclass
class SamplesDataset(Dataset):
    unserialize: Unserializer
    encoding: EncodingParams
    probe_size: int
    action_size: int
    randomize_uids: bool
    dir: str

    def __post_init__(self):
        self.sample_ids = indexed_elements_in_dir(self.dir)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, i: int) -> SamplesBatch:
        sid = self.sample_ids[i]
        sample = load_sample(
            self.unserialize, os.path.join(self.dir, str(sid)))
        tensorized = tensorize_sample(
            sample, self.encoding, self.probe_size, self.action_size,
            randomize_uids=self.randomize_uids)
        return tensorized


def to_device(batch: SamplesBatch[Tensor], device: str) -> SamplesBatch[Tensor]:
    return SamplesBatch(
        choice=batch.choice.to(device=device),
        value_target=batch.value_target.to(device=device),
        policy_target=batch.policy_target.to(device=device),
        extra=batch.extra)
