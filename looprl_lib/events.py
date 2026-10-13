from dataclasses import dataclass

import numpy as np
from looprl import AgentSpec


@dataclass
class EventsSpec:
    """
    Wrapper around AgentSpec that caches some additional
    info for computing event rewards and values.
    """
    agent_spec: AgentSpec
    outcome_rewards_vec: np.ndarray
    event_rewards_vec: np.ndarray
    default_pred_vec: np.ndarray
    event_offsets: list[int]
    pred_vec_length: int
    min_anticipated_final_reward: float

    def __init__(self, agent_spec: AgentSpec):
        self.agent_spec = agent_spec
        self.event_offsets = compute_event_offsets(agent_spec)
        self.event_rewards_vec = compute_event_rewards_vec(agent_spec)
        self.outcome_rewards_vec = compute_outcome_rewards_vec(agent_spec)
        success_val = agent_spec['outcome_rewards'][agent_spec['success_code']]
        self.min_anticipated_final_reward = (
            agent_spec['min_success_reward'] - success_val)
        self.default_pred_vec = compute_default_event_prediction_vec(agent_spec)
        self.pred_vec_length = compute_pred_vec_length(agent_spec)


def event_counts(spec: AgentSpec, events: list[int]) -> np.ndarray:
    max_occs = spec['event_max_occurences']
    counts = np.zeros(len(max_occs), dtype=np.int32)
    for e in events:
        counts[e] = min(counts[e] + 1, max_occs[e])
    return counts


def final_reward(spec: AgentSpec, events: list[int], outcome: int) -> float:
    r = spec['outcome_rewards'][outcome]
    if outcome != spec['success_code']:
        return r
    else:
        counts = event_counts(spec, events)
        for e, c in enumerate(counts):
            r += c * spec['event_rewards'][e]
        return max(r, spec['min_success_reward'])


def compute_outcome_rewards_vec(spec: AgentSpec) -> np.ndarray:
    return np.array(spec['outcome_rewards'], dtype=np.float32)


def compute_event_rewards_vec(spec: AgentSpec) -> np.ndarray:
    rs: list[float] = []
    for r, m in zip(spec['event_rewards'], spec['event_max_occurences']):
        rs += [r * i for i in range(m + 1)]
    return np.array(rs, dtype=np.float32)


def compute_event_offsets(spec: AgentSpec) -> list[int]:
    offsets = [0]
    for m in spec['event_max_occurences']:
        offsets.append(offsets[-1] + m + 1)
    return offsets[:-1]


def num_outcomes(spec: AgentSpec) -> int:
    return len(spec['outcome_rewards'])


def compute_pred_vec_length(spec: AgentSpec) -> int:
    return num_outcomes(spec) + len(compute_event_rewards_vec(spec))


def compute_default_event_prediction_vec(spec: AgentSpec) -> np.ndarray:
    n = compute_pred_vec_length(spec)
    nout = num_outcomes(spec)
    pred = np.zeros(n, dtype=np.float32)
    pred[spec['success_code']] = 0.5
    pred[spec['default_failure_code']] = 0.5
    for offset in compute_event_offsets(spec):
        pred[nout + offset] = 1
    return pred


def value_prediction(
    preds: np.ndarray,
    spec: EventsSpec,
    events: list[int],
    eps: float = 1e-10
) -> float:
    nout = num_outcomes(spec.agent_spec)
    opreds = preds[:nout].copy()
    epreds = preds[nout:].copy()
    counts = event_counts(spec.agent_spec, events)
    for e, c in enumerate(counts):
        if c > 0:
            offset = spec.event_offsets[e]
            max_occ = spec.agent_spec['event_max_occurences'][e]
            epreds[offset:offset+c] = 0
            rest = epreds[offset+c:offset+max_occ+1]
            # If rest is all zero, all weights will be concentrated on
            # the current count
            rest[0] = rest[0] + eps
            rest /= np.sum(rest)
            epreds[offset+c:offset+max_occ+1] = rest
    psuccess = preds[spec.agent_spec['success_code']]
    value = np.dot(opreds, spec.outcome_rewards_vec)
    mafr = spec.min_anticipated_final_reward
    value += psuccess * max(mafr, np.dot(epreds, spec.event_rewards_vec))
    return value


def pred_target(
    espec: EventsSpec,
    outcome: int,
    events: list[int],
) -> list[float]:
    aspec = espec.agent_spec
    nout = num_outcomes(aspec)
    counts = event_counts(aspec, events)
    target = [0.] * espec.pred_vec_length
    target[outcome] = 1.
    for count, offset in zip(counts, espec.event_offsets):
        target[offset+nout+count] = 1.
    return target


def event_counts_dict(events: list[int], spec: AgentSpec) -> dict[str, int]:
    counts = event_counts(spec, events)
    return dict(zip(spec['event_names'], counts))


def event_and_outcomes_counts_dict(
    outcome: int,
    events: list[int],
    spec: AgentSpec
) -> dict[str, int]:
    res = event_counts_dict(events, spec)
    for i, oname in enumerate(spec['outcome_names']):
        res[oname] = int(i == outcome)
    return res
