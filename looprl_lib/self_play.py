from dataclasses import dataclass
from sys import stderr
from typing import Any, Iterable, Optional

import numpy as np
from looprl import AgentSpec  # type: ignore
from scipy.stats import entropy  # type: ignore

from looprl_lib.env_wrapper import (ChoiceState, FinalState, OutcomeType,
                                    StateWrapper, T)
from looprl_lib.events import (EventsSpec, event_and_outcomes_counts_dict,
                               final_reward, pred_target)
from looprl_lib.mcts import Node
from looprl_lib.params import MctsParams
from looprl_lib.samples import Sample


@dataclass
class TraceElement:
    state: StateWrapper
    policy_target: np.ndarray


@dataclass
class Trace:
    outcome: int
    events: list[int]
    choices: list[TraceElement]


async def self_play_episode(
    state: StateWrapper[T],
    mcts_params: MctsParams,
    rng: np.random.Generator,
    gumbel_exploration: bool
) -> tuple[Trace, Optional[T]]:
    choices: list[TraceElement] = []
    tree: Optional[Node] = None
    while True:
        status = await state.status()
        if isinstance(status, FinalState):
            final = (
                state.success_value
                if status.outcome_type == OutcomeType.SUCCESS else None)
            return (Trace(status.outcome_code, state.events, choices), final)
        elif isinstance(status, ChoiceState):
            if tree is None:
                tree = await Node.make(state, mcts_params)
            if gumbel_exploration:
                gumbel = await tree.gumbel_explore(rng)
                action = gumbel.selected
                target_policy = tree.target_policy(fpu_red=False)
            else:
                await tree.explore()
                target_policy = tree.target_policy(fpu_red=False)
                action = int(np.argmax(target_policy))
            choices.append(TraceElement(state, target_policy))
            child_tree = tree.children[action]
            if child_tree is None:
                # This should not happen as we never use fpu_red!=0
                # at the root. Still, let's try and be sage in case
                # some numerical error happens.
                print("An unvisited MCTS child was selected.", file=stderr)
                state = state.select(action)
                tree = None
            else:
                state = child_tree.state
                tree = child_tree
            if mcts_params.reset_tree:
                tree = None


def generate_samples(
    trace: Trace,
    problem_id: Optional[int],
    espec: EventsSpec,
) -> Iterable[Sample]:
    for e in trace.choices:
        probe = e.state.probe
        actions = e.state.actions
        value_target = pred_target(espec, trace.outcome, trace.events)
        policy_target = e.policy_target.tolist()
        yield Sample(
            probe, actions, value_target, policy_target, problem_id)


def trace_summary_stats(trace: Trace, spec: AgentSpec) -> dict[str, float]:
    res: dict[str, float] = {}
    res['rewards'] = final_reward(spec, trace.events, trace.outcome)
    res['trace-length'] = float(len(trace.choices))
    if trace.choices:
        res['target-entropy'] = np.mean([
            entropy(e.policy_target, base=2) for e in trace.choices])
    else:
        res['target-entropy'] = 0.0
    counts = event_and_outcomes_counts_dict(trace.outcome, trace.events, spec)
    res.update({e: float(c) for e, c in counts.items()})
    return res


def aggregate_trace_summaries(
    summaries: list[dict[str, float]]
) -> dict[str, float]:
    assert summaries
    return {k: np.mean([r[k] for r in summaries]) for k in summaries[0].keys()}
