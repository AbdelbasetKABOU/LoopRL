from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from sys import stderr
from typing import Awaitable, Callable, Generic, Optional, TypeVar

import looprl
import numpy as np
from looprl import CamlRng, Graphable, Prog, SearchTree

from looprl_lib.events import EventsSpec, final_reward, value_prediction

from .params import AgentParams, EncodingParams
from .tensors import ChoiceTensors, tensorize_choice_state


@dataclass
class OracleOutput:
    events: np.ndarray
    policy: np.ndarray

    @staticmethod
    def dummy(spec: EventsSpec, num_actions: int):
        events = spec.default_pred_vec.copy()
        policy = np.ones(num_actions, dtype=np.float32) / num_actions
        return OracleOutput(events, policy)


Oracle = Callable[[ChoiceTensors], Awaitable[OracleOutput]]


@dataclass
class WrapperParams:
    max_proof_len: int
    max_probe_size: int
    max_action_size: int
    skip_singleton_choices: bool
    log_messages: bool
    encoding: EncodingParams
    espec: EventsSpec
    oracle: Oracle
    max_num_actions: int


class OutcomeType(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    EMPTY_CHOICE = auto()
    PROOF_SIZE_LIMIT_EXCEEDED = auto()
    PROBE_SIZE_LIMIT_EXCEEDED = auto()
    ACTION_SIZE_LIMIT_EXCEEDED = auto()


class StateStatus:
    ...


@dataclass
class FinalState(StateStatus):
    outcome_code: int
    outcome_type: OutcomeType


@dataclass
class ChoiceState(StateStatus):
    """
    It is possible to hide some actions.
    The indices of valid actions are collected in [valid_actions].
    The action_prior array has size [len(valid_action)].
    """
    valid_actions: list[int]
    max_action_size: int
    oracle_output: OracleOutput
    predicted_value: float


T = TypeVar("T", covariant=True)


@dataclass
class StateWrapper(Generic[T]):
    _state: SearchTree[T]
    params: WrapperParams
    nsteps: int = 0
    events: list[int] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    _status: Optional[StateStatus] = None

    def __post_init__(self):
        self.reach_next_choice_point()

    def reach_next_choice_point(self) -> None:
        i = 0
        while self.nsteps <= self.params.max_proof_len:
            i += 1
            if i > 1000:
                assert False, "Unable to reach the next choice point."
            if self._state.is_message():
                if self.params.log_messages:
                    self.messages.append(self._state.message())
                self._state = self._state.next()
            elif self._state.is_event():
                event = self._state.event_code()
                self.events.append(event)
                self._state = self._state.next()
            elif (self.params.skip_singleton_choices and
                  self._state.is_choice() and len(self._state.choices()) == 1):
                self.nsteps += 1
                self._state = self._state.select(0)
            else:
                state = self._state
                assert (
                    state.is_choice() or state.is_failure() or
                    state.is_success())
                break

    async def compute_status(self) -> StateStatus:
        if self._state.is_success():
            code = self.params.espec.agent_spec['success_code']
            return FinalState(code, OutcomeType.SUCCESS)
        elif self.nsteps > self.params.max_proof_len:
            code = self.params.espec.agent_spec['size_limit_exceeded_code']
            return FinalState(code, OutcomeType.PROOF_SIZE_LIMIT_EXCEEDED)
        elif self._state.is_failure():
            code = self._state.failure_code()
            return FinalState(code, OutcomeType.FAILURE)
        elif self._state.is_choice() and not self._state.choices():
            code = self.params.espec.agent_spec['default_failure_code']
            return FinalState(code, OutcomeType.EMPTY_CHOICE)
        elif self._state.is_chance():
            # In this case, no need to call the oracle:
            # we generate dummy oracle values
            num_actions = len(self._state.choices())
            valid_actions = list(range(num_actions))
            dummy_out = OracleOutput.dummy(self.params.espec, num_actions)
            predicted_value = value_prediction(
                dummy_out.events, self.params.espec, self.events)
            return ChoiceState(valid_actions, 0, dummy_out, predicted_value)
        else:
            assert self._state.is_choice()
            num_actions = len(self._state.choices())
            if num_actions > self.params.max_num_actions:
                print(f"State with {num_actions} actions", file=stderr)
                print(str(self._state.probe()))
                print("Actions:")
                for a in self._state.choices():
                    print(str(a))
                assert False
            tensors = tensorize_choice_state(self._state, self.params.encoding)
            probe_size = tensors['probe']['nodes'].shape[0]
            if probe_size > self.params.max_probe_size:
                code = self.params.espec.agent_spec['size_limit_exceeded_code']
                return FinalState(
                    code, OutcomeType.PROBE_SIZE_LIMIT_EXCEEDED)
            asizes = [a['nodes'].shape[0] for a in tensors['actions']]
            valid_actions = [
                i for i, s in enumerate(asizes)
                if s <= self.params.max_action_size]
            if not valid_actions:
                code = self.params.espec.agent_spec['size_limit_exceeded_code']
                return FinalState(
                    code, OutcomeType.ACTION_SIZE_LIMIT_EXCEEDED)
            tensors['actions'] = [tensors['actions'][i] for i in valid_actions]
            oracle_output = await self.params.oracle(tensors)
            predicted_value = value_prediction(
                oracle_output.events, self.params.espec, self.events)
            return ChoiceState(
                valid_actions, max(asizes), oracle_output, predicted_value)

    async def status(self) -> StateStatus:
        if self._status is None:
            self._status = await self.compute_status()
        return self._status

    @property
    def cached_status(self):
        return self._status

    @property
    def _valid_actions(self):
        status = self.cached_status
        assert isinstance(status, ChoiceState)
        return status.valid_actions

    @property
    def actions(self) -> list[Graphable]:
        valid = self._valid_actions
        actions = self._state.choices()
        return [actions[i] for i in valid]

    @property
    def success_value(self) -> T:
        return self._state.success_value()

    @property
    def failure_message(self) -> str:
        return self._state.failure_message()

    @property
    def probe(self) -> Graphable:
        return self._state.probe()

    @property
    def probe_size(self) -> int:
        tensors = tensorize_choice_state(self._state, self.params.encoding)
        return tensors['probe']['nodes'].shape[0]

    @property
    def weights(self) -> np.ndarray:
        valid = self._valid_actions
        base_weights = self._state.weights()
        weights = np.array([base_weights[i] for i in valid], dtype=np.float32)
        return weights

    @property
    def bias_distribution(self) -> Optional[np.ndarray]:
        weights = self.weights
        sum_weights = np.sum(weights)
        assert sum_weights >= 0
        if sum_weights == 0:
            return None
        else:
            weights /= sum_weights
            return weights

    @property
    def normalized_weights(self) -> np.ndarray:
        normalized = self.bias_distribution
        assert normalized is not None
        return normalized

    @property
    def is_chance_node(self) -> bool:
        return self._state.is_chance()

    @property
    def final_reward(self) -> float:
        assert isinstance(self._status, FinalState)
        return final_reward(
            self.params.espec.agent_spec,
            self.events, self._status.outcome_code)

    def select(self, i: int) -> 'StateWrapper':
        """
        This function should only be called if status
        returns ChoiceState.
        """
        valid = self._valid_actions
        return StateWrapper(
            self._state.select(valid[i]), self.params, self.nsteps + 1,
            events=self.events.copy(),
            messages=self.messages.copy())


def dummy_oracle(espec: EventsSpec) -> Oracle:
    async def oracle(choice: ChoiceTensors):
        num_actions = len(choice['actions'])
        return OracleOutput.dummy(espec, num_actions)
    return oracle


def init_solver(
    prog: Prog,
    params: AgentParams,
    oracle: Optional[Oracle] = None,
    log_messages: bool = False,
):
    espec = EventsSpec(looprl.solver_spec)
    state = looprl.init_solver(prog)
    if oracle is None:
        oracle = dummy_oracle(espec)
    wrapper_params = WrapperParams(
        max_proof_len=params.search.max_proof_length,
        max_probe_size=params.search.max_probe_size,
        max_action_size=params.search.max_action_size,
        skip_singleton_choices=True,
        log_messages=log_messages,
        espec=espec,
        encoding=params.encoding,
        oracle=oracle,
        max_num_actions=14)
    return StateWrapper(state, wrapper_params)


def init_teacher(
    params: AgentParams,
    rng: CamlRng,
    oracle: Optional[Oracle] = None,
    log_messages: bool = False,
    spec_sexp: Optional[str] = None
):
    espec = EventsSpec(looprl.teacher_spec)
    if spec_sexp is None:
        state = looprl.init_teacher(rng)
    else:
        state = looprl.init_teacher_with_spec(rng, spec_sexp)
    if oracle is None:
        oracle = dummy_oracle(espec)
    wrapper_params = WrapperParams(
        max_proof_len=params.search.max_proof_length,
        max_probe_size=params.search.max_probe_size,
        max_action_size=params.search.max_action_size,
        skip_singleton_choices=True,
        log_messages=log_messages,
        espec=espec,
        encoding=params.encoding,
        oracle=oracle,
        max_num_actions=14)
    return StateWrapper(state, wrapper_params)
