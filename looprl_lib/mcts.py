"""
A simple Gumbel-MCTS implementation for Looprl

Optimizations:
  - All children are not necessarily expanded.
  - Additional fpu_red hyperparameter.
  - Track the path to the best successful proof
  - The tree is not reset between each move and there is a
    max_tree_size param.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import softmax  # type: ignore

from .env_wrapper import ChoiceState, FinalState, OutcomeType, StateWrapper
from .params import MctsParams


@dataclass
class Node:
    state: StateWrapper
    params: MctsParams
    total_rewards: float
    num_visits: int
    success_value: Optional[float]
    children: list[Optional['Node']]

    def __init__(self, state: StateWrapper, params: MctsParams) -> None:
        """
        This constructor should not be used directly as it
        does not fully initialize a node.
        """
        self.state = state
        self.params = params

    def reset(self) -> None:
        """
        Reset an MCTS node that has already been properly initialized.
        """
        self.num_visits = 1
        self.success_value = None
        status = self.state.cached_status
        assert status is not None
        if isinstance(status, FinalState):
            final_reward = self.state.final_reward
            self.total_rewards = final_reward
            self.children = []
            if status.outcome_type == OutcomeType.SUCCESS:
                self.success_value = final_reward
        elif isinstance(status, ChoiceState):
            self.total_rewards = status.predicted_value
            self.children = [None] * len(status.oracle_output.policy)
        else:
            assert False

    async def initialize(self) -> None:
        self.num_visits = 1
        await self.state.status()
        self.reset()

    @staticmethod
    async def make(state: StateWrapper, params: MctsParams) -> 'Node':
        node = Node(state, params)
        await node.initialize()
        return node

    @property
    def success(self) -> bool:
        return self.success_value is not None

    @property
    def value(self) -> float:
        return self.total_rewards / self.num_visits

    @property
    def predicted_value(self) -> float:
        status = self.state.cached_status
        assert isinstance(status, ChoiceState)
        return status.predicted_value

    @property
    def policy_prior(self) -> np.ndarray:
        status = self.state.cached_status
        assert isinstance(status, ChoiceState)
        policy = status.oracle_output.policy
        if (eps := self.params.bias_eps) != 0:
            if (bias := self.state.bias_distribution) is not None:
                policy = (1-eps)*policy + eps*bias
        return policy

    def is_final_state(self) -> bool:
        return len(self.children) == 0

    def qvalues(self) -> np.ndarray:
        # Non visited children are assigned a NaN q-value
        qs = [
            c.value if c is not None else math.nan
            for c in self.children]
        return np.array(qs, dtype=np.float32)

    def visited_children_indices(self) -> list[int]:
        return [i for i, c in enumerate(self.children) if c is not None]

    def non_visited_children_indices(self) -> list[int]:
        return [i for i, c in enumerate(self.children) if c is None]

    def children_visits(self) -> np.ndarray:
        return np.array([
            c.num_visits if c is not None else 0
            for c in self.children])

    def best_value_estimate(self) -> float:
        # One when to estimate the value is to just look at
        # total_rewards / num_visits. We propose a more precise estimate here
        # that is less sensitive to exploratory moves.
        status = self.state.cached_status
        assert isinstance(status, ChoiceState)
        oracle_value = status.predicted_value
        visited = self.visited_children_indices()
        if visited:
            prior = status.oracle_output.policy[visited]
            qvalues = self.qvalues()
            children_value = np.sum(prior * qvalues[visited]) / np.sum(prior)
        else:
            children_value = 0.
        num_cvisits = self.num_visits - 1  # num of children visits
        return (oracle_value + num_cvisits * children_value) / (num_cvisits + 1)

    def completed_qvalues(self, fpu_red: bool) -> np.ndarray:
        root_est = self.best_value_estimate()
        if fpu_red:
            root_est -= self.params.fpu_red
        qvalues = self.qvalues()
        qvalues[self.non_visited_children_indices()] = root_est
        return qvalues

    def qcoeff(self) -> float:
        max_visits = max(
            (c.num_visits for c in self.children if c is not None),
            default=0)
        params = self.params
        return params.value_scale * (params.max_visit_init + max_visits)

    def target_policy(self, fpu_red: bool) -> np.ndarray:
        qs = self.completed_qvalues(fpu_red)
        return softmax(np.log(self.policy_prior) + self.qcoeff() * qs)

    def select_action(self, root: bool) -> np.signedinteger:
        policy = self.target_policy(fpu_red=(not root))
        return np.argmax(
            policy - self.children_visits() / (self.num_visits + 1))

    async def run_simulation_from_child(self, child_id: int) -> float:
        status = self.state.cached_status
        assert isinstance(status, ChoiceState)
        child = self.children[child_id]
        if child is None:
            child = await self.expand_child(child_id)
            value = child.value
        else:
            value = await child.run_simulation()
        if child.success_value is not None:
            success_qvalue = child.success_value
            if (self.success_value is None or
                    self.success_value < success_qvalue):
                self.success_value = success_qvalue
        return value

    async def run_simulation(self, root: bool=False) -> float:
        status = self.state.cached_status
        assert status is not None
        if isinstance(status, FinalState):
            value = self.value
        elif isinstance(status, ChoiceState):
            best_child = self.select_action(root=root)
            value = await self.run_simulation_from_child(int(best_child))
        else:
            assert False
        self.total_rewards += value
        self.num_visits += 1
        return value

    async def gumbel_explore(self, rng: np.random.Generator) -> 'GumbelOutput':
        n = len(self.children)
        assert n > 0
        if n == 1:
            # This is possible despite the env_wrapper filter
            # in case some actions are rejected for having big encodings.
            pass
        gscores = rng.gumbel(size=n)
        prior = self.policy_prior
        if (eps := self.params.dirichlet_eps) is not None:
            if (biases := self.state.bias_distribution) is None:
                biases = np.ones(n) / n
            alphas = self.params.dirichlet_alpha * biases
            noise = rng.dirichlet(alphas)
            prior = (1-eps) * prior + eps * noise
        base_scores = gscores + np.log(prior)
        num_considered = min(
            self.params.num_considered_actions, len(self.children))
        considered = np.argsort(-base_scores)[:num_considered]
        # Sorting considered options
        def sort_options(considered):
            qs = self.qvalues()[considered]
            scores = base_scores[considered] + self.qcoeff() * qs
            return considered[np.argsort(-scores)]
        # Sequential halving
        num_prev_sims = 0
        num_halving_steps = math.ceil(math.log2(num_considered))
        num_halving_steps = max(num_halving_steps, 1)  # case where n=1
        sims_per_step = self.params.num_simulations / num_halving_steps
        gsims = np.zeros(len(self.children), dtype=np.int32)
        while True:
            num_visits = max(1, math.floor(sims_per_step / num_considered))
            for _ in range(num_visits):
                # If we do not have enough simulations left to visit
                # every one, then we must visit the most promising
                # actions in priority
                if num_prev_sims + num_considered > self.params.num_simulations:
                    considered = sort_options(considered)
                # We visit all considered actions once
                for i in considered:
                    num_prev_sims += 1
                    gsims[i] += 1
                    # No need to perform the visit if the maximum tree size is
                    # already exceeded and the child is already visited enough
                    if ((ts := self.params.max_tree_size) is None or
                        self.num_visits < ts or self.children[i] is None or
                        self.children[i].num_visits < gsims[i]):
                        value = await self.run_simulation_from_child(i)
                        self.total_rewards += value
                        self.num_visits += 1
                    if num_prev_sims >= self.params.num_simulations:
                        considered = sort_options(considered)
                        return GumbelOutput(gscores, gsims, considered[0])
            # Halving step
            num_considered = max(2, num_considered // 2)
            considered = sort_options(considered)
            considered = considered[:num_considered]

    async def expand_child(self, i: int) -> 'Node':
        new_child = await Node.make(self.state.select(i), self.params)
        self.children[i] = new_child
        return new_child

    async def explore(self, n: Optional[int] = None) -> None:
        if n is None:
            n = self.params.num_simulations
        for _ in range(n):
            if (self.params.max_tree_size is not None and
                self.num_visits > self.params.max_tree_size):
                break
            await self.run_simulation(root=True)

    async def solve(self, timeout: int) -> bool:
        for _ in range(timeout):
            if self.success:
                return True
            await self.run_simulation(root=True)
        return self.success

    def get_node(self, path: list[int]) -> Optional['Node']:
        node = self
        for i in path:
            if 0 <= i and i < len(node.children):
                c = node.children[i]
                if c is None:
                    return None
                node = c
        return node

    def policy(self) -> np.ndarray:
        visits = self.children_visits()
        assert visits.sum() > 0
        return visits / visits.sum()


async def init_mcts(state: StateWrapper, params: MctsParams) -> Node:
    return await Node.make(state, params)


@dataclass
class GumbelOutput:
    gumbel_vars: np.ndarray
    gumbel_visits: np.ndarray
    selected: int
