"""
We want ray to manipulate numpy arrays because these
can be serialized at no cost by ray.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Sequence, TypedDict, TypeVar, Union

import looprl
import numpy as np
import torch
from looprl import (Graphable, GraphTensors, SearchTree, TensorizerConfig,
                    UidMap)

from looprl_lib.params import EncodingParams


class ChoiceTensors(TypedDict):
    """
    All tensors that encode a choice point.
    """
    probe: GraphTensors
    actions: list[GraphTensors]


T = TypeVar('T', bound=Union[torch.Tensor, np.ndarray])


@dataclass
class GraphsBatch(Generic[T]):
    """
    A representation of a batch of graphable objects.

    Tensor fields:
        - nodes: shape (batch_size, num_toks, num_features)
        - edges: shape (num_edges, 4), contains (batch, src, dst, typ) tuples
        - pos_emb: shape (batch_size, num_toks, d_model)
        - mask: shape (batch_size, num_toks, num_toks); it is
            currently only used for padding: token i cannot attend token j
            iff mask[i,j]=0. in practice, we use shape (batch_size, 1, num_toks)
            and rely on broadcasting.
    """
    nodes: T
    edges: T
    pos_embs: T
    mask: T

    @property
    def batch_size(self) -> int:
        return self.nodes.shape[0]

    @staticmethod
    def make_empty(
        num_toks: int,
        num_features: int,
        d_model: int
    ) -> 'GraphsBatch[np.ndarray]':
        nodes = np.zeros((0, num_toks, num_features), dtype=np.float32)
        edges = np.zeros((0, 4), dtype=np.int64)
        pos_embs = np.zeros((0, num_toks, d_model), dtype=np.float32)
        mask = np.zeros((0, 1, num_toks), dtype=np.float32)
        return GraphsBatch(nodes, edges, pos_embs, mask)

    @staticmethod
    def make(
        tensors: Sequence[GraphTensors],
        num_toks: Optional[int] = None,
        allow_empty: bool = False,
        tconf: Optional[TensorizerConfig] = None
    ) -> 'GraphsBatch[np.ndarray]':
        """
        Specifying the number of features is only useful when
        `tensors` is an empty sequence.
        """
        if not tensors:
            assert allow_empty and num_toks is not None and tconf is not None
            num_features = looprl.token_encoding_size(tconf)
            d_model = tconf['d_model']
            return GraphsBatch.make_empty(num_toks, num_features, d_model)
        if num_toks is None:
            num_toks = max(t['nodes'].shape[0] for t in tensors)
        nodes = np.stack(
            [pad_to_height(t['nodes'], num_toks) for t in tensors])
        pos_embs = np.stack(
            [pad_to_height(t['pos_emb'], num_toks) for t in tensors])
        edges = np.concatenate([
            append_const_on_left(t['edges'], i)
            for (i, t) in enumerate(tensors)])
        mask = np.stack([
            one_then_zero(t['nodes'].shape[0], num_toks)
            for t in tensors])
        mask = mask[:, np.newaxis, :]
        return GraphsBatch(nodes, edges, pos_embs, mask)

    @staticmethod
    def concatenate(
        batches: Sequence['GraphsBatch[np.ndarray]']
    ) -> 'GraphsBatch[np.ndarray]':
        nodes = np.concatenate([b.nodes for b in batches], axis=0)
        pos_embs = np.concatenate([b.pos_embs for b in batches], axis=0)
        mask = np.concatenate([b.mask for b in batches], axis=0)
        # We only have heavy work to do for edges
        edges_list = []
        offset = 0
        for b in batches:
            e = b.edges.copy()
            e[:,0] += offset
            edges_list.append(e)
            offset += b.batch_size
        edges = np.concatenate(edges_list, axis=0)
        return GraphsBatch(nodes, edges, pos_embs, mask)


@dataclass
class ChoicesBatch(Generic[T]):
    """
    A representation of a batch of choices.
    The 'batch' field has shape (num_actions,) and 'batch[i]' denotes
    the probe id associated with the ith action.
    """
    probes: GraphsBatch[T]
    actions: GraphsBatch[T]
    batch: T
    num_actions: np.ndarray

    @property
    def batch_size(self) -> int:
        return self.probes.batch_size

    @staticmethod
    def make(
        choices: Sequence[ChoiceTensors],
        num_probe_toks: Optional[int] = None,
        num_action_toks: Optional[int] = None,
        allow_empty: bool = False,
        tconf: Optional[TensorizerConfig] = None
    ) -> 'ChoicesBatch[np.ndarray]':
        probes = GraphsBatch.make(
            [c['probe'] for c in choices],
            num_toks=num_probe_toks, allow_empty=allow_empty, tconf=tconf)
        actions = GraphsBatch.make(
            [a for c in choices for a in c['actions']],
            num_toks=num_action_toks, allow_empty=allow_empty, tconf=tconf)
        batch = []
        for (i, c) in enumerate(choices):
            batch += [i] * len(c['actions'])
        np_batch = np.array(batch, dtype=np.int64)
        num_actions = np.array(
            [len(c['actions']) for c in choices], dtype=np.int64)
        return ChoicesBatch(probes, actions, np_batch, num_actions)

    @staticmethod
    def concatenate(
        batches: Sequence['ChoicesBatch[np.ndarray]']
    ) -> 'ChoicesBatch[np.ndarray]':
        probes = GraphsBatch.concatenate([b.probes for b in batches])
        actions = GraphsBatch.concatenate([b.actions for b in batches])
        batch_list = []
        offset = 0
        for b in batches:
            batch_list.append(b.batch + offset)
            offset += b.batch_size
        batch = np.concatenate(batch_list, axis=0)
        num_actions = np.concatenate([b.num_actions for b in batches], axis=0)
        return ChoicesBatch(probes, actions, batch, num_actions)

    def to(self, *, device: str):
        return to_device(self, device)  # type: ignore

    def derive_num_actions(self) -> list[int]:
        """
        Return a list of size self.batch_size indicating
        the number of actions for each batch element.
        This is too slow so we cache this information instead
        """
        res: list[int] = []
        i = 0
        n = len(self.batch)
        for b in range(self.batch_size):
            d = 0
            while i < n and self.batch[i] == b:
                i += 1
                d += 1
            res.append(d)
        assert sum(res) == n
        return res


#####
## Utilities
#####


def tensorize_choice(
    probe: Graphable,
    actions: Sequence[Graphable],
    config: EncodingParams
) -> ChoiceTensors:
    tensorizer = config.tensorizer_config
    tokenizer = config.tokenizer_config
    probe_tensors, uids = probe.tensorize(tensorizer, tokenizer, UidMap())
    action_tensors = []
    for a in actions:
        tensors, _ = a.tensorize(tensorizer, tokenizer, uids)
        action_tensors.append(tensors)
    return {'probe': probe_tensors, 'actions': action_tensors}


def tensorize_choice_state(
    st: SearchTree,
    config: EncodingParams
) -> ChoiceTensors:
    assert st.is_choice()
    return tensorize_choice(st.probe(), st.choices(), config)


def pad_to_height(t: np.ndarray, n: int) -> np.ndarray:
    """
    Pad a 2D tensor with zeros.
    """
    if n - t.shape[0] < 0:
        assert False, f"Trying to pad {t.shape[0] } tokens to {n}."
    padding = np.zeros((n - t.shape[0], t.shape[1]), dtype=t.dtype)
    return np.concatenate((t, padding))


def append_const_on_left(t: np.ndarray, c) -> np.ndarray:
    return np.concatenate((
        np.array(c, t.dtype).repeat(t.shape[0])[:,np.newaxis],
        t), axis=1)


def one_then_zero(num_ones: int, len: int):
    return np.concatenate(
        (np.ones(num_ones, np.int64),
        np.zeros(len-num_ones, np.int64)))


#####
## Torch utilities
#####


def map_graphs_batch(f, b: GraphsBatch) -> GraphsBatch:
    return GraphsBatch(
        f(b.nodes),
        f(b.edges),
        f(b.pos_embs),
        f(b.mask))


def map_choices_batch(f, b: ChoicesBatch) -> ChoicesBatch:
    return ChoicesBatch(
        map_graphs_batch(f, b.probes),
        map_graphs_batch(f, b.actions),
        f(b.batch),
        b.num_actions)


def to_torch(
    b: ChoicesBatch[np.ndarray]
) -> ChoicesBatch[torch.Tensor]:
    return map_choices_batch(torch.from_numpy, b)


def to_device(
    b: ChoicesBatch[torch.Tensor],
    device:str
) -> ChoicesBatch[torch.Tensor]:
    return map_choices_batch(lambda t: t.to(device=device), b)


#####
## Numpy utilities
#####


def copy_graph_tensors(ts: GraphTensors) -> GraphTensors:
    """
    The GraphTensors returned by the OCaml Looprl library
    have arrays of type ocaml_bigarray. This works most of the time as
    ocaml_bigarray is a subtype of np.ndarray but there are cases in which
    we want a real np.ndarray object (for example when using ray).
    """
    return {
        'nodes': np.copy(ts['nodes']),
        'edges': np.copy(ts['edges']),
        'pos_emb': np.copy(ts['pos_emb'])}


def copy_choice_tensors(c: ChoiceTensors) -> ChoiceTensors:
    return {
        'probe': copy_graph_tensors(c['probe']),
        'actions': [copy_graph_tensors(a) for a in c['actions']] }


def unnest_choices_batch(c: ChoicesBatch) -> dict[str, np.ndarray]:
    return {
        'probes_nodes': np.copy(c.probes.nodes),
        'probes_edges': np.copy(c.probes.edges),
        'probes_pos_embs': np.copy(c.probes.pos_embs),
        'probes_pos_mask': np.copy(c.probes.mask),
        'actions_nodes': np.copy(c.actions.nodes),
        'actions_edges': np.copy(c.actions.edges),
        'actions_pos_embs': np.copy(c.actions.pos_embs),
        'actions_mask': np.copy(c.actions.mask),
        'batch': np.copy(c.batch),
        'num_actions': np.copy(c.num_actions) }


def nest_choices_batch(d: dict[str, np.ndarray]) -> ChoicesBatch:
    return ChoicesBatch(
        GraphsBatch(
            d['probes_nodes'], d['probes_edges'],
            d['probes_pos_embs'], d['probes_pos_mask']),
        GraphsBatch(
            d['actions_nodes'], d['actions_edges'],
            d['actions_pos_embs'], d['actions_mask']),
        d['batch'],
        d['num_actions'])


def summarize_batch(batch: ChoicesBatch):
    for k, v in unnest_choices_batch(batch).items():
        print(f"{k}: {v.shape}")


#####
## Shuffle UIDs
#####


def shuffle_uids(
    choice: ChoiceTensors,
    config: TensorizerConfig,
    rng: np.random.Generator,
    pshuffle: float = 1.0,
) -> ChoiceTensors:
    if rng.random() > pshuffle:
        return choice
    # The last uid has a special meaning as several variables
    # can share it in case there aren't enough uids available.
    # num_uids = config['uid_emb_size'] - 1
    num_uids = 3  # TODO: change this
    assert num_uids >= 0
    offset = looprl.uid_encoding_offset(config)
    perm = np.random.permutation(num_uids)
    def shuffle(nodes: np.ndarray):
        nodes = nodes.copy()
        uids_enc = nodes[:, offset:offset+num_uids]
        assert np.all(uids_enc.sum(axis=1) <= 1)
        nodes[:, offset:offset+num_uids] = uids_enc[:, perm]
        assert np.all(nodes[:, offset:offset+num_uids].sum(axis=1) <= 1)
        return nodes
    probe = choice['probe'].copy()
    actions = [a.copy() for a in choice['actions']]
    probe['nodes'] = shuffle(probe['nodes'])
    for a in actions:
        a['nodes'] = shuffle(a['nodes'])
    return {'probe': probe, 'actions': actions}
