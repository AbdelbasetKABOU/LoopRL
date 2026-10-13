"""
Defining the neural networks used in the Looprl agent.
"""

import math
from copy import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from looprl import AgentSpec, TensorizerConfig, token_encoding_size
from torch import Tensor
from torch_scatter import scatter_softmax  # type: ignore

from .dgt import (AttentionParams, TransformerEncoder, TransformerParams,
                  xavier_reset)
from .tensors import ChoicesBatch, GraphsBatch


class Encoder(nn.Module):

    def __init__(
        self,
        params: TransformerParams,
        tconf: TensorizerConfig,
        ignore_pos_encoding: bool
    ):
        super(Encoder, self).__init__()
        assert tconf['d_model'] == params.att.hidden_dim
        self.hyper = params
        self.tconf = tconf
        self.dgt = TransformerEncoder(params)
        self.ignore_pos_encoding = ignore_pos_encoding
        self.emb = nn.Parameter(torch.Tensor(
            tconf['d_model'], token_encoding_size(tconf)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.dgt.reset_parameters()
        nn.init.normal_(self.emb)

    def forward(self, batch: GraphsBatch) -> Tensor:
        init = F.linear(batch.nodes, self.emb)
        # We choose not to apply dropout to the tree positional embeddings
        if not self.ignore_pos_encoding:
            init = init + batch.pos_embs
        return self.dgt(init, batch.edges, batch.mask)


#####
## Looprl Network
#####


@dataclass
class LooprlNetworkParams:
    att: AttentionParams
    probe_encoder_layers: int
    action_encoder_layers: int
    combiner_layers: int
    ff_dim: int
    value_head_num_layers: int
    value_head_input_dim: int
    policy_head_num_layers: int
    policy_head_input_dim: int
    dropout_rate: float
    ignore_pos_encoding: bool

    def subnet_params(
        self, num_layers: int,
        force_ignore_edges: bool = False
    ) -> TransformerParams:
        att = copy(self.att)
        if force_ignore_edges:
            att.ignore_edges = True
        return TransformerParams(
            att, num_layers, self.ff_dim, self.dropout_rate)


class FeedForward(nn.Module):

    def __init__(self,
        num_layers: int, dim: int, out_dim: int, dropout: float
    ):
        super(FeedForward, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers-1):
            self.layers += [
                nn.Linear(dim, dim, bias=True),
                nn.Dropout(p=dropout),
                nn.ReLU()]
        self.layers.append(nn.Linear(dim, out_dim, bias=True))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Pooling(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(Pooling, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.max(self.lin(x), dim=1)[0]


class LooprlNetwork(nn.Module):
    """
    The network used by Looprl.
    It takes as an input a batch of choices and returns:
        - A value vector of shape (num_probes,)
        - An action score vector of shape (num_actions,)
    """

    @staticmethod
    def subnet_params(
        p: LooprlNetworkParams,
        num_layers: int
    ) -> TransformerParams:
        return TransformerParams(p.att, num_layers, p.ff_dim, p.dropout_rate)

    def __init__(self,
        params: LooprlNetworkParams,
        tconf: TensorizerConfig,
        agent_spec: AgentSpec
    ):
        super(LooprlNetwork, self).__init__()
        d_model = tconf['d_model']
        self.probe_encoder = Encoder(
            params.subnet_params(params.probe_encoder_layers), tconf,
            params.ignore_pos_encoding)
        self.action_encoder = Encoder(
            params.subnet_params(params.action_encoder_layers), tconf,
            params.ignore_pos_encoding)
        self.combiner = TransformerEncoder(
            params.subnet_params(
                params.combiner_layers, force_ignore_edges=True))
        # Register 'vgroups' so that it is moved to GPU
        self.vgroups: Tensor
        self.register_buffer(
            "vgroups", value_head_prediction_groups(agent_spec),
            persistent=False)
        self.value_pooling = Pooling(d_model, params.value_head_input_dim)
        self.value_head = FeedForward(
            num_layers=params.value_head_num_layers,
            dim=params.value_head_input_dim,
            out_dim=len(self.vgroups),
            dropout=params.dropout_rate)
        self.policy_pooling = Pooling(d_model, params.policy_head_input_dim)
        self.policy_head = FeedForward(
            num_layers=params.policy_head_num_layers,
            dim=params.policy_head_input_dim,
            out_dim=1,
            dropout=params.dropout_rate)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.probe_encoder.reset_parameters()
        self.action_encoder.reset_parameters()
        self.combiner.reset_parameters()
        for l in [
            self.value_head, self.policy_head,
            self.value_pooling, self.policy_pooling]:
            xavier_reset(l)

    def forward(self, batch: ChoicesBatch[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Return an annotation tensor of shape (max_num_tokens, num_annots).
        """
        probes = self.probe_encoder(batch.probes)
        # Value prediction
        values = self.value_head(self.value_pooling(probes))  # (batch_size, npreds)
        values = scatter_softmax(values, self.vgroups)
        # Policy prediction
        actions = self.action_encoder(batch.actions)
        combiner_inp = torch.concat((probes[batch.batch], actions), dim=1)
        combiner_mask = torch.concat((
            batch.probes.mask[batch.batch],
            batch.actions.mask), dim=-1)
        combiner_edges = torch.tensor([])
        out = self.combiner(combiner_inp, combiner_edges, combiner_mask)
        action_scores = self.policy_head(self.policy_pooling(out))
        action_scores = scatter_softmax(action_scores.squeeze(-1), batch.batch)
        return values, action_scores

    def get_device(self):
        return next(self.parameters()).device


#####
## Structure of the value head output
#####

def value_head_prediction_groups(spec: AgentSpec) -> Tensor:
    groups = [0] * len(spec['outcome_rewards'])
    for i, m in enumerate(spec['event_max_occurences']):
        assert m > 0
        groups += [i+1] * (m + 1)
    return torch.tensor(groups, dtype=torch.long)


def num_value_targets(spec: AgentSpec) -> int:
    return len(value_head_prediction_groups(spec))


#####
## References
#####


# http://jalammar.github.io/illustrated-bert/
