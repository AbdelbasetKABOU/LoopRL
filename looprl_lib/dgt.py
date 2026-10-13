"""
Implementation of a Dynamic Graph Transformer Encoder (Pre-LN variant).

References:
  - GREAT model: http://vhellendoorn.github.io/PDF/iclr2020.pdf
  - Dynamic Graph Transformer: https://arxiv.org/pdf/2012.11401.pdf
"""


import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter  # type: ignore
from torch import Tensor


class AttentionParams:

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_edge_types: int,
        ignore_edges: bool = False,
        att_dim: Optional[int] = None,
        val_dim: Optional[int] = None
    ):

        self.hidden_dim = hidden_dim
        assert num_heads >= 1
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.att_dim = att_dim if att_dim is not None else hidden_dim
        self.val_dim = val_dim if val_dim is not None else self.att_dim
        assert self.att_dim % self.num_heads == 0
        assert self.val_dim % self.num_heads == 0
        self.att_dim_per_head = self.att_dim // self.num_heads
        self.val_dim_per_head = self.val_dim // self.num_heads
        self.ignore_edges = ignore_edges


class AttentionLayer(nn.Module):

    def __init__(self, params: AttentionParams):
        super(AttentionLayer, self).__init__()
        self.params = params
        self.attn_query = nn.Parameter(torch.Tensor(
            params.hidden_dim, params.num_heads, params.att_dim_per_head))
        self.attn_keys = nn.Parameter(torch.Tensor(
            params.hidden_dim, params.num_heads, params.att_dim_per_head))
        self.attn_values = nn.Parameter(torch.Tensor(
            params.hidden_dim, params.num_heads, params.val_dim_per_head))
        self.weight_out = nn.Parameter(torch.Tensor(
            params.num_heads, params.val_dim_per_head, params.hidden_dim))
        self.attn_biases = nn.Linear(
            params.hidden_dim, params.num_heads * params.num_edge_types)
        self.reset_parameters()

    def reset_parameters(self):
        xavier_reset(self)

    def compute_qkvb(self, states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Argument:
          - states: shape (batch, node, hidden_dim)
        Returns:
          - query, key: shape (batch, node, head, att_dim/head)
          - value: shape (batch, node, head, val_dim/head)
        """
        queries = torch.einsum('btd,dha->btha', states, self.attn_query)
        keys    = torch.einsum('btd,dha->btha', states, self.attn_keys)
        values  = torch.einsum('btd,dhv->bthv', states, self.attn_values)
        return queries, keys, values

    def get_edge_biases(self, states: Tensor) -> Tensor:
        """
        Returns:
          - bias tensor of shape (batch, node, head, edge_type)
        """
        biases_shape = (*states.shape[:2],
                         self.params.num_heads, self.params.num_edge_types)
        return self.attn_biases(states).reshape(biases_shape)

    def get_attention_weights(
        self,
        queries: Tensor,
        keys: Tensor,
        edges: Tensor,
        biases: Union[Tensor, None],
        mask: Tensor
    ) -> Tensor:
        """
        Arguments:
          - queries, keys: shape (batch, node, head, att_dim/head)
          - biases: shape (batch, node, head, edge_type)
          - alpha, result: shape (batch, head, query_node, key_node)
          - mask: a tensor of shape (batch, query_node, key_node)
        """

        alpha = torch.einsum('bqha,bkha->bhqk', queries, keys)

        # ADD THE BIAS (alpha += b*sum(k))
        if biases is not None:
            ebatch, esrc, edst, etype = (
                edges[:,0], edges[:,1], edges[:,2], edges[:,3])
            ebiases = biases[ebatch, edst, :, etype]  # size (num_edges, num_heads)
            # number of batches, number of nodes (queries), number of heads
            nb, nn, nh = queries.shape[:3]
            # 1D indexes for edges in a 'bqk' tensor (q=dst, k=src)
            eidx = esrc + nn * edst + (nn * nn) * ebatch
            biases_bqkh = (
                torch_scatter.scatter(ebiases, eidx, dim=0, dim_size=nb*nn*nn)
                .reshape(nb, nn, nn, nh))
            summed_keys = torch.sum(keys, dim=-1) # bkh
            alpha = alpha + (
                torch.einsum('bqkh,bkh->bhqk', biases_bqkh, summed_keys))

        # MASK ATTENTION
        mask = mask.unsqueeze(1)  # head dimension
        alpha = alpha.masked_fill(mask == 0, -1e9)

        # SCALE ATTENTION
        alpha = alpha / math.sqrt(self.params.att_dim_per_head)
        return F.softmax(alpha, dim=-1)

    def forward(self, states: Tensor, edges: Tensor, mask: Tensor) -> Tensor:
        """
        Arguments:
            - state: shape (batch_size, num_tokens, hidden_dim)
            - edges: a long tensor with shape (num_edges, 4).
              An edge is represented as a (batch, src, dst, type) quadruple.
            - mask: a tensor of shape (batch_size, num_tokens, num_tokens).
              mask[b,i,j] is 1 if i can attend j in batch b and 0 otherwise.
              If mask is only used to ignore padding, the second dimension is
              typically 1.

        Returns:
            - att_values: shape (batch_size, num_tokens, val_dim)
        """
        queries, keys, values = self.compute_qkvb(states)
        biases = (self.get_edge_biases(states) if not self.params.ignore_edges
                  else None)
        scores = self.get_attention_weights(queries, keys, edges, biases, mask)
        # (batch, query_node, head, val_dim/head)
        aggr = torch.einsum('bhqk,bkhv->bqhv', scores, values)
        return torch.einsum('bqhv,hvd->bqd', aggr, self.weight_out)


@dataclass
class TransformerParams:
    att: AttentionParams
    num_layers: int
    ff_dim: int  # Dimension for feed forward layers
    dropout_rate: float
    use_final_layer_norm: bool = False


class TransformerEncoder(nn.Module):

    def __init__(self, params: TransformerParams):
        super(TransformerEncoder, self).__init__()
        self.params = params
        self.attention = nn.ModuleList([])
        self.layer_norm = nn.ModuleList([])
        self.ff_1 = nn.ModuleList([])
        self.ff_2 = nn.ModuleList([])
        self.layer_norm_out = nn.LayerNorm(params.att.hidden_dim)
        for _ in range(params.num_layers):
            self.layer_norm.append(nn.LayerNorm(params.att.hidden_dim))
            self.attention.append(AttentionLayer(params.att))
            self.layer_norm.append(nn.LayerNorm(params.att.hidden_dim))
            self.ff_1.append(nn.Linear(params.att.hidden_dim, params.ff_dim))
            self.ff_2.append(nn.Linear(params.ff_dim, params.att.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # By default, we use Xavier initialization
        xavier_reset(self)
        # Layer norm initialization may be a little different
        for l in self.layer_norm:
            l.reset_parameters()
        self.layer_norm_out.reset_parameters()

    def dropout(self, x):
        return F.dropout(x, p=self.params.dropout_rate, training=self.training)

    def forward(self, states: Tensor, edges: Tensor, mask: Tensor) -> Tensor:
        """
        Returns a tensor of shape (batch_size, num_tokens, d_model)
        """
        for i in range(self.params.num_layers):
            # First part: attention
            new_states = self.layer_norm[2*i](states)
            new_states = self.attention[i](new_states, edges, mask)
            new_states = self.dropout(new_states)
            states = states + new_states
            # Second part: feed forward network
            new_states = self.layer_norm[2*i+1](states)
            new_states = self.ff_1[i](new_states)
            new_states = self.dropout(new_states)
            new_states = F.relu(new_states)
            new_states = self.ff_2[i](new_states)
            new_states = self.dropout(new_states)
            states = states + new_states
        if self.params.use_final_layer_norm:
            states = self.layer_norm_out(states)
        return states


def xavier_reset(net):
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


#####
## References
#####


# Graph encoding:
# https://arxiv.org/pdf/1711.00740.pdf
# https://papers.nips.cc/paper/2018/file/65b1e92c585fd4c2159d5f33b5030ff2-Paper.pdf

# Missing scatter op in pytorch
# https://github.com/pytorch/pytorch/issues/21815
# https://www.reddit.com/r/pytorch/comments/ev0ahj/torchscatter_add_to_multiple_dimensions/


#####
## Roadmap
#####


# TODO: use scatter_coo for optimization
