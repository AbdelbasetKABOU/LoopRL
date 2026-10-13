"""
Utilities for updating neural networks.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import looprl
import numpy as np
import psutil  # type: ignore
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm  # type: ignore

from looprl_lib.samples import (SamplesBatch, convert_and_collate_samples,
                                to_device)

from .dgt import AttentionParams
from .models import LooprlNetwork, LooprlNetworkParams
from .params import NetworkParams, TrainerParams

FLOAT_EPS = 1e-15


NETWORK_FILE = "net.pt"
EPOCHS_STATS_FILE = "epochs.json"
STEPS_STATS_FILE = "steps.json"


@dataclass
class TrainingLogger:
    """
    We log data at every step and every epoch.
    - At every step: epoch, loss, loss components, learning rate
    - At very epoch: validation loss (+ components)
    """
    steps_file: str
    epochs_file: str

    def __post_init__(self):
        self.steps = open(self.steps_file, "w")
        self.epochs = open(self.epochs_file, "w")

    def log_step(self, stats: dict[str, Any]):
        print(json.dumps(stats), file=self.steps)
        self.steps.flush()

    def log_epoch(self, stats: dict[str, Any]):
        print(json.dumps(stats), file=self.epochs)
        self.epochs.flush()

    def close(self) -> None:
        self.steps.close()
        self.epochs.close()


def make_network(
    params: NetworkParams,
    tconf: looprl.TensorizerConfig,
    agent_spec: looprl.AgentSpec
):
    net_params = LooprlNetworkParams(
        att = AttentionParams(
            hidden_dim=tconf['d_model'],
            num_heads=params.num_heads,
            ignore_edges=params.ignore_edges,
            num_edge_types=looprl.num_edge_types()),
        probe_encoder_layers=params.probe_encoder_layers,
        action_encoder_layers=params.action_encoder_layers,
        combiner_layers=params.combiner_layers,
        ff_dim=tconf['d_model']*2,
        dropout_rate=params.dropout_rate,
        value_head_input_dim=params.head_dim,
        policy_head_input_dim=params.head_dim,
        value_head_num_layers=params.num_head_layers,
        policy_head_num_layers=params.num_head_layers,
        ignore_pos_encoding=params.ignore_pos_encoding)
    return LooprlNetwork(net_params, tconf, agent_spec)


class WarmupCosineSchedule(LambdaLR):
    """
    - Linear warmup and then cosine decay.
    - Linearly increases learning rate from 0 to 1 over
     `  `warmup_steps` training steps.
    - Decreases learning rate from 1. to 0. over remaining
        `total_steps - warmup_steps` steps following a cosine curve.
    - If `cycles` (default=0.5) is different from default,
        learning rate follows cosine function after warmup.

    References:
      - github.com/TalSchuster/pytorch-transformers/
      - huggingface.co/docs/transformers/main_classes/optimizer_schedules
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        cycles: float = .5,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = float(cycles)
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = (
            float(step - self.warmup_steps) /
            float(max(1, self.total_steps - self.warmup_steps)))
        return max(
            0.0,
            0.5 * (1. + math.cos(2.0 * math.pi * self.cycles * progress)))


#####
## Standard training loop
#####


def update_network(
    train_set: Dataset,
    validation_set: Dataset,
    in_net_file: Optional[str],
    training_dir: str,
    net_params: NetworkParams,
    tconf: looprl.TensorizerConfig,
    train_params: TrainerParams,
    agent_spec: looprl.AgentSpec,
    max_cuda_memory_fraction: Optional[float],
    log: Callable[[str], None] = lambda msg: print(msg, end="\n\n")
):
    if max_cuda_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(max_cuda_memory_fraction)
    # Loading the network weights
    if in_net_file is not None:
        net_state_dict = torch.load(in_net_file)
    else:
        net_state_dict = None
    # Setting params
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = psutil.cpu_count(logical=True)
    # Creating the network
    net = make_network(net_params, tconf, agent_spec)
    if net_state_dict is not None:
        net.load_state_dict(net_state_dict)
    # Initializing the optimizer
    samples_per_epoch = len(train_set)
    batch_size = train_params.batch_size
    steps_per_epoch = samples_per_epoch / batch_size
    warmup_steps = math.ceil(steps_per_epoch * train_params.warmup_epochs)
    total_steps = math.ceil(steps_per_epoch * train_params.max_epochs)
    optimizer = AdamW(
        net.parameters(),
        lr=train_params.lr_base,
        weight_decay=train_params.weight_decay,
        betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)
    # Creating the dataloaders
    train = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(not isinstance(train_set, IterableDataset)),
        num_workers=num_workers,
        drop_last=False, collate_fn=convert_and_collate_samples)
    validation = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False, collate_fn=convert_and_collate_samples)
    # Printing a summary to the user
    os.makedirs(training_dir, exist_ok=True)
    log(f"Starting network updates.")
    summary: list[str] = []
    summary += [f"Loading network from: {in_net_file}"]
    summary += [f"Training dir: {training_dir}"]
    summary += [f"Num training samples: {len(train_set)}"]
    if validation is not None:
        summary += [f"Num validation samples: {len(validation_set)}"]
    summary += [f"Batch size: {train_params.batch_size}"]
    summary += [f"Max epochs: {train_params.max_epochs}"]
    summary += [f"Warmup epochs: {train_params.warmup_epochs}"]
    summary += [f"LR base: {train_params.lr_base}"]
    log("\n".join(summary))
    logger = TrainingLogger(
        os.path.join(training_dir, STEPS_STATS_FILE),
        os.path.join(training_dir, EPOCHS_STATS_FILE))
    # Training loop
    best_weights = net.state_dict()
    without_improvement = 0
    lowest_valid_loss = math.inf
    net.to(device=device)
    for i in range(train_params.max_epochs + 1):
        # Training step
        if i > 0:
            skipped: list[int] = []
            net.train(mode=True)
            for batch in tqdm(train):
                batch = to_device(batch, device)
                optimizer.zero_grad()
                try:
                    loss, components = alphazero_loss(
                        net, batch, train_params, agent_spec)
                except Exception as e:
                    if train_params.skip_batch_on_exception:
                        num_actions = batch.choice.actions.batch_size
                        skipped += [num_actions]
                        continue
                    else:
                        raise e
                loss.backward()
                optimizer.step()
                scheduler.step()
                stats = {
                    'epoch': i,
                    'lr': scheduler.get_last_lr()[0],
                    'train_loss': loss.item() }
                for k, v in components.items():
                    stats['train_' + k] = v.item()
                logger.log_step(stats)
            if skipped:
                min_a = min(skipped)
                max_a = max(skipped)
                log(f"Skipped {len(skipped)} batches ({min_a} - {max_a}).")
        # Validation step
        net.train(mode=False)
        loss_history = []
        components_history = []
        with torch.no_grad():
            for batch in tqdm(validation):
                batch = to_device(batch, device)
                loss, components = alphazero_loss(
                    net, batch, train_params, agent_spec)
                loss_history.append(loss)
                components_history.append(components)
            assert loss_history, "Empty validation set"
            mean_loss = np.mean([l.item() for l in loss_history])
            stats = {'epoch': i, 'valid_loss': mean_loss}
            for k in components_history[0]:
                mean_comp = np.mean([cs[k].item() for cs in components_history])
                stats['valid_' + k] = mean_comp
            if mean_loss < lowest_valid_loss:
                log(f"Epoch {i}: best validation loss ({mean_loss:.5f}).")
                lowest_valid_loss = mean_loss
                without_improvement = 0
                net.to(device='cpu')
                best_weights = net.state_dict()
                output_file = os.path.join(training_dir, NETWORK_FILE)
                torch.save(best_weights, output_file)
                net.to(device=device)
            else:
                log(f"Epoch {i}: validation loss: ({mean_loss:.5f}).")
                without_improvement += 1
                if without_improvement >= train_params.improvement_required:
                    break
            logger.log_epoch(stats)
    logger.close()


def init_and_save_network(
    net_params: NetworkParams,
    tconf: looprl.TensorizerConfig,
    agent_spec: looprl.AgentSpec,
    file: str
):
    net = make_network(net_params, tconf, agent_spec)
    torch.save(net.state_dict(), file)


#####
## Standard losses
#####


def alphazero_loss(
    net: LooprlNetwork,
    batch: SamplesBatch[torch.Tensor],
    params: TrainerParams,
    agent_spec: looprl.AgentSpec
):
    vpreds, policy = net(batch.choice)
    batch_size = vpreds.shape[0]
    # Policy loss
    ptarget = batch.policy_target
    log_policy = torch.log(policy + FLOAT_EPS)
    log_ptarget = torch.log(ptarget + FLOAT_EPS)
    ploss = (ptarget * (log_ptarget - log_policy)).sum() / batch_size
    # Event and outcome prediction losses
    num_outcomes = len(agent_spec['event_rewards'])
    num_events = len(agent_spec['event_rewards'])
    vtarget = batch.value_target
    log_vtarget = torch.log(vtarget + FLOAT_EPS)
    log_vpreds = torch.log(vpreds + FLOAT_EPS)
    vloss = vtarget * (log_vtarget - log_vpreds)
    oloss = vloss[:, :num_outcomes].sum() / batch_size
    eloss = vloss[:, num_outcomes:].sum() / batch_size / num_events
    # Multiplying by coeffs
    oloss = params.outcome_loss_coeff * oloss
    eloss = params.event_loss_coeff * eloss
    ploss = params.policy_loss_coeff * ploss
    # Adding loss terms together
    components = {
        'outcome_loss': oloss,
        'event_loss': eloss,
        'policy_loss': ploss}
    return oloss + eloss + ploss, components
