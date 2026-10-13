from dataclasses import dataclass, field
from typing import Any, Optional

from dataclasses_json import dataclass_json
from looprl import TensorizerConfig, TokenizerConfig

from looprl_lib.param_util import from_diff


@from_diff(name='encoding')
@dataclass_json
@dataclass(frozen=True)
class EncodingParams:
    d_model: int = 128
    pos_enc_size: int = 32
    uid_emb_size: int = 16
    const_emb_size: int = 0  # no binary encoding for constants
    enable_numerical_edges: bool = True
    add_reverse_edges: bool = True

    @property
    def tensorizer_config(self) -> TensorizerConfig:
        return self.__dict__  #type: ignore

    @property
    def tokenizer_config(self) -> TokenizerConfig:
        return self.__dict__  #type: ignore


@from_diff(name='trainer')
@dataclass_json
@dataclass(frozen=True)
class TrainerParams:
    max_epochs: int = 6
    improvement_required: int = 1
    warmup_epochs: float = 0.2
    batch_size: int = 400
    lr_base: float = 5e-4
    weight_decay: float = 0.01
    outcome_loss_coeff: float = 1.
    event_loss_coeff: float = 1.
    policy_loss_coeff: float = 1.
    skip_batch_on_exception: bool = True


@from_diff(name='network')
@dataclass_json
@dataclass(frozen=True)
class NetworkParams:
    # d_model is in EncodingParams
    num_heads: int = 4
    probe_encoder_layers: int = 6
    action_encoder_layers: int = 3
    combiner_layers: int = 1
    dropout_rate: float = 0.05
    num_head_layers: int = 2
    head_dim: int = 256
    ignore_edges: bool = False
    ignore_pos_encoding: bool = False


@from_diff(name='search')
@dataclass_json
@dataclass(frozen=True)
class SearchLimits:
    max_proof_length: int
    max_probe_size: int = 80
    max_action_size: int = 12


@from_diff(name='mcts')
@dataclass_json
@dataclass(frozen=True)
class MctsParams:
    num_simulations: int = 64
    num_considered_actions: int = 8
    value_scale: float = 0.1
    max_visit_init: float = 50
    fpu_red: float = 0.
    reset_tree: bool = False
    max_tree_size: Optional[int] = 256
    bias_eps: float = 0.0
    dirichlet_alpha: float = 10.
    dirichlet_eps: Optional[float] = None


@from_diff(name='pretraining')
@dataclass_json
@dataclass(frozen=True)
class PretrainingParams:
    enable_pretraining: bool = False
    num_samples: int = 500_000
    num_validation_samples: int = 100_000
    true_false: bool = False
    randomize_uids: bool = False
    training: TrainerParams = TrainerParams(
        batch_size=1024,
        lr_base=1e-3,
        warmup_epochs=0.2,
        max_epochs=8,
        improvement_required=2)


@from_diff(name='agent')
@dataclass_json
@dataclass(frozen=True)
class AgentParams:
    search: SearchLimits
    encoding: EncodingParams = EncodingParams()
    network: NetworkParams = NetworkParams()
    mcts: MctsParams = MctsParams()
    training: TrainerParams = TrainerParams()
    pretraining: PretrainingParams = PretrainingParams()
    use_biases_for_initial_policy: bool = False
    num_workers: int = 600
    num_processes: Optional[int] = None
    num_waited_processes: Optional[int] = None
    num_iters: int = 20
    num_problems_per_iter: int = 8000
    num_validation_problems: int = 800
    training_window: list[int] = field(default_factory=lambda:
        [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7])


@from_diff()
@dataclass_json
@dataclass(frozen=True)
class TeacherParams:
    agent: AgentParams = AgentParams(
        use_biases_for_initial_policy=False,
        search=SearchLimits(
            max_proof_length=60,
            max_probe_size=80),
        mcts = MctsParams(
            dirichlet_eps=0.25,
            reset_tree=True,
            fpu_red=0.1),
        training=TrainerParams(
            outcome_loss_coeff = 0.7,
            event_loss_coeff = 3.0,
            policy_loss_coeff = 1.))


@from_diff()
@dataclass_json
@dataclass(frozen=True)
class SolverParams:
    agent: AgentParams = AgentParams(
        num_iters=20,
        num_problems_per_iter=20_000,
        num_validation_problems=5_000,
        network = NetworkParams(
            dropout_rate=0.1),
        mcts = MctsParams(
            num_considered_actions=8,
            num_simulations=32,
            reset_tree=False),
        training=TrainerParams(
            max_epochs=1,
            batch_size=300,
            lr_base=3e-4),
        search=SearchLimits(max_proof_length=12))


@from_diff()
@dataclass_json
@dataclass(frozen=True)
class Params:
    teacher: TeacherParams = TeacherParams()
    solver: SolverParams = SolverParams()
    extra_teacher_problems: int = 10_000
    num_teacher_iters_used_by_solver: int = 5
    max_cuda_memory_fraction: Optional[float] = 0.85

    @staticmethod
    def from_diff(d: dict[str, Any]) -> 'Params': ...

    def update(self, diff: dict[str, Any]) -> 'Params': ...


#####
# Standard parameter sets
#####


ParamsDiff = dict[str, Any]


def toy_params() -> ParamsDiff:
    """
    Params that lead to a very short training session.
    Useful to run a sanity check before a large experiment.
    """
    return {
        '::agent.num_iters': 2,
        '::agent.num_problems_per_iter': 10,
        '::agent.num_validation_problems': 10,
        '::pretraining.num_samples': 10,
        '::pretraining.num_validation_samples': 10,
        '::agent.training.max_epochs': 2,
        '::agent.num_workers': 8,
        '::agent.training.batch_size': 64,
        'extra_teacher_problems': 10,
        'num_teacher_iters_used_by_solver': 2}


def alien_params() -> ParamsDiff:
    return {}


def aurora_params() -> ParamsDiff:
    return {
        '::agent.num_workers': 300,
        '::agent.training.batch_size': 256,
        '::agent.training.lr_base': 3e-4,
        '::pretraining.training.batch_size': 1024,
        '::pretraining.training.lr_base': 1e-3}


STD_PARAMS = {
    'toy': toy_params(),
    'alien': alien_params(),
    'aurora': aurora_params()}
