from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from omegaconf import MISSING

@dataclass
class EnvironmentConfig:
    name: str = MISSING
    max_steps: int = 100
    render_mode: Optional[str] = None

@dataclass
class DatasetConfig:
    name: str = MISSING
    path: str = "datasets/"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    split: str = "train"

@dataclass
class AgentConfig:
    encoder_type: str = MISSING
    decoder_type: str = MISSING
    hidden_dim: int = 256
    learning_rate: float = 1e-4

@dataclass
class CommunicationConfig:
    protocol: str = "discrete"
    vocab_size: int = 20
    max_length: int = 10
    temperature: float = 1.0
    noise_prob: float = 0.0

@dataclass
class RewardConfig:
    alpha_reconstruction: float = 1.0
    beta_compression: float = 0.1
    gamma_diversity: float = 0.1
    delta_efficiency: float = 0.0
    penalties: float = 0.5

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4

@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 500

@dataclass
class TrainingConfig:
    epochs: int = 100
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    clip_grad_norm: float = 1.0

@dataclass
class EvaluationConfig:
    eval_freq: int = 1000
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss", "compression", "entropy"])

@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    log_dir: str = "logs/"
    checkpoint_dir: str = "checkpoints/"

@dataclass
class Config:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
