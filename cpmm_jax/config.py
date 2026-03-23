"""Configuration objects for the CPMM JAX training stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    max_seq_len: int = 1024
    d_model: int = 384
    conv_kernel_size: int = 5
    conv_expansion: int = 2
    num_slots: int = 8
    chunk_size: int = 64
    refinement_steps: int = 2
    refinement_step_size: float = 0.15
    relation_rank: int = 64
    graph_steps: int = 3
    num_graph_symbols: int = 512
    num_graph_values: int = 512
    dropout_rate: float = 0.0


@dataclass
class CPMMConfig:
    lambda_lm: float = 1.0
    lambda_answer: float = 1.0
    lambda_graph_answer: float = 0.8
    lambda_graph_structure: float = 0.15


@dataclass
class CodeDataConfig:
    dataset_name: str = "bigcode/the-stack-v2-dedup"
    language: str = "Python"
    tokenizer_path: str = "drive/MyDrive/cpmm_code_llm/tokenizer.model"
    cache_dir: str = "drive/MyDrive/cpmm_code_llm/cache"
    train_shards_glob: str = "drive/MyDrive/cpmm_code_llm/data/train/*.jsonl"
    eval_shards_glob: str = "drive/MyDrive/cpmm_code_llm/data/eval/*.jsonl"
    parser_max_nodes: int = 512
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3


@dataclass
class TrainingConfig:
    seed: int = 7
    global_batch_size: int = 64
    micro_batch_size: int = 8
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    total_steps: int = 10000
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    log_every: int = 25
    eval_every: int = 250
    save_every: int = 250
    max_checkpoints_to_keep: int = 3
    mesh_batch_axis: str = "data"
    mesh_shape: tuple[int, ...] = (8,)


@dataclass
class ChatTuneConfig:
    train_steps: int = 1000
    learning_rate: float = 1e-4
    dialogue_mix_ratio: float = 0.5
    code_task_mix_ratio: float = 0.5
    system_prompt: str = "You are a helpful coding assistant."


@dataclass
class CheckpointConfig:
    drive_root: str = "/content/drive/MyDrive/cpmm_code_llm"
    checkpoint_dir: str = "/content/drive/MyDrive/cpmm_code_llm/checkpoints"
    metadata_path: str = "/content/drive/MyDrive/cpmm_code_llm/checkpoints/metadata.json"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cpmm: CPMMConfig = field(default_factory=CPMMConfig)
    data: CodeDataConfig = field(default_factory=CodeDataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    chat: ChatTuneConfig = field(default_factory=ChatTuneConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def to_json(config: ExperimentConfig) -> str:
    return json.dumps(asdict(config), indent=2)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(to_json(config), encoding="utf-8")


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> ExperimentConfig:
    base = asdict(ExperimentConfig())
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    merged = _deep_update(base, loaded)
    return ExperimentConfig(
        model=ModelConfig(**merged["model"]),
        cpmm=CPMMConfig(**merged["cpmm"]),
        data=CodeDataConfig(**merged["data"]),
        train=TrainingConfig(**merged["train"]),
        chat=ChatTuneConfig(**merged["chat"]),
        checkpoint=CheckpointConfig(**merged["checkpoint"]),
    )
