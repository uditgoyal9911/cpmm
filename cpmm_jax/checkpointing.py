"""Drive-backed Orbax checkpoint utilities for resumable Colab training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp

from .config import CheckpointConfig
from .data_pipeline import LoaderState


def ensure_checkpoint_dirs(config: CheckpointConfig) -> None:
    Path(config.drive_root).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def create_checkpoint_manager(config: CheckpointConfig, max_to_keep: int) -> ocp.CheckpointManager:
    ensure_checkpoint_dirs(config)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(
        directory=config.checkpoint_dir,
        checkpointers={
            "state": ocp.StandardCheckpointer(),
            "metadata": ocp.JsonCheckpointHandler(),
        },
        options=options,
    )


def metadata_payload(
    step: int,
    epoch: int,
    rng_seed: int,
    loader_state: LoaderState,
    tokenizer_path: str,
    stage: str,
) -> dict[str, Any]:
    return {
        "step": step,
        "epoch": epoch,
        "rng_seed": rng_seed,
        "loader_state": {
            "shard_index": loader_state.shard_index,
            "sample_offset": loader_state.sample_offset,
            "epoch": loader_state.epoch,
            "rng_seed": loader_state.rng_seed,
        },
        "tokenizer_path": tokenizer_path,
        "stage": stage,
    }


def save_checkpoint(
    manager: ocp.CheckpointManager,
    step: int,
    train_state: Any,
    metadata: dict[str, Any],
) -> None:
    manager.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(train_state),
            metadata=ocp.args.JsonSave(metadata),
        ),
    )


def latest_step(manager: ocp.CheckpointManager) -> int | None:
    return manager.latest_step()


def restore_checkpoint(manager: ocp.CheckpointManager, step: int, state_shape: Any) -> tuple[Any, dict[str, Any]]:
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(state_shape),
            metadata=ocp.args.JsonRestore(),
        ),
    )
    return restored.state, restored.metadata


def save_lightweight_metadata(config: CheckpointConfig, payload: dict[str, Any]) -> None:
    Path(config.metadata_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_lightweight_metadata(config: CheckpointConfig) -> dict[str, Any] | None:
    path = Path(config.metadata_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
