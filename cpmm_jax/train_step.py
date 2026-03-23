"""Train and eval step helpers for the JAX CPMM code model."""

from __future__ import annotations

from typing import Any

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from .config import CPMMConfig, ModelConfig, TrainingConfig
from .model import CPMMCodeModel


class CPMMTrainState(train_state.TrainState):
    rng: jax.Array


def create_learning_rate_schedule(config: TrainingConfig):
    warmup = optax.linear_schedule(0.0, config.learning_rate, config.warmup_steps)
    decay = optax.cosine_decay_schedule(config.learning_rate, max(config.total_steps - config.warmup_steps, 1))
    return optax.join_schedules([warmup, decay], boundaries=[config.warmup_steps])


def create_train_state(
    rng: jax.Array,
    model_config: ModelConfig,
    cpmm_config: CPMMConfig,
    train_config: TrainingConfig,
) -> tuple[CPMMTrainState, CPMMCodeModel]:
    del cpmm_config
    model = CPMMCodeModel(model_config)
    init_tokens = jnp.zeros((1, model_config.max_seq_len), dtype=jnp.int32)
    init_lengths = jnp.ones((1,), dtype=jnp.int32)
    init_query_idx = jnp.zeros((1,), dtype=jnp.int32)
    init_query_mask = jnp.ones((1,), dtype=jnp.bool_)
    init_graph_axis = jnp.zeros((1, model_config.max_seq_len), dtype=jnp.int32)
    variables = model.init(
        rng,
        init_tokens,
        init_lengths,
        init_query_idx,
        init_query_mask,
        init_graph_axis,
        init_graph_axis,
        init_graph_axis.astype(jnp.bool_),
        init_graph_axis,
        init_graph_axis.astype(jnp.bool_),
        init_graph_axis,
        init_graph_axis.astype(jnp.bool_),
    )
    schedule = create_learning_rate_schedule(train_config)
    tx = optax.chain(
        optax.clip_by_global_norm(train_config.grad_clip_norm),
        optax.adamw(schedule, weight_decay=train_config.weight_decay),
    )
    state = CPMMTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        rng=rng,
    )
    return state, model


def compute_losses(
    params: Any,
    model: CPMMCodeModel,
    batch: dict[str, jax.Array],
    cpmm_config: CPMMConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    outputs = model.apply(
        {"params": params},
        batch["token_ids"],
        batch["lengths"],
        batch["query_idx"],
        batch["query_mask"],
        batch["event_markers"],
        batch["source_idx"],
        batch["source_mask"],
        batch["target_symbol_idx"],
        batch["target_symbol_mask"],
        batch["target_value_idx"],
        batch["target_value_mask"],
    )
    seq_len = batch["token_ids"].shape[1]
    token_mask = batch["lengths"][:, None] > jnp.arange(seq_len)[None, :]
    lm_targets = jnp.clip(batch["token_ids"], 0)
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(outputs.lm_logits, lm_targets)
    lm_loss = (per_token_loss * token_mask).sum() / jnp.maximum(token_mask.sum(), 1)
    answer_loss = optax.softmax_cross_entropy_with_integer_labels(outputs.answer_logits, batch["answer_idx"]).mean()
    graph_answer_loss = optax.softmax_cross_entropy_with_integer_labels(outputs.graph_logits, batch["answer_idx"]).mean()
    map_loss = jnp.mean((outputs.map_memory - batch["map_target"]) ** 2)
    step_loss = jnp.mean((outputs.step_memory - batch["step_target"]) ** 2)
    graph_structure_loss = map_loss + step_loss
    total = (
        cpmm_config.lambda_lm * lm_loss
        + cpmm_config.lambda_answer * answer_loss
        + cpmm_config.lambda_graph_answer * graph_answer_loss
        + cpmm_config.lambda_graph_structure * graph_structure_loss
    )
    metrics = {
        "loss": total,
        "lm_loss": lm_loss,
        "answer_loss": answer_loss,
        "graph_answer_loss": graph_answer_loss,
        "graph_structure_loss": graph_structure_loss,
    }
    return total, metrics


@jax.jit
def train_step(
    state: CPMMTrainState,
    model: CPMMCodeModel,
    batch: dict[str, jax.Array],
    cpmm_config: CPMMConfig,
) -> tuple[CPMMTrainState, dict[str, jax.Array]]:
    grad_fn = jax.value_and_grad(compute_losses, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, model, batch, cpmm_config)
    del loss
    state = state.apply_gradients(grads=grads)
    return state, metrics


@jax.jit
def eval_step(
    state: CPMMTrainState,
    model: CPMMCodeModel,
    batch: dict[str, jax.Array],
    cpmm_config: CPMMConfig,
) -> dict[str, jax.Array]:
    _, metrics = compute_losses(state.params, model, batch, cpmm_config)
    return metrics
