"""Instruction/chat fine-tuning helpers for the CPMM code model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp


CHAT_SYSTEM = "<|system|>"
CHAT_USER = "<|user|>"
CHAT_ASSISTANT = "<|assistant|>"
CHAT_END = "<|end|>"


@dataclass
class ChatExample:
    system: str
    user: str
    assistant: str


def format_chat_example(example: ChatExample) -> str:
    return (
        f"{CHAT_SYSTEM}\n{example.system}\n{CHAT_END}\n"
        f"{CHAT_USER}\n{example.user}\n{CHAT_END}\n"
        f"{CHAT_ASSISTANT}\n{example.assistant}\n{CHAT_END}\n"
    )


def build_chat_corpus(examples: Iterable[ChatExample]) -> list[str]:
    return [format_chat_example(example) for example in examples]


def tokenize_chat_examples(
    examples: Sequence[ChatExample],
    tokenizer,
    max_seq_len: int,
    pad_id: int,
) -> dict[str, jnp.ndarray]:
    token_rows = []
    for example in examples:
        token_ids = tokenizer.encode(format_chat_example(example))
        token_ids = token_ids[:max_seq_len]
        padded = token_ids + [pad_id] * max(0, max_seq_len - len(token_ids))
        token_rows.append(padded[:max_seq_len])
    token_array = jnp.asarray(token_rows, dtype=jnp.int32)
    return {
        "input_ids": token_array[:, :-1],
        "target_ids": token_array[:, 1:],
    }


def answer_mask(token_ids: jnp.ndarray, assistant_token_id: int, end_token_id: int) -> jnp.ndarray:
    """Build a loss mask that focuses updates on assistant spans."""
    batch_size, seq_len = token_ids.shape
    mask = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
    for batch_idx in range(batch_size):
        active = False
        row_mask = []
        for token in token_ids[batch_idx]:
            token_int = int(token)
            if token_int == assistant_token_id:
                active = True
            elif token_int == end_token_id and active:
                active = False
            row_mask.append(active)
        mask = mask.at[batch_idx].set(jnp.asarray(row_mask, dtype=jnp.bool_))
    return mask


def masked_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, loss_mask: jnp.ndarray) -> jnp.ndarray:
    log_probs = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1).squeeze(-1)
    masked = -log_probs * loss_mask.astype(jnp.float32)
    return masked.sum() / jnp.maximum(loss_mask.sum(), 1)
