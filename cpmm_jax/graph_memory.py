"""Typed graph memory for the JAX/Flax CPMM code model."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass
class GraphMemoryState:
    map_memory: jnp.ndarray
    step_memory: jnp.ndarray


class GraphMemoryUpdate(nn.Module):
    """Gated MAP/STEP update for a single event."""

    d_model: int
    num_symbols: int
    num_values: int

    @nn.compact
    def __call__(
        self,
        state: GraphMemoryState,
        evidence: jnp.ndarray,
        marker_id: jnp.ndarray,
        source_idx: jnp.ndarray,
        source_mask: jnp.ndarray,
        target_symbol_idx: jnp.ndarray,
        target_symbol_mask: jnp.ndarray,
        target_value_idx: jnp.ndarray,
        target_value_mask: jnp.ndarray,
    ) -> GraphMemoryState:
        map_gain = nn.Sequential(
            [nn.Dense(self.d_model), nn.gelu, nn.Dense(1)],
            name="map_gain",
        )(evidence)
        step_gain = nn.Sequential(
            [nn.Dense(self.d_model), nn.gelu, nn.Dense(1)],
            name="step_gain",
        )(evidence)

        src_onehot = jax.nn.one_hot(source_idx, self.num_symbols, dtype=jnp.float32)
        target_symbol_onehot = jax.nn.one_hot(target_symbol_idx, self.num_symbols, dtype=jnp.float32)
        target_value_onehot = jax.nn.one_hot(target_value_idx, self.num_values, dtype=jnp.float32)

        map_mask = ((marker_id == 1) | (marker_id == 2)) & source_mask & target_value_mask
        step_mask = (marker_id == 3) & source_mask & target_symbol_mask

        map_update = jnp.einsum("bs,bv->bsv", src_onehot, target_value_onehot) * jax.nn.sigmoid(map_gain)[:, None, :]
        step_update = jnp.einsum("bs,bt->bst", src_onehot, target_symbol_onehot) * jax.nn.sigmoid(step_gain)[:, None, :]

        next_map = state.map_memory + map_update * map_mask[:, None, None]
        next_step = state.step_memory + step_update * step_mask[:, None, None]
        return GraphMemoryState(map_memory=next_map, step_memory=next_step)


class GraphMemoryQuery(nn.Module):
    """Multi-hop graph walk readout."""

    d_model: int
    num_symbols: int
    num_values: int
    graph_steps: int = 3

    @nn.compact
    def __call__(
        self,
        state: GraphMemoryState,
        query_idx: jnp.ndarray,
        query_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        symbol_embed = nn.Embed(self.num_symbols, self.d_model, name="symbol_embed")
        value_embed = nn.Embed(self.num_values, self.d_model, name="value_embed")

        all_symbol_ids = jnp.arange(self.num_symbols)
        all_value_ids = jnp.arange(self.num_values)
        symbol_bank = symbol_embed(all_symbol_ids)
        value_bank = value_embed(all_value_ids)

        query_state = jax.nn.one_hot(query_idx, self.num_symbols, dtype=jnp.float32)
        query_state = query_state * query_mask[:, None]

        accumulated_values = jnp.einsum("bs,bsv->bv", query_state, state.map_memory)
        accumulated_symbols = jnp.einsum("bs,sd->bd", query_state, symbol_bank)

        def walk_step(carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _: None):
            walk, values, symbols = carry
            next_walk = jnp.einsum("bs,bst->bt", walk, state.step_memory)
            next_values = values + jnp.einsum("bs,bsv->bv", next_walk, state.map_memory)
            next_symbols = symbols + jnp.einsum("bs,sd->bd", next_walk, symbol_bank)
            return (next_walk, next_values, next_symbols), None

        (_, accumulated_values, accumulated_symbols), _ = jax.lax.scan(
            walk_step,
            (query_state, accumulated_values, accumulated_symbols),
            xs=None,
            length=self.graph_steps,
        )

        graph_value_vec = jnp.einsum("bv,vd->bd", accumulated_values, value_bank)
        graph_state = accumulated_symbols + graph_value_vec

        logits = nn.Sequential(
            [nn.Dense(self.d_model), nn.gelu, nn.Dense(self.num_values)],
            name="output_head",
        )(graph_state)
        feedback = nn.Sequential(
            [nn.Dense(self.d_model), nn.gelu, nn.Dense(self.d_model)],
            name="feedback_head",
        )(graph_state)
        return logits, feedback


class TypedGraphMemory(nn.Module):
    """Combined update + query graph memory module."""

    d_model: int
    num_symbols: int
    num_values: int
    graph_steps: int = 3

    def setup(self) -> None:
        self.updater = GraphMemoryUpdate(
            d_model=self.d_model,
            num_symbols=self.num_symbols,
            num_values=self.num_values,
            name="updater",
        )
        self.querier = GraphMemoryQuery(
            d_model=self.d_model,
            num_symbols=self.num_symbols,
            num_values=self.num_values,
            graph_steps=self.graph_steps,
            name="querier",
        )

    def init_state(self, batch_size: int) -> GraphMemoryState:
        return GraphMemoryState(
            map_memory=jnp.zeros((batch_size, self.num_symbols, self.num_values), dtype=jnp.float32),
            step_memory=jnp.zeros((batch_size, self.num_symbols, self.num_symbols), dtype=jnp.float32),
        )

    def update(
        self,
        state: GraphMemoryState,
        evidence: jnp.ndarray,
        marker_id: jnp.ndarray,
        source_idx: jnp.ndarray,
        source_mask: jnp.ndarray,
        target_symbol_idx: jnp.ndarray,
        target_symbol_mask: jnp.ndarray,
        target_value_idx: jnp.ndarray,
        target_value_mask: jnp.ndarray,
    ) -> GraphMemoryState:
        return self.updater(
            state,
            evidence,
            marker_id,
            source_idx,
            source_mask,
            target_symbol_idx,
            target_symbol_mask,
            target_value_idx,
            target_value_mask,
        )

    def __call__(
        self,
        state: GraphMemoryState,
        query_idx: jnp.ndarray,
        query_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.querier(state, query_idx, query_mask)
