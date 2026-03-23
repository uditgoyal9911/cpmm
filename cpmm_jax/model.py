"""Pure post-transformer CPMM code language model in Flax."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import ModelConfig
from .graph_memory import GraphMemoryState, TypedGraphMemory


@dataclass
class CPMMOutputs:
    lm_logits: jnp.ndarray
    answer_logits: jnp.ndarray
    graph_logits: jnp.ndarray
    map_memory: jnp.ndarray
    step_memory: jnp.ndarray
    slot_state: jnp.ndarray


class LocalChunkEncoder(nn.Module):
    """Local token encoder: embedding + positional + two conv layers."""

    config: ModelConfig

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        x = nn.Embed(self.config.vocab_size, self.config.d_model, name="token_embed")(token_ids)
        x = x + nn.Embed(self.config.max_seq_len, self.config.d_model, name="pos_embed")(
            jnp.arange(token_ids.shape[1])[None, :]
        )
        x = nn.Conv(
            features=self.config.d_model * self.config.conv_expansion,
            kernel_size=(self.config.conv_kernel_size,),
            padding="SAME",
            name="conv1",
        )(x)
        x = nn.gelu(x)
        x = nn.Conv(
            features=self.config.d_model,
            kernel_size=(self.config.conv_kernel_size,),
            padding="SAME",
            name="conv2",
        )(x)
        x = nn.LayerNorm(name="encoder_norm")(x)
        return x


class SlotRelationMixer(nn.Module):
    """Low-rank slot-slot interaction without full-sequence attention."""

    d_model: int
    relation_rank: int

    @nn.compact
    def __call__(self, slots: jnp.ndarray) -> jnp.ndarray:
        q = nn.Dense(self.relation_rank, name="q_proj")(slots)
        k = nn.Dense(self.relation_rank, name="k_proj")(slots)
        v = nn.Dense(self.d_model, name="v_proj")(slots)
        attn = jnp.einsum("bsd,btd->bst", q, k) / jnp.sqrt(float(self.relation_rank))
        attn = jax.nn.softmax(attn, axis=-1)
        mixed = jnp.einsum("bst,btd->bsd", attn, v)
        return nn.LayerNorm(name="mixer_norm")(slots + mixed)


class SlotWriter(nn.Module):
    """Writes evidence + graph feedback into slot memory."""

    d_model: int
    num_slots: int

    @nn.compact
    def __call__(
        self,
        slots: jnp.ndarray,
        evidence: jnp.ndarray,
        graph_feedback: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        projected_evidence = nn.Dense(self.d_model, name="evidence_proj")(evidence)
        projected_feedback = nn.Dense(self.d_model, name="feedback_proj")(graph_feedback)
        combined = projected_evidence + projected_feedback

        slot_scores = jnp.einsum(
            "bsd,bd->bs",
            nn.Dense(self.d_model, name="slot_key")(slots),
            nn.Dense(self.d_model, name="event_key")(evidence),
        ) / jnp.sqrt(float(self.d_model))
        slot_attention = jax.nn.softmax(slot_scores, axis=-1)[..., None]

        write_input = jnp.concatenate(
            [slots, jnp.broadcast_to(combined[:, None, :], slots.shape)],
            axis=-1,
        )
        gates = jax.nn.sigmoid(nn.Dense(1, name="gate")(write_input)) * slot_attention
        delta = jnp.tanh(nn.Dense(self.d_model, name="write")(write_input))
        return slots + gates * delta, gates


class SlotDynamics(nn.Module):
    """Predict next slot state from current state."""

    d_model: int

    @nn.compact
    def __call__(self, slots: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.d_model, name="dyn1")(slots)
        x = nn.tanh(x)
        x = nn.Dense(self.d_model, name="dyn2")(x)
        return nn.LayerNorm(name="slot_norm")(slots + x)


class EnergyDecoder(nn.Module):
    """Reconstructs evidence from slot summary for energy computation."""

    d_model: int

    @nn.compact
    def __call__(self, slots: jnp.ndarray) -> jnp.ndarray:
        summary = slots.mean(axis=1)
        x = nn.Dense(self.d_model, name="dec1")(summary)
        x = nn.gelu(x)
        return nn.Dense(self.d_model, name="dec2")(x)


class CPMMCodeModel(nn.Module):
    """Chunk-scanned CPMM with slot memory, graph memory, and LM heads."""

    config: ModelConfig

    def setup(self) -> None:
        self.encoder = LocalChunkEncoder(self.config, name="local_chunk_encoder")
        self.graph_memory = TypedGraphMemory(
            d_model=self.config.d_model,
            num_symbols=self.config.num_graph_symbols,
            num_values=self.config.num_graph_values,
            graph_steps=self.config.graph_steps,
            name="typed_graph_memory",
        )
        self.slot_dynamics = SlotDynamics(self.config.d_model, name="slot_dynamics")
        self.slot_writer = SlotWriter(self.config.d_model, self.config.num_slots, name="slot_writer")
        self.energy_decoder = EnergyDecoder(self.config.d_model, name="energy_decoder")
        self.relation_mixer = SlotRelationMixer(
            self.config.d_model, self.config.relation_rank, name="slot_relation_mixer"
        )
        self.lm_head = nn.Dense(self.config.vocab_size, name="lm_head")
        self.answer_head = nn.Sequential(
            [nn.Dense(self.config.d_model), nn.gelu, nn.Dense(self.config.num_graph_values)],
            name="answer_head",
        )
        self.slot_seed = self.param(
            "slot_seed",
            nn.initializers.normal(stddev=0.02),
            (self.config.num_slots, self.config.d_model),
        )

    def _energy(
        self,
        slot_state: jnp.ndarray,
        predicted_slots: jnp.ndarray,
        evidence: jnp.ndarray,
        gates: jnp.ndarray,
    ) -> jnp.ndarray:
        reconstruction = self.energy_decoder(slot_state)
        e_obs = jnp.mean((reconstruction - evidence) ** 2)
        e_dyn = jnp.mean((slot_state - predicted_slots) ** 2)
        e_sparse = jnp.mean(gates)
        return e_obs + 0.35 * e_dyn + 0.02 * e_sparse

    def _refine(
        self,
        slots: jnp.ndarray,
        predicted_slots: jnp.ndarray,
        evidence: jnp.ndarray,
        gates: jnp.ndarray,
    ) -> jnp.ndarray:
        def refine_body(current: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
            grad = jax.grad(lambda s: self._energy(s, predicted_slots, evidence, gates))(current)
            return current - self.config.refinement_step_size * grad, None

        refined, _ = jax.lax.scan(refine_body, slots, xs=None, length=self.config.refinement_steps)
        return refined

    def __call__(
        self,
        token_ids: jnp.ndarray,
        lengths: jnp.ndarray,
        query_idx: jnp.ndarray,
        query_mask: jnp.ndarray,
        event_markers: jnp.ndarray,
        source_idx: jnp.ndarray,
        source_mask: jnp.ndarray,
        target_symbol_idx: jnp.ndarray,
        target_symbol_mask: jnp.ndarray,
        target_value_idx: jnp.ndarray,
        target_value_mask: jnp.ndarray,
        deterministic: bool = True,
    ) -> CPMMOutputs:
        del deterministic
        batch_size, seq_len = token_ids.shape
        hidden = self.encoder(token_ids)

        chunk_size = self.config.chunk_size
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        padded_len = num_chunks * chunk_size
        pad_width = padded_len - seq_len

        def pad2(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.pad(x, ((0, 0), (0, pad_width)))

        hidden_p = jnp.pad(hidden, ((0, 0), (0, pad_width), (0, 0)))
        em_p = pad2(event_markers)
        si_p = pad2(source_idx)
        sm_p = pad2(source_mask)
        tsi_p = pad2(target_symbol_idx)
        tsm_p = pad2(target_symbol_mask)
        tvi_p = pad2(target_value_idx)
        tvm_p = pad2(target_value_mask)

        def reshape_chunks(x: jnp.ndarray, last_dim: int | None = None) -> jnp.ndarray:
            if last_dim is not None:
                return x.reshape(batch_size, num_chunks, chunk_size, last_dim)
            return x.reshape(batch_size, num_chunks, chunk_size)

        hidden_chunks = reshape_chunks(hidden_p, self.config.d_model)
        em_chunks = reshape_chunks(em_p)
        si_chunks = reshape_chunks(si_p)
        sm_chunks = reshape_chunks(sm_p)
        tsi_chunks = reshape_chunks(tsi_p)
        tsm_chunks = reshape_chunks(tsm_p)
        tvi_chunks = reshape_chunks(tvi_p)
        tvm_chunks = reshape_chunks(tvm_p)

        initial_slots = jnp.broadcast_to(
            self.slot_seed[None, :, :],
            (batch_size, self.config.num_slots, self.config.d_model),
        )
        initial_graph = self.graph_memory.init_state(batch_size)

        def process_chunk(
            carry: tuple[jnp.ndarray, GraphMemoryState],
            xs: tuple,
        ) -> tuple[tuple[jnp.ndarray, GraphMemoryState], None]:
            slots, graph_state = carry
            (
                chunk_hidden,
                chunk_em,
                chunk_si,
                chunk_sm,
                chunk_tsi,
                chunk_tsm,
                chunk_tvi,
                chunk_tvm,
            ) = xs

            evidence = chunk_hidden.mean(axis=1)
            marker = jnp.max(chunk_em, axis=1)
            src = chunk_si[:, 0]
            src_m = jnp.max(chunk_sm, axis=1).astype(jnp.bool_)
            tgt_sym = chunk_tsi[:, 0]
            tgt_sym_m = jnp.max(chunk_tsm, axis=1).astype(jnp.bool_)
            tgt_val = chunk_tvi[:, 0]
            tgt_val_m = jnp.max(chunk_tvm, axis=1).astype(jnp.bool_)

            graph_state = self.graph_memory.update(
                graph_state, evidence, marker, src, src_m, tgt_sym, tgt_sym_m, tgt_val, tgt_val_m
            )
            _, graph_feedback = self.graph_memory(graph_state, src, src_m)

            predicted_slots = self.slot_dynamics(slots)
            candidate_slots, gates = self.slot_writer(predicted_slots, evidence, graph_feedback)
            refined_slots = self._refine(candidate_slots, predicted_slots, evidence, gates)
            mixed_slots = self.relation_mixer(refined_slots)
            return (mixed_slots, graph_state), None

        (final_slots, final_graph), _ = jax.lax.scan(
            process_chunk,
            (initial_slots, initial_graph),
            (
                jnp.swapaxes(hidden_chunks, 0, 1),
                jnp.swapaxes(em_chunks, 0, 1),
                jnp.swapaxes(si_chunks, 0, 1),
                jnp.swapaxes(sm_chunks, 0, 1),
                jnp.swapaxes(tsi_chunks, 0, 1),
                jnp.swapaxes(tsm_chunks, 0, 1),
                jnp.swapaxes(tvi_chunks, 0, 1),
                jnp.swapaxes(tvm_chunks, 0, 1),
            ),
        )

        lm_logits = self.lm_head(hidden[:, :seq_len, :])
        answer_context = hidden[jnp.arange(batch_size), jnp.clip(lengths - 1, 0, seq_len - 1)]
        graph_logits, graph_feedback = self.graph_memory(final_graph, query_idx, query_mask)
        slot_scores = jnp.einsum("bsd,bd->bs", final_slots, answer_context) / jnp.sqrt(float(self.config.d_model))
        slot_readout = jnp.einsum("bs,bsd->bd", jax.nn.softmax(slot_scores, axis=-1), final_slots)
        answer_logits = self.answer_head(
            jnp.concatenate([answer_context, slot_readout, graph_feedback], axis=-1)
        ) + 2.0 * graph_logits

        return CPMMOutputs(
            lm_logits=lm_logits,
            answer_logits=answer_logits,
            graph_logits=graph_logits,
            map_memory=final_graph.map_memory,
            step_memory=final_graph.step_memory,
            slot_state=final_slots,
        )
