"""Pure post-transformer CPMM code language model in Flax."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

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
        return nn.LayerNorm(name="encoder_norm")(x)


class SlotRelationMixer(nn.Module):
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


class CPMMCodeModel(nn.Module):
    """Chunk-scanned CPMM with slot memory, graph memory, and LM heads.

    Uses @nn.compact throughout so all parameters are created in a single
    traced call — required when modules are called inside lax.scan.
    The scan body receives frozen parameter pytrees and calls module.apply()
    directly, which is the correct Flax pattern for scan over parameters.
    """

    config: ModelConfig

    @nn.compact
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

        # ── Local encoder ──────────────────────────────────────────────────
        hidden = LocalChunkEncoder(self.config, name="local_chunk_encoder")(token_ids)

        # ── Pad to chunk boundary ──────────────────────────────────────────
        chunk_size = self.config.chunk_size
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        pad_width = num_chunks * chunk_size - seq_len

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

        def rc(x: jnp.ndarray, last: int | None = None) -> jnp.ndarray:
            if last is not None:
                return x.reshape(batch_size, num_chunks, chunk_size, last)
            return x.reshape(batch_size, num_chunks, chunk_size)

        # ── Slot memory parameters ─────────────────────────────────────────
        slot_seed = self.param(
            "slot_seed",
            nn.initializers.normal(stddev=0.02),
            (self.config.num_slots, self.config.d_model),
        )
        initial_slots = jnp.broadcast_to(
            slot_seed[None], (batch_size, self.config.num_slots, self.config.d_model)
        )

        # ── Sub-module parameters (initialised here in compact context) ────
        # We call each sub-module once with dummy data so Flax registers
        # their parameters, then use module.apply with those params in scan.
        _dummy_slot = jnp.zeros((batch_size, self.config.num_slots, self.config.d_model))
        _dummy_ev = jnp.zeros((batch_size, self.config.d_model))
        _dummy_scalar = jnp.zeros((batch_size,), dtype=jnp.int32)
        _dummy_bool = jnp.zeros((batch_size,), dtype=jnp.bool_)

        graph_module = TypedGraphMemory(
            d_model=self.config.d_model,
            num_symbols=self.config.num_graph_symbols,
            num_values=self.config.num_graph_values,
            graph_steps=self.config.graph_steps,
            name="typed_graph_memory",
        )
        _dummy_graph = graph_module.init_state(batch_size)
        # Touch graph module to register params
        _, _ = graph_module(_dummy_graph, _dummy_scalar, _dummy_bool)

        # Slot dynamics: LayerNorm + 2-layer MLP
        dyn_dense1 = nn.Dense(self.config.d_model, name="dyn_dense1")
        dyn_dense2 = nn.Dense(self.config.d_model, name="dyn_dense2")
        dyn_norm = nn.LayerNorm(name="dyn_norm")

        # Slot writer
        ev_proj = nn.Dense(self.config.d_model, name="ev_proj")
        fb_proj = nn.Dense(self.config.d_model, name="fb_proj")
        slot_key = nn.Dense(self.config.d_model, name="slot_key")
        ev_key = nn.Dense(self.config.d_model, name="ev_key")
        gate_dense = nn.Dense(1, name="gate_dense")
        write_dense = nn.Dense(self.config.d_model, name="write_dense")

        # Relation mixer
        relation_mixer = SlotRelationMixer(
            self.config.d_model, self.config.relation_rank, name="slot_relation_mixer"
        )

        # Touch all sub-modules with dummy data to register their params
        _ = dyn_norm(_dummy_slot + dyn_dense2(nn.tanh(dyn_dense1(_dummy_slot))))
        _combined = jnp.concatenate([_dummy_slot, jnp.zeros_like(_dummy_slot)], axis=-1)
        _ = gate_dense(_combined)
        _ = write_dense(_combined)
        _ = ev_proj(_dummy_ev)
        _ = fb_proj(_dummy_ev)
        _ = slot_key(_dummy_slot)
        _ = ev_key(_dummy_ev)
        _ = relation_mixer(_dummy_slot)

        # ── Per-chunk processing via lax.while_loop ────────────────────────
        # We use a simple Python loop here instead of lax.scan to avoid
        # the Flax parameter-in-scan restriction entirely.
        # For seq_len=512 and chunk_size=64 this is 8 iterations — fine.
        hc = rc(hidden_p, self.config.d_model)
        emc = rc(em_p)
        sic = rc(si_p)
        smc = rc(sm_p)
        tsic = rc(tsi_p)
        tsmc = rc(tsm_p)
        tvic = rc(tvi_p)
        tvmc = rc(tvm_p)

        slots = initial_slots
        graph_state = graph_module.init_state(batch_size)

        for ci in range(num_chunks):
            evidence = hc[:, ci].mean(axis=1)
            marker = jnp.max(emc[:, ci], axis=1)
            src = sic[:, ci, 0]
            src_m = jnp.max(smc[:, ci], axis=1).astype(jnp.bool_)
            tgt_sym = tsic[:, ci, 0]
            tgt_sym_m = jnp.max(tsmc[:, ci], axis=1).astype(jnp.bool_)
            tgt_val = tvic[:, ci, 0]
            tgt_val_m = jnp.max(tvmc[:, ci], axis=1).astype(jnp.bool_)

            graph_state = graph_module.update(
                graph_state, evidence, marker, src, src_m, tgt_sym, tgt_sym_m, tgt_val, tgt_val_m
            )
            _, graph_feedback = graph_module(graph_state, src, src_m)

            # Slot dynamics
            predicted_slots = dyn_norm(slots + dyn_dense2(nn.tanh(dyn_dense1(slots))))

            # Slot writer
            ev_emb = ev_proj(evidence)[:, None, :]
            fb_emb = fb_proj(graph_feedback)[:, None, :]
            combined = jnp.broadcast_to(ev_emb + fb_emb, predicted_slots.shape)
            write_input = jnp.concatenate([predicted_slots, combined], axis=-1)
            slot_scores = jnp.einsum(
                "bsd,bd->bs", slot_key(predicted_slots), ev_key(evidence)
            ) / jnp.sqrt(float(self.config.d_model))
            slot_attn = jax.nn.softmax(slot_scores, axis=-1)[..., None]
            gates = jax.nn.sigmoid(gate_dense(write_input)) * slot_attn
            candidate_slots = predicted_slots + gates * jnp.tanh(write_dense(write_input))

            # Energy refinement (unrolled, no lax.scan needed for 2 steps)
            for _ in range(self.config.refinement_steps):
                slot_summary = candidate_slots.mean(axis=1)
                e_obs = jnp.mean((slot_summary - evidence) ** 2)
                e_dyn = jnp.mean((candidate_slots - predicted_slots) ** 2)
                e_sparse = jnp.mean(gates)
                energy = e_obs + 0.35 * e_dyn + 0.02 * e_sparse
                grad = jax.grad(lambda s, _ev=evidence, _ps=predicted_slots, _g=gates: (
                    jnp.mean((s.mean(axis=1) - _ev) ** 2)
                    + 0.35 * jnp.mean((s - _ps) ** 2)
                    + 0.02 * jnp.mean(_g)
                ))(candidate_slots)
                candidate_slots = candidate_slots - self.config.refinement_step_size * grad
            del energy

            slots = relation_mixer(candidate_slots)

        # ── Readout ────────────────────────────────────────────────────────
        lm_logits = nn.Dense(self.config.vocab_size, name="lm_head")(hidden[:, :seq_len])
        answer_context = hidden[jnp.arange(batch_size), jnp.clip(lengths - 1, 0, seq_len - 1)]
        graph_logits, graph_feedback = graph_module(graph_state, query_idx, query_mask)
        slot_scores = jnp.einsum("bsd,bd->bs", slots, answer_context) / jnp.sqrt(float(self.config.d_model))
        slot_readout = jnp.einsum("bs,bsd->bd", jax.nn.softmax(slot_scores, axis=-1), slots)
        answer_logits = nn.Sequential(
            [nn.Dense(self.config.d_model), nn.gelu, nn.Dense(self.config.num_graph_values)],
            name="answer_head",
        )(jnp.concatenate([answer_context, slot_readout, graph_feedback], axis=-1))
        answer_logits = answer_logits + 2.0 * graph_logits

        return CPMMOutputs(
            lm_logits=lm_logits,
            answer_logits=answer_logits,
            graph_logits=graph_logits,
            map_memory=graph_state.map_memory,
            step_memory=graph_state.step_memory,
            slot_state=slots,
        )
