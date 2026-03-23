"""Models for the post-transformer prototype."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .data import (
    ALIAS_START,
    ASK,
    BRIDGE_START,
    KEY_START,
    MAP,
    NOISE_START,
    NUM_KEYS,
    NUM_VALUES,
    PAD,
    PASS,
    SEP,
    STEP,
    TEXT_ALIAS,
    TEXT_ANSWER,
    TEXT_BRIDGE,
    TEXT_EOS,
    TEXT_ENTITY,
    TEXT_LINKS,
    TEXT_MAPS,
    TEXT_NAME_ALIAS_START,
    TEXT_NAME_BRIDGE_START,
    TEXT_NAME_KEY_START,
    TEXT_NAME_VALUE_START,
    TEXT_PAD,
    TEXT_PERIOD,
    TEXT_QMARK,
    TEXT_REFERS,
    TEXT_TO,
    TEXT_VALUE,
    TEXT_VOCAB_SIZE,
    VALUE_START,
    VOCAB_SIZE,
)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (tokens * weights).sum(dim=1) / denom


class TypedGraphMemory(nn.Module):
    """Differentiable typed graph memory for symbolic event storage."""

    def __init__(self, d_model: int, num_keys: int, num_values: int) -> None:
        super().__init__()
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_symbols = num_keys * 2
        self.symbol_embedding = nn.Embedding(self.num_symbols, d_model)
        self.value_embedding = nn.Embedding(num_values, d_model)
        self.map_gain = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.step_gain = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.slot_feedback = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.output_head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, num_values))
        self.graph_steps = 3

    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "map_memory": torch.zeros(batch_size, self.num_symbols, self.num_values, device=device),
            "step_memory": torch.zeros(batch_size, self.num_symbols, self.num_symbols, device=device),
        }

    def update(
        self,
        state: Dict[str, torch.Tensor],
        event_marker: torch.Tensor,
        source_idx: torch.Tensor,
        source_valid: torch.Tensor,
        target_symbol_idx: torch.Tensor,
        target_symbol_valid: torch.Tensor,
        target_value_idx: torch.Tensor,
        target_value_valid: torch.Tensor,
        evidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = event_marker.size(0)
        src_onehot = F.one_hot(source_idx.clamp(min=0, max=self.num_symbols - 1), num_classes=self.num_symbols).float()
        tgt_symbol = F.one_hot(
            target_symbol_idx.clamp(min=0, max=self.num_symbols - 1),
            num_classes=self.num_symbols,
        ).float()
        tgt_value = F.one_hot(
            target_value_idx.clamp(min=0, max=self.num_values - 1),
            num_classes=self.num_values,
        ).float()

        map_gain = torch.sigmoid(self.map_gain(evidence)).view(batch_size, 1, 1)
        step_gain = torch.sigmoid(self.step_gain(evidence)).view(batch_size, 1, 1)

        map_mask = (event_marker.eq(PASS) | event_marker.eq(MAP)) & source_valid & target_value_valid
        step_mask = event_marker.eq(STEP) & source_valid & target_symbol_valid

        map_update = torch.einsum("bs,bv->bsv", src_onehot, tgt_value) * map_gain
        step_update = torch.einsum("bs,bt->bst", src_onehot, tgt_symbol) * step_gain

        state["map_memory"] = state["map_memory"] + map_update * map_mask.view(batch_size, 1, 1)
        state["step_memory"] = state["step_memory"] + step_update * step_mask.view(batch_size, 1, 1)
        return state

    def query_readout(
        self,
        state: Dict[str, torch.Tensor],
        query_idx: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query_idx.size(0)
        walk = F.one_hot(query_idx.clamp(min=0, max=self.num_symbols - 1), num_classes=self.num_symbols).float()
        walk = walk * query_valid.unsqueeze(-1)
        symbol_bank = self.symbol_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        value_bank = self.value_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        accumulated_values = torch.einsum("bs,bsv->bv", walk, state["map_memory"])
        accumulated_symbols = torch.einsum("bs,bsd->bd", walk, symbol_bank)
        for _ in range(self.graph_steps):
            walk = torch.einsum("bs,bst->bt", walk, state["step_memory"])
            accumulated_values = accumulated_values + torch.einsum("bs,bsv->bv", walk, state["map_memory"])
            accumulated_symbols = accumulated_symbols + torch.einsum("bs,bsd->bd", walk, symbol_bank)

        graph_value_vec = torch.einsum("bv,bvd->bd", accumulated_values, value_bank)
        graph_state = torch.cat([accumulated_symbols, graph_value_vec], dim=-1)
        logits = self.output_head(graph_state)
        return logits, self.slot_feedback(graph_state)


class CausalPredictiveMemoryMachine(nn.Module):
    """Chunked slot-memory model with an energy-refinement loop."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        num_slots: int = 6,
        chunk_size: int = 16,
        refinement_steps: int = 2,
        refine_step_size: float = 0.3,
        use_relations: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.chunk_size = chunk_size
        self.refinement_steps = refinement_steps
        self.refine_step_size = refine_step_size
        self.use_relations = use_relations

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.type_embedding = nn.Embedding(6, d_model)
        self.index_embedding = nn.Embedding(NUM_KEYS + 1, d_model)
        self.graph_memory = TypedGraphMemory(d_model, NUM_KEYS, NUM_VALUES)
        self.slot_seed = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.dynamics = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.evidence_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        self.write_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.obs_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_proj = nn.Linear(d_model, d_model)
        self.event_key_proj = nn.Linear(d_model, d_model)
        self.slot_key_proj = nn.Linear(d_model, d_model)
        self.query_update = nn.GRUCell(d_model, d_model)
        self.read_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, NUM_VALUES),
        )
        if use_relations:
            self.relation_mixer = nn.MultiheadAttention(
                d_model,
                num_heads=4,
                dropout=0.0,
                batch_first=True,
            )
            self.relation_norm = nn.LayerNorm(d_model)
        self.slot_norm = nn.LayerNorm(d_model)

        self.lambda_obs = 1.0
        self.lambda_dyn = 0.35
        self.lambda_sparse = 0.02
        self.query_steps = 2
        self.graph_scale = 2.5

    def initial_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.slot_seed.expand(batch_size, -1, -1).to(device)

    def symbol_index(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.zeros_like(tokens)
        valid = torch.zeros_like(tokens, dtype=torch.bool)

        key_mask = (tokens >= KEY_START) & (tokens < ALIAS_START)
        alias_mask = (tokens >= ALIAS_START) & (tokens < VALUE_START)
        bridge_mask = (tokens >= BRIDGE_START) & (tokens < NOISE_START)

        indices = torch.where(key_mask, tokens - KEY_START, indices)
        indices = torch.where(alias_mask, tokens - ALIAS_START, indices)
        indices = torch.where(bridge_mask, NUM_KEYS + (tokens - BRIDGE_START), indices)
        valid = key_mask | alias_mask | bridge_mask
        return indices, valid

    def value_index(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        value_mask = (tokens >= VALUE_START) & (tokens < BRIDGE_START)
        indices = torch.where(value_mask, tokens - VALUE_START, torch.zeros_like(tokens))
        return indices, value_mask

    def factorized_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        base = self.embedding(tokens)
        type_ids = torch.full_like(tokens, 5)
        index_ids = torch.zeros_like(tokens)

        type_ids = torch.where(tokens.eq(PAD), torch.zeros_like(type_ids), type_ids)
        special_mask = (
            tokens.eq(PASS)
            | tokens.eq(MAP)
            | tokens.eq(STEP)
            | tokens.eq(ASK)
            | tokens.eq(SEP)
        )
        type_ids = torch.where(special_mask, torch.ones_like(type_ids), type_ids)

        key_mask = (tokens >= KEY_START) & (tokens < ALIAS_START)
        alias_mask = (tokens >= ALIAS_START) & (tokens < VALUE_START)
        value_mask = (tokens >= VALUE_START) & (tokens < BRIDGE_START)
        bridge_mask = (tokens >= BRIDGE_START) & (tokens < NOISE_START)

        type_ids = torch.where(key_mask | alias_mask, torch.full_like(type_ids, 2), type_ids)
        type_ids = torch.where(value_mask, torch.full_like(type_ids, 3), type_ids)
        type_ids = torch.where(bridge_mask, torch.full_like(type_ids, 4), type_ids)

        index_ids = torch.where(key_mask, tokens - KEY_START + 1, index_ids)
        index_ids = torch.where(alias_mask, tokens - ALIAS_START + 1, index_ids)
        index_ids = torch.where(value_mask, tokens - VALUE_START + 1, index_ids)
        index_ids = torch.where(bridge_mask, tokens - BRIDGE_START + 1, index_ids)

        return base + self.type_embedding(type_ids) + self.index_embedding(index_ids)

    def encode_events(
        self,
        chunk_tokens: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if chunk_tokens.size(1) < 3:
            empty = chunk_tokens.new_zeros((chunk_tokens.size(0), 0), dtype=torch.bool)
            return self.embedding(chunk_tokens[:, :0]), empty
        windows = torch.stack(
            [chunk_tokens[:, :-2], chunk_tokens[:, 1:-1], chunk_tokens[:, 2:]],
            dim=2,
        )
        valid_mask = (
            chunk_mask[:, :-2]
            & chunk_mask[:, 1:-1]
            & chunk_mask[:, 2:]
            & (
                windows[:, :, 0].eq(PASS)
                | windows[:, :, 0].eq(MAP)
                | windows[:, :, 0].eq(STEP)
            )
        )
        window_emb = self.factorized_embed(windows)
        event_vec = self.event_encoder(window_emb.reshape(window_emb.size(0), window_emb.size(1), -1))
        return event_vec, valid_mask

    def compute_energy(
        self,
        state: torch.Tensor,
        predicted_state: torch.Tensor,
        evidence: torch.Tensor,
        gates: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        slot_summary = state.mean(dim=1)
        reconstructed = self.obs_decoder(slot_summary)
        e_obs = F.mse_loss(reconstructed, evidence)
        e_dyn = F.mse_loss(state, predicted_state)
        e_sparse = gates.mean()
        energy = (
            self.lambda_obs * e_obs
            + self.lambda_dyn * e_dyn
            + self.lambda_sparse * e_sparse
        )
        stats = {
            "obs": e_obs.detach(),
            "dyn": e_dyn.detach(),
            "sparse": e_sparse.detach(),
        }
        return energy, stats

    def refine_state(
        self,
        candidate_state: torch.Tensor,
        predicted_state: torch.Tensor,
        evidence: torch.Tensor,
        gates: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        state = candidate_state
        stats: Dict[str, float] = {"energy_obs": 0.0, "energy_dyn": 0.0, "energy_sparse": 0.0}
        with torch.enable_grad():
            for _ in range(self.refinement_steps):
                state = state.detach().requires_grad_(True)
                energy, energy_stats = self.compute_energy(state, predicted_state, evidence, gates)
                grad = torch.autograd.grad(energy, state, create_graph=self.training)[0]
                state = state - self.refine_step_size * grad
                stats["energy_obs"] = float(energy_stats["obs"])
                stats["energy_dyn"] = float(energy_stats["dyn"])
                stats["energy_sparse"] = float(energy_stats["sparse"])
        return state, stats

    def relation_update(self, slots: torch.Tensor) -> torch.Tensor:
        if not self.use_relations:
            return slots
        mixed, _ = self.relation_mixer(slots, slots, slots, need_weights=False)
        return self.relation_norm(slots + mixed)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        query_tokens: torch.Tensor,
        query_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, total_len = context_tokens.shape
        device = context_tokens.device
        slots = self.initial_slots(batch_size, device)
        graph_state = self.graph_memory.init_state(batch_size, device)
        diagnostics: Dict[str, float] = {}

        for start in range(0, total_len, self.chunk_size):
            end = min(total_len, start + self.chunk_size)
            chunk = context_tokens[:, start:end]
            positions = torch.arange(start, end, device=device).unsqueeze(0)
            chunk_mask = positions < context_lengths.unsqueeze(1)
            if not torch.any(chunk_mask):
                continue
            event_vectors, event_mask = self.encode_events(chunk, chunk_mask)
            if event_vectors.size(1) == 0:
                continue
            for event_index in range(event_vectors.size(1)):
                if not torch.any(event_mask[:, event_index]):
                    continue
                evidence = event_vectors[:, event_index]
                event_window = chunk[:, event_index : event_index + 3]
                event_marker = event_window[:, 0]
                source_idx, source_valid = self.symbol_index(event_window[:, 1])
                target_symbol_idx, target_symbol_valid = self.symbol_index(event_window[:, 2])
                target_value_idx, target_value_valid = self.value_index(event_window[:, 2])
                graph_state = self.graph_memory.update(
                    graph_state,
                    event_marker,
                    source_idx,
                    source_valid,
                    target_symbol_idx,
                    target_symbol_valid,
                    target_value_idx,
                    target_value_valid,
                    evidence,
                )
                predicted_slots = self.slot_norm(slots + self.dynamics(slots))
                query_feedback_logits, graph_feedback = self.graph_memory.query_readout(
                    graph_state,
                    source_idx,
                    source_valid,
                )
                evidence_slots = self.evidence_proj(evidence).unsqueeze(1).expand(-1, self.num_slots, -1)
                graph_feedback_slots = graph_feedback.unsqueeze(1).expand(-1, self.num_slots, -1)
                write_input = torch.cat([predicted_slots, evidence_slots + graph_feedback_slots], dim=-1)
                slot_scores = torch.einsum(
                    "bsd,bd->bs",
                    self.slot_key_proj(predicted_slots),
                    self.event_key_proj(evidence),
                ) / math.sqrt(self.d_model)
                slot_attn = slot_scores.softmax(dim=-1).unsqueeze(-1)
                gates = torch.sigmoid(self.gate_proj(write_input)) * slot_attn
                write_delta = torch.tanh(self.write_proj(write_input))
                candidate_slots = predicted_slots + gates * write_delta
                if self.refinement_steps > 0:
                    candidate_slots, diagnostics = self.refine_state(
                        candidate_slots,
                        predicted_slots,
                        evidence,
                        gates,
                    )
                slots = self.relation_update(candidate_slots)
                diagnostics["graph_feedback_mean"] = float(
                    query_feedback_logits.softmax(dim=-1).max(dim=-1).values.mean().detach()
                )

        query_emb = self.factorized_embed(query_tokens)
        query_mask = torch.arange(query_tokens.size(1), device=device).unsqueeze(0) < query_lengths.unsqueeze(1)
        query_state = self.query_proj(masked_mean(query_emb, query_mask))
        readout = slots.mean(dim=1)
        for _ in range(self.query_steps):
            scores = torch.einsum("bsd,bd->bs", self.slot_key_proj(slots), query_state) / math.sqrt(self.d_model)
            attn = scores.softmax(dim=-1)
            readout = torch.einsum("bs,bsd->bd", attn, slots)
            query_state = self.query_update(readout, query_state)
        pooled_slots = slots.mean(dim=1)
        last_query_positions = (query_lengths - 1).clamp(min=0)
        query_symbols = query_tokens[torch.arange(batch_size, device=device), last_query_positions]
        query_idx, query_valid = self.symbol_index(query_symbols)
        graph_logits, graph_feedback = self.graph_memory.query_readout(graph_state, query_idx, query_valid)
        pooled_slots = pooled_slots + graph_feedback
        logits = self.read_proj(torch.cat([query_state, readout, pooled_slots], dim=-1))
        logits = logits + self.graph_scale * graph_logits
        return logits, diagnostics


class TransformerBaseline(nn.Module):
    """Small decoder-style transformer classifier."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, NUM_VALUES)

    def ensure_position_capacity(self, seq_len: int, device: torch.device) -> None:
        current_capacity = self.position.num_embeddings
        if seq_len <= current_capacity:
            return
        new_capacity = max(seq_len, current_capacity * 2)
        new_position = nn.Embedding(new_capacity, self.d_model, device=device)
        with torch.no_grad():
            new_position.weight[:current_capacity] = self.position.weight.data
            nn.init.normal_(new_position.weight[current_capacity:], mean=0.0, std=0.02)
        self.position = new_position

    def forward(self, sequence_tokens: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = sequence_tokens.shape
        self.ensure_position_capacity(seq_len, sequence_tokens.device)
        positions = torch.arange(seq_len, device=sequence_tokens.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.embedding(sequence_tokens) + self.position(positions)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=sequence_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = sequence_tokens.eq(PAD)
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)
        last_indices = (sequence_lengths - 1).clamp(min=0)
        final = hidden[torch.arange(batch_size, device=sequence_tokens.device), last_indices]
        return self.classifier(final)


class GRUBaseline(nn.Module):
    """Simple recurrent baseline for long-context comparison."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(d_model, NUM_VALUES)

    def forward(self, sequence_tokens: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequence_tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.classifier(hidden[-1])


class GraphTextLanguageModel(nn.Module):
    """Hybrid LM with next-token, answer, and graph-memory objectives."""

    def __init__(
        self,
        vocab_size: int = TEXT_VOCAB_SIZE,
        d_model: int = 96,
        num_slots: int = 4,
        refinement_steps: int = 1,
        use_relations: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_slots = num_slots
        self.refinement_steps = refinement_steps
        self.use_relations = use_relations

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=TEXT_PAD)
        self.position = nn.Embedding(2048, d_model)
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.graph_memory = TypedGraphMemory(d_model, NUM_KEYS, NUM_VALUES)
        self.slot_seed = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.slot_norm = nn.LayerNorm(d_model)
        self.dynamics = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        self.evidence_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.write_proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.gate_proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.event_key_proj = nn.Linear(d_model, d_model)
        self.slot_key_proj = nn.Linear(d_model, d_model)
        self.obs_decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.answer_proj = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, NUM_VALUES))
        self.graph_scale = 2.0
        self.lambda_obs = 1.0
        self.lambda_dyn = 0.25
        self.lambda_sparse = 0.02

        if use_relations:
            self.relation_mixer = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.0, batch_first=True)
            self.relation_norm = nn.LayerNorm(d_model)

    def ensure_position_capacity(self, seq_len: int, device: torch.device) -> None:
        current_capacity = self.position.num_embeddings
        if seq_len <= current_capacity:
            return
        new_capacity = max(seq_len, current_capacity * 2)
        new_position = nn.Embedding(new_capacity, self.d_model, device=device)
        with torch.no_grad():
            new_position.weight[:current_capacity] = self.position.weight.data
            nn.init.normal_(new_position.weight[current_capacity:], mean=0.0, std=0.02)
        self.position = new_position

    def name_to_symbol(self, name_token: int, type_token: int) -> tuple[int, bool]:
        if type_token == TEXT_ENTITY and TEXT_NAME_KEY_START <= name_token < TEXT_NAME_ALIAS_START:
            return name_token - TEXT_NAME_KEY_START, True
        if type_token == TEXT_ALIAS and TEXT_NAME_ALIAS_START <= name_token < TEXT_NAME_BRIDGE_START:
            return name_token - TEXT_NAME_ALIAS_START, True
        if type_token == TEXT_BRIDGE and TEXT_NAME_BRIDGE_START <= name_token < TEXT_NAME_VALUE_START:
            return NUM_KEYS + (name_token - TEXT_NAME_BRIDGE_START), True
        return 0, False

    def name_to_value(self, name_token: int) -> tuple[int, bool]:
        if TEXT_NAME_VALUE_START <= name_token < TEXT_NAME_VALUE_START + NUM_VALUES:
            return name_token - TEXT_NAME_VALUE_START, True
        return 0, False

    def parse_events(self, tokens: torch.Tensor, lengths: torch.Tensor) -> list[list[dict[str, int]]]:
        parsed: list[list[dict[str, int]]] = []
        for row in range(tokens.size(0)):
            seq_len = int(lengths[row].item())
            seq = tokens[row, :seq_len].tolist()
            row_events: list[dict[str, int]] = []
            for start in range(max(0, seq_len - 6)):
                window = seq[start : start + 7]
                if len(window) < 7:
                    continue
                if window[0] in {TEXT_ENTITY, TEXT_BRIDGE} and window[2] == TEXT_MAPS and window[3] == TEXT_TO and window[4] == TEXT_VALUE and window[6] == TEXT_PERIOD:
                    source_idx, source_valid = self.name_to_symbol(window[1], window[0])
                    value_idx, value_valid = self.name_to_value(window[5])
                    if source_valid and value_valid:
                        row_events.append(
                            {
                                "start": start,
                                "end": start + 7,
                                "marker": PASS,
                                "source_idx": source_idx,
                                "source_valid": 1,
                                "target_symbol_idx": 0,
                                "target_symbol_valid": 0,
                                "target_value_idx": value_idx,
                                "target_value_valid": 1,
                            }
                        )
                elif window[0] == TEXT_ALIAS and window[2] == TEXT_REFERS and window[3] == TEXT_TO and window[4] == TEXT_ENTITY and window[6] == TEXT_PERIOD:
                    source_idx, source_valid = self.name_to_symbol(window[1], window[0])
                    target_idx, target_valid = self.name_to_symbol(window[5], window[4])
                    if source_valid and target_valid:
                        row_events.append(
                            {
                                "start": start,
                                "end": start + 7,
                                "marker": STEP,
                                "source_idx": source_idx,
                                "source_valid": 1,
                                "target_symbol_idx": target_idx,
                                "target_symbol_valid": 1,
                                "target_value_idx": 0,
                                "target_value_valid": 0,
                            }
                        )
                elif window[0] in {TEXT_ENTITY, TEXT_BRIDGE} and window[2] == TEXT_LINKS and window[3] == TEXT_TO and window[4] in {TEXT_ENTITY, TEXT_BRIDGE} and window[6] == TEXT_PERIOD:
                    source_idx, source_valid = self.name_to_symbol(window[1], window[0])
                    target_idx, target_valid = self.name_to_symbol(window[5], window[4])
                    if source_valid and target_valid:
                        row_events.append(
                            {
                                "start": start,
                                "end": start + 7,
                                "marker": STEP,
                                "source_idx": source_idx,
                                "source_valid": 1,
                                "target_symbol_idx": target_idx,
                                "target_symbol_valid": 1,
                                "target_value_idx": 0,
                                "target_value_valid": 0,
                            }
                        )
            parsed.append(row_events)
        return parsed

    def extract_query_symbol(self, tokens: torch.Tensor, answer_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = tokens.size(0)
        query_idx = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
        query_valid = torch.zeros(batch_size, dtype=torch.bool, device=tokens.device)
        for row in range(batch_size):
            pos = int(answer_positions[row].item())
            if pos < 4:
                continue
            name_token = int(tokens[row, pos - 2].item())
            type_token = int(tokens[row, pos - 3].item())
            idx, valid = self.name_to_symbol(name_token, type_token)
            query_idx[row] = idx
            query_valid[row] = valid
        return query_idx, query_valid

    def compute_energy(
        self,
        state: torch.Tensor,
        predicted_state: torch.Tensor,
        evidence: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        slot_summary = state.mean(dim=1)
        reconstructed = self.obs_decoder(slot_summary)
        e_obs = F.mse_loss(reconstructed, evidence)
        e_dyn = F.mse_loss(state, predicted_state)
        e_sparse = gates.mean()
        return self.lambda_obs * e_obs + self.lambda_dyn * e_dyn + self.lambda_sparse * e_sparse

    def relation_update(self, slots: torch.Tensor) -> torch.Tensor:
        if not self.use_relations:
            return slots
        mixed, _ = self.relation_mixer(slots, slots, slots, need_weights=False)
        return self.relation_norm(slots + mixed)

    def forward(
        self,
        input_tokens: torch.Tensor,
        lengths: torch.Tensor,
        answer_positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device
        self.ensure_position_capacity(seq_len, device)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_hidden, _ = self.encoder(self.embedding(input_tokens) + self.position(positions))
        lm_logits = self.lm_head(token_hidden)

        parsed_events = self.parse_events(input_tokens, lengths)
        graph_state = self.graph_memory.init_state(batch_size, device)
        map_state = []
        step_state = []
        slot_bank = []
        for row in range(batch_size):
            slots = self.slot_seed.expand(1, -1, -1).to(device)
            single_graph = self.graph_memory.init_state(1, device)
            for event in parsed_events[row]:
                evidence = token_hidden[row : row + 1, event["start"] : event["end"]].mean(dim=1)
                single_graph = self.graph_memory.update(
                    single_graph,
                    torch.tensor([event["marker"]], device=device),
                    torch.tensor([event["source_idx"]], device=device),
                    torch.tensor([bool(event["source_valid"])], dtype=torch.bool, device=device),
                    torch.tensor([event["target_symbol_idx"]], device=device),
                    torch.tensor([bool(event["target_symbol_valid"])], dtype=torch.bool, device=device),
                    torch.tensor([event["target_value_idx"]], device=device),
                    torch.tensor([bool(event["target_value_valid"])], dtype=torch.bool, device=device),
                    evidence,
                )
                _, graph_feedback = self.graph_memory.query_readout(
                    single_graph,
                    torch.tensor([event["source_idx"]], device=device),
                    torch.tensor([bool(event["source_valid"])], dtype=torch.bool, device=device),
                )
                predicted_slots = self.slot_norm(slots + self.dynamics(slots))
                evidence_slots = self.evidence_proj(evidence).unsqueeze(1).expand(-1, self.num_slots, -1)
                graph_feedback_slots = graph_feedback.unsqueeze(1).expand(-1, self.num_slots, -1)
                write_input = torch.cat([predicted_slots, evidence_slots + graph_feedback_slots], dim=-1)
                slot_scores = torch.einsum(
                    "bsd,bd->bs",
                    self.slot_key_proj(predicted_slots),
                    self.event_key_proj(evidence),
                ) / math.sqrt(self.d_model)
                gates = torch.sigmoid(self.gate_proj(write_input)) * slot_scores.softmax(dim=-1).unsqueeze(-1)
                candidate_slots = predicted_slots + gates * torch.tanh(self.write_proj(write_input))
                if self.refinement_steps > 0:
                    with torch.enable_grad():
                        candidate_slots = candidate_slots.detach().requires_grad_(True)
                        energy = self.compute_energy(candidate_slots, predicted_slots.detach(), evidence.detach(), gates.detach())
                        grad = torch.autograd.grad(energy, candidate_slots, create_graph=self.training)[0]
                        candidate_slots = candidate_slots - 0.25 * grad
                slots = self.relation_update(candidate_slots)
            map_state.append(single_graph["map_memory"][0])
            step_state.append(single_graph["step_memory"][0])
            slot_bank.append(slots[0])
            graph_state["map_memory"][row] = single_graph["map_memory"][0]
            graph_state["step_memory"][row] = single_graph["step_memory"][0]

        final_slots = torch.stack(slot_bank, dim=0)
        map_memory = torch.stack(map_state, dim=0)
        step_memory = torch.stack(step_state, dim=0)

        answer_context = token_hidden[torch.arange(batch_size, device=device), answer_positions]
        query_idx, query_valid = self.extract_query_symbol(input_tokens, answer_positions)
        graph_logits, graph_feedback = self.graph_memory.query_readout(graph_state, query_idx, query_valid)
        slot_scores = torch.einsum("bsd,bd->bs", self.slot_key_proj(final_slots), answer_context) / math.sqrt(self.d_model)
        slot_readout = torch.einsum("bs,bsd->bd", slot_scores.softmax(dim=-1), final_slots)
        answer_logits = self.answer_proj(torch.cat([answer_context, slot_readout, graph_feedback], dim=-1))
        answer_logits = answer_logits + self.graph_scale * graph_logits

        return {
            "lm_logits": lm_logits,
            "answer_logits": answer_logits,
            "graph_logits": graph_logits,
            "map_memory": map_memory,
            "step_memory": step_memory,
        }
