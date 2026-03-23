"""Python code preprocessing and resumable shard loading for CPMM JAX."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
import glob
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Iterator, Protocol

import numpy as np

from .config import CodeDataConfig


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|->|:=|[^\s]")


@dataclass
class LoaderState:
    shard_index: int = 0
    sample_offset: int = 0
    epoch: int = 0
    rng_seed: int = 0


@dataclass
class GraphTargets:
    event_markers: np.ndarray
    source_idx: np.ndarray
    source_mask: np.ndarray
    target_symbol_idx: np.ndarray
    target_symbol_mask: np.ndarray
    target_value_idx: np.ndarray
    target_value_mask: np.ndarray
    query_idx: np.ndarray
    query_mask: np.ndarray
    answer_idx: np.ndarray
    map_target: np.ndarray
    step_target: np.ndarray


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


def tokenize_python_code(code: str) -> list[str]:
    return TOKEN_PATTERN.findall(code)


def build_vocab_from_tokenizer_ids(tokens: list[int], max_seq_len: int, pad_id: int) -> np.ndarray:
    arr = np.full((max_seq_len,), pad_id, dtype=np.int32)
    arr[: min(len(tokens), max_seq_len)] = np.asarray(tokens[:max_seq_len], dtype=np.int32)
    return arr


class PythonGraphBuilder(ast.NodeVisitor):
    """Extract simple parser-derived graph edges from Python code."""

    def __init__(self) -> None:
        self.symbol_to_idx: dict[str, int] = {}
        self.next_idx = 0
        self.map_edges: list[tuple[int, int]] = []
        self.step_edges: list[tuple[int, int]] = []
        self.query_idx = 0
        self.answer_idx = 0

    def symbol(self, name: str) -> int:
        if name not in self.symbol_to_idx:
            self.symbol_to_idx[name] = self.next_idx
            self.next_idx += 1
        return self.symbol_to_idx[name]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        func_idx = self.symbol(node.name)
        for arg in node.args.args:
            arg_idx = self.symbol(arg.arg)
            self.step_edges.append((func_idx, arg_idx))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        class_idx = self.symbol(node.name)
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.step_edges.append((class_idx, self.symbol(base.id)))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.symbol(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module_idx = self.symbol(node.module or "module")
        for alias in node.names:
            self.step_edges.append((module_idx, self.symbol(alias.name)))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            value_idx = self.symbol(repr(node.value.value))
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.map_edges.append((self.symbol(target.id), value_idx))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            call_idx = self.symbol(node.func.id)
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    self.step_edges.append((call_idx, self.symbol(arg.id)))
        self.generic_visit(node)


def parse_python_graph(code: str, config: CodeDataConfig) -> PythonGraphBuilder:
    builder = PythonGraphBuilder()
    try:
        tree = ast.parse(code)
        builder.visit(tree)
    except SyntaxError:
        pass
    builder.next_idx = min(builder.next_idx, config.parser_max_nodes)
    return builder


def build_graph_targets(code: str, config: CodeDataConfig) -> GraphTargets:
    builder = parse_python_graph(code, config)
    num_symbols = config.parser_max_nodes
    num_values = config.parser_max_nodes
    map_target = np.zeros((num_symbols, num_values), dtype=np.float32)
    step_target = np.zeros((num_symbols, num_symbols), dtype=np.float32)
    for src, dst in builder.map_edges:
        if src < num_symbols and dst < num_values:
            map_target[src, dst] = 1.0
    for src, dst in builder.step_edges:
        if src < num_symbols and dst < num_symbols:
            step_target[src, dst] = 1.0

    query_idx = np.asarray(builder.query_idx, dtype=np.int32)
    answer_idx = np.asarray(builder.answer_idx, dtype=np.int32)
    return GraphTargets(
        event_markers=np.zeros((config.parser_max_nodes,), dtype=np.int32),
        source_idx=np.zeros((config.parser_max_nodes,), dtype=np.int32),
        source_mask=np.zeros((config.parser_max_nodes,), dtype=np.bool_),
        target_symbol_idx=np.zeros((config.parser_max_nodes,), dtype=np.int32),
        target_symbol_mask=np.zeros((config.parser_max_nodes,), dtype=np.bool_),
        target_value_idx=np.zeros((config.parser_max_nodes,), dtype=np.int32),
        target_value_mask=np.zeros((config.parser_max_nodes,), dtype=np.bool_),
        query_idx=query_idx,
        query_mask=np.asarray(builder.query_idx >= 0, dtype=np.bool_),
        answer_idx=answer_idx,
        map_target=map_target,
        step_target=step_target,
    )


def save_loader_state(path: str | Path, state: LoaderState) -> None:
    Path(path).write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def load_loader_state(path: str | Path) -> LoaderState:
    return LoaderState(**json.loads(Path(path).read_text(encoding="utf-8")))


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlShardLoader:
    """Resumable shard iterator for Colab restart-safe training."""

    def __init__(self, pattern: str, seed: int) -> None:
        self.pattern = pattern
        self.seed = seed
        self.shards = sorted(glob.glob(pattern))
        if not self.shards:
            raise FileNotFoundError(f"No shards found for pattern: {pattern}")

    def iter_records(self, state: LoaderState) -> Iterator[tuple[LoaderState, dict[str, Any]]]:
        rng = random.Random(state.rng_seed or self.seed)
        shard_indices = list(range(len(self.shards)))
        if state.epoch == 0 and state.shard_index == 0:
            rng.shuffle(shard_indices)

        while True:
            for shard_pos in range(state.shard_index, len(shard_indices)):
                shard_path = self.shards[shard_indices[shard_pos]]
                for sample_offset, record in enumerate(iter_jsonl_records(shard_path)):
                    if shard_pos == state.shard_index and sample_offset < state.sample_offset:
                        continue
                    next_state = LoaderState(
                        shard_index=shard_pos,
                        sample_offset=sample_offset + 1,
                        epoch=state.epoch,
                        rng_seed=self.seed,
                    )
                    yield next_state, record
                state = LoaderState(shard_index=shard_pos + 1, sample_offset=0, epoch=state.epoch, rng_seed=self.seed)
            state = LoaderState(shard_index=0, sample_offset=0, epoch=state.epoch + 1, rng_seed=self.seed)
            rng.shuffle(shard_indices)


def collate_batch(items: list[dict[str, Any]], max_seq_len: int, pad_id: int) -> dict[str, np.ndarray]:
    batch_size = len(items)
    token_ids = np.full((batch_size, max_seq_len), pad_id, dtype=np.int32)
    lengths = np.zeros((batch_size,), dtype=np.int32)
    query_idx = np.zeros((batch_size,), dtype=np.int32)
    query_mask = np.zeros((batch_size,), dtype=np.bool_)
    answer_idx = np.zeros((batch_size,), dtype=np.int32)
    map_targets = np.stack([np.asarray(item["graph_targets"]["map_target"], dtype=np.float32) for item in items])
    step_targets = np.stack([np.asarray(item["graph_targets"]["step_target"], dtype=np.float32) for item in items])
    event_markers = np.stack([np.asarray(item["graph_targets"]["event_markers"], dtype=np.int32) for item in items])
    source_idx = np.stack([np.asarray(item["graph_targets"]["source_idx"], dtype=np.int32) for item in items])
    source_mask = np.stack([np.asarray(item["graph_targets"]["source_mask"], dtype=np.bool_) for item in items])
    target_symbol_idx = np.stack([np.asarray(item["graph_targets"]["target_symbol_idx"], dtype=np.int32) for item in items])
    target_symbol_mask = np.stack([np.asarray(item["graph_targets"]["target_symbol_mask"], dtype=np.bool_) for item in items])
    target_value_idx = np.stack([np.asarray(item["graph_targets"]["target_value_idx"], dtype=np.int32) for item in items])
    target_value_mask = np.stack([np.asarray(item["graph_targets"]["target_value_mask"], dtype=np.bool_) for item in items])

    for row, item in enumerate(items):
        ids = np.asarray(item["token_ids"][:max_seq_len], dtype=np.int32)
        token_ids[row, : ids.shape[0]] = ids
        lengths[row] = ids.shape[0]
        query_idx[row] = int(item["graph_targets"]["query_idx"])
        query_mask[row] = bool(item["graph_targets"]["query_mask"])
        answer_idx[row] = int(item["graph_targets"]["answer_idx"])
    return {
        "token_ids": token_ids,
        "lengths": lengths,
        "query_idx": query_idx,
        "query_mask": query_mask,
        "answer_idx": answer_idx,
        "map_target": map_targets,
        "step_target": step_targets,
        "event_markers": event_markers,
        "source_idx": source_idx,
        "source_mask": source_mask,
        "target_symbol_idx": target_symbol_idx,
        "target_symbol_mask": target_symbol_mask,
        "target_value_idx": target_value_idx,
        "target_value_mask": target_value_mask,
    }


def build_training_record(
    code: str,
    tokenizer: TokenizerProtocol,
    config: CodeDataConfig,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token_ids = tokenizer.encode(code)
    graph_targets = build_graph_targets(code, config)
    return {
        "token_ids": token_ids[: config.parser_max_nodes],
        "graph_targets": {
            "event_markers": graph_targets.event_markers.tolist(),
            "source_idx": graph_targets.source_idx.tolist(),
            "source_mask": graph_targets.source_mask.tolist(),
            "target_symbol_idx": graph_targets.target_symbol_idx.tolist(),
            "target_symbol_mask": graph_targets.target_symbol_mask.tolist(),
            "target_value_idx": graph_targets.target_value_idx.tolist(),
            "target_value_mask": graph_targets.target_value_mask.tolist(),
            "query_idx": int(graph_targets.query_idx),
            "query_mask": bool(graph_targets.query_mask),
            "answer_idx": int(graph_targets.answer_idx),
            "map_target": graph_targets.map_target.tolist(),
            "step_target": graph_targets.step_target.tolist(),
        },
        "metadata": metadata or {},
    }


def write_jsonl_shards(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    examples_per_shard: int,
    prefix: str,
) -> list[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    shard_index = 0
    written = 0
    handle = None
    try:
        for record in records:
            if handle is None or written >= examples_per_shard:
                if handle is not None:
                    handle.close()
                shard_file = output_path / f"{prefix}_{shard_index:05d}.jsonl"
                shard_paths.append(str(shard_file))
                handle = shard_file.open("w", encoding="utf-8")
                shard_index += 1
                written = 0
            handle.write(json.dumps(record) + "\n")
            written += 1
    finally:
        if handle is not None:
            handle.close()
    return shard_paths
