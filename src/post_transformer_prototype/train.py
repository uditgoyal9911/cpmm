"""Training and evaluation helpers for the post-transformer prototype."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import LongContextDataset, collate_batch, task_names
from .models import CausalPredictiveMemoryMachine, GRUBaseline, TransformerBaseline


@dataclass
class ExperimentConfig:
    train_context_length: int = 128
    eval_context_lengths: Tuple[int, ...] = (128, 256, 384, 512)
    train_samples: int = 3072
    eval_samples: int = 384
    batch_size: int = 64
    epochs: int = 5
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 7
    tasks: Tuple[str, ...] = ("passkey", "associative", "sequential")
    num_workers: int = 0
    cpu_threads: int = 12
    parallel_model_jobs: int = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_torch_threads(cpu_threads: int) -> None:
    torch.set_num_threads(max(1, cpu_threads))
    interop_threads = max(1, min(4, cpu_threads // 2))
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # PyTorch only allows setting inter-op threads once per process.
        pass


def worker_thread_budget(config: ExperimentConfig) -> int:
    return max(1, config.cpu_threads // max(1, config.parallel_model_jobs))


def pick_device() -> torch.device:
    return torch.device("cpu")


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def build_model(name: str, **kwargs: object) -> nn.Module:
    if name == "cpmm":
        return CausalPredictiveMemoryMachine(**kwargs)
    if name == "transformer":
        return TransformerBaseline(**kwargs)
    if name == "gru":
        return GRUBaseline(**kwargs)
    raise ValueError(f"Unknown model: {name}")


def move_batch(batch: Dict[str, torch.Tensor | List[str]], device: torch.device) -> Dict[str, torch.Tensor | List[str]]:
    moved: Dict[str, torch.Tensor | List[str]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def forward_for_batch(model_name: str, model: nn.Module, batch: Dict[str, torch.Tensor | List[str]]) -> Tuple[torch.Tensor, Dict[str, float]]:
    if model_name == "cpmm":
        logits, diagnostics = model(  # type: ignore[misc]
            batch["context_tokens"],
            batch["context_lengths"],
            batch["query_tokens"],
            batch["query_lengths"],
        )
        return logits, diagnostics
    if model_name in {"transformer", "gru"}:
        logits = model(batch["sequence_tokens"], batch["sequence_lengths"])  # type: ignore[misc]
        return logits, {}
    raise ValueError(f"Unsupported model: {model_name}")


def train_model(
    model_name: str,
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    dataset = LongContextDataset(
        size=config.train_samples,
        context_length=config.train_context_length,
        tasks=config.tasks,
        seed=config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    model.to(device)
    model.train()

    start = time.perf_counter()
    epoch_metrics: List[Dict[str, float]] = []
    for epoch in range(config.epochs):
        running_loss = 0.0
        running_correct = 0
        total = 0
        diagnostics: Dict[str, float] = {}
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, diag = forward_for_batch(model_name, model, batch)
            loss = F.cross_entropy(logits, batch["answer"])
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=-1)
            running_loss += float(loss.item()) * logits.size(0)
            running_correct += int((predictions == batch["answer"]).sum().item())
            total += logits.size(0)
            diagnostics = diag or diagnostics

        epoch_metrics.append(
            {
                "epoch": float(epoch + 1),
                "loss": running_loss / max(total, 1),
                "accuracy": running_correct / max(total, 1),
                **diagnostics,
            }
        )
    elapsed = time.perf_counter() - start
    return {
        "epochs": config.epochs,
        "metrics": epoch_metrics,
        "train_seconds": elapsed,
    }


@torch.no_grad()
def evaluate_model(
    model_name: str,
    model: nn.Module,
    context_length: int,
    tasks: Iterable[str],
    eval_samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    dataset = LongContextDataset(
        size=eval_samples,
        context_length=context_length,
        tasks=tuple(tasks),
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )
    model.eval()

    correct = 0
    total = 0
    task_correct: Dict[str, int] = {task: 0 for task in task_names()}
    task_total: Dict[str, int] = {task: 0 for task in task_names()}
    for batch in loader:
        batch = move_batch(batch, device)
        logits, _ = forward_for_batch(model_name, model, batch)
        predictions = logits.argmax(dim=-1)
        matches = predictions == batch["answer"]
        correct += int(matches.sum().item())
        total += matches.numel()
        for task, match in zip(batch["tasks"], matches.tolist()):
            task_total[task] += 1
            task_correct[task] += int(match)

    return {
        "context_length": context_length,
        "accuracy": correct / max(total, 1),
        "task_accuracy": {
            task: task_correct[task] / max(task_total[task], 1) for task in task_total
        },
    }


def run_model_job(
    label: str,
    model_name: str,
    model_kwargs: Dict[str, object],
    config: ExperimentConfig,
    eval_context_lengths: Tuple[int, ...],
    eval_seed_offset: int,
) -> Tuple[str, Dict[str, object]]:
    seed_everything(config.seed)
    configure_torch_threads(worker_thread_budget(config))
    device = pick_device()
    model = build_model(model_name, **model_kwargs)
    train_summary = train_model(model_name, model, config, device)
    evaluations = {
        str(context_length): evaluate_model(
            model_name,
            model,
            context_length=context_length,
            tasks=config.tasks,
            eval_samples=config.eval_samples,
            batch_size=config.batch_size,
            seed=config.seed + eval_seed_offset + context_length,
            device=device,
        )
        for context_length in eval_context_lengths
    }
    return (
        label,
        {
            "parameters": parameter_count(model),
            "train_summary": train_summary,
            "evaluations": evaluations,
        },
    )


def run_experiment_suite(config: ExperimentConfig) -> Dict[str, object]:
    seed_everything(config.seed)
    configure_torch_threads(worker_thread_budget(config))
    device = pick_device()
    results: Dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "models": {},
    }

    transformer_max_seq_len = max(config.train_context_length, max(config.eval_context_lengths)) + 8
    base_specs = {
        "cpmm": {"d_model": 64, "num_slots": 6, "chunk_size": 16, "refinement_steps": 2, "use_relations": True},
        "transformer": {"d_model": 64, "num_layers": 2, "nhead": 4, "max_seq_len": transformer_max_seq_len},
        "gru": {"d_model": 64, "num_layers": 1},
    }
    ablation_specs = {
        "cpmm_no_refinement": {"d_model": 64, "num_slots": 6, "chunk_size": 16, "refinement_steps": 0, "use_relations": True},
        "cpmm_single_slot": {"d_model": 64, "num_slots": 1, "chunk_size": 16, "refinement_steps": 2, "use_relations": True},
        "cpmm_no_relations": {"d_model": 64, "num_slots": 6, "chunk_size": 16, "refinement_steps": 2, "use_relations": False},
    }

    max_workers = max(1, min(config.parallel_model_jobs, len(base_specs)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_model_job,
                model_label,
                "cpmm" if model_label == "cpmm" else model_label,
                kwargs,
                config,
                config.eval_context_lengths,
                1000,
            )
            for model_label, kwargs in base_specs.items()
        ]
        for future in as_completed(futures):
            label, result = future.result()
            results["models"][label] = result

    ablation_contexts = (config.eval_context_lengths[1], config.eval_context_lengths[-1])
    ablation_config = ExperimentConfig(
        train_context_length=config.train_context_length,
        eval_context_lengths=ablation_contexts,
        train_samples=config.train_samples,
        eval_samples=config.eval_samples,
        batch_size=config.batch_size,
        epochs=max(3, config.epochs - 1),
        lr=config.lr,
        weight_decay=config.weight_decay,
        seed=config.seed,
        tasks=config.tasks,
        num_workers=config.num_workers,
        cpu_threads=config.cpu_threads,
        parallel_model_jobs=config.parallel_model_jobs,
    )
    ablation_workers = max(1, min(config.parallel_model_jobs, len(ablation_specs)))
    with ProcessPoolExecutor(max_workers=ablation_workers) as executor:
        futures = [
            executor.submit(
                run_model_job,
                model_label,
                "cpmm",
                kwargs,
                ablation_config,
                ablation_contexts,
                2000,
            )
            for model_label, kwargs in ablation_specs.items()
        ]
        for future in as_completed(futures):
            label, result = future.result()
            results["models"][label] = result

    return results


def save_results(results: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
