"""Language-model milestone training for the post-transformer prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Dict, List

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import (
    NUM_KEYS,
    NUM_VALUES,
    TEXT_NAME_VALUE_START,
    TEXT_PAD,
    TextGraphDataset,
    collate_text_graph_batch,
)
from .models import GraphTextLanguageModel
from .train import configure_torch_threads, pick_device


@dataclass
class LanguageMilestoneConfig:
    train_samples: int = 4096
    eval_samples: int = 512
    batch_size: int = 64
    epochs: int = 6
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 11
    tasks: tuple[str, ...] = ("passkey", "associative", "sequential", "compositional")
    cpu_threads: int = 8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch: Dict[str, torch.Tensor | List[str]], device: torch.device) -> Dict[str, torch.Tensor | List[str]]:
    moved: Dict[str, torch.Tensor | List[str]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor | List[str]]) -> Dict[str, torch.Tensor]:
    lm_logits = outputs["lm_logits"]
    target_tokens = batch["target_tokens"]
    lm_loss = F.cross_entropy(
        lm_logits.reshape(-1, lm_logits.size(-1)),
        target_tokens.reshape(-1),
        ignore_index=TEXT_PAD,
    )

    answer_targets = batch["answer_token"] - TEXT_NAME_VALUE_START
    answer_loss = F.cross_entropy(outputs["answer_logits"], answer_targets)
    graph_answer_loss = F.cross_entropy(outputs["graph_logits"], answer_targets)

    map_loss = F.mse_loss(outputs["map_memory"], batch["map_target"])
    step_loss = F.mse_loss(outputs["step_memory"], batch["step_target"])
    graph_loss = map_loss + step_loss
    total_loss = lm_loss + 1.0 * answer_loss + 1.0 * graph_answer_loss + 0.25 * graph_loss
    return {
        "total": total_loss,
        "lm": lm_loss,
        "answer": answer_loss,
        "graph_answer": graph_answer_loss,
        "graph": graph_loss,
    }


def train_language_milestone(config: LanguageMilestoneConfig) -> Dict[str, object]:
    seed_everything(config.seed)
    configure_torch_threads(config.cpu_threads)
    device = pick_device()

    train_dataset = TextGraphDataset(config.train_samples, config.tasks, config.seed)
    eval_dataset = TextGraphDataset(config.eval_samples, config.tasks, config.seed + 10000)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_text_graph_batch)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_text_graph_batch)

    model = GraphTextLanguageModel()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start = time.perf_counter()
    history: List[Dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        running = {"total": 0.0, "lm": 0.0, "answer": 0.0, "graph_answer": 0.0, "graph": 0.0}
        total_tokens = 0
        total_examples = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["input_tokens"], batch["lengths"], batch["answer_position"])
            losses = compute_losses(outputs, batch)
            losses["total"].backward()
            optimizer.step()

            running["total"] += float(losses["total"].item()) * batch["input_tokens"].size(0)
            running["lm"] += float(losses["lm"].item()) * batch["input_tokens"].size(0)
            running["answer"] += float(losses["answer"].item()) * batch["input_tokens"].size(0)
            running["graph_answer"] += float(losses["graph_answer"].item()) * batch["input_tokens"].size(0)
            running["graph"] += float(losses["graph"].item()) * batch["input_tokens"].size(0)
            total_examples += batch["input_tokens"].size(0)
            total_tokens += int((batch["target_tokens"] != TEXT_PAD).sum().item())

        eval_metrics = evaluate_language_milestone(model, eval_loader, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_total_loss": running["total"] / max(total_examples, 1),
                "train_lm_loss": running["lm"] / max(total_examples, 1),
                "train_answer_loss": running["answer"] / max(total_examples, 1),
                "train_graph_answer_loss": running["graph_answer"] / max(total_examples, 1),
                "train_graph_loss": running["graph"] / max(total_examples, 1),
                **eval_metrics,
            }
        )

    elapsed = time.perf_counter() - start
    return {
        "config": asdict(config),
        "device": str(device),
        "parameters": sum(param.numel() for param in model.parameters()),
        "history": history,
        "elapsed_seconds": elapsed,
    }


@torch.no_grad()
def evaluate_language_milestone(
    model: GraphTextLanguageModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_lm = 0.0
    total_answer = 0.0
    total_graph_answer = 0.0
    total_graph = 0.0
    total_examples = 0
    total_tokens = 0
    correct_answer = 0
    correct_graph_answer = 0
    token_correct = 0

    task_correct = {task: 0 for task in ("passkey", "associative", "sequential", "compositional")}
    task_total = {task: 0 for task in ("passkey", "associative", "sequential", "compositional")}

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["input_tokens"], batch["lengths"], batch["answer_position"])
        losses = compute_losses(outputs, batch)
        batch_size = batch["input_tokens"].size(0)
        total_examples += batch_size
        total_loss += float(losses["total"].item()) * batch_size
        total_lm += float(losses["lm"].item()) * batch_size
        total_answer += float(losses["answer"].item()) * batch_size
        total_graph_answer += float(losses["graph_answer"].item()) * batch_size
        total_graph += float(losses["graph"].item()) * batch_size

        mask = batch["target_tokens"] != TEXT_PAD
        token_correct += int((outputs["lm_logits"].argmax(dim=-1)[mask] == batch["target_tokens"][mask]).sum().item())
        total_tokens += int(mask.sum().item())

        answer_pred = outputs["answer_logits"].argmax(dim=-1)
        graph_answer_pred = outputs["graph_logits"].argmax(dim=-1)
        answer_target = batch["answer_token"] - TEXT_NAME_VALUE_START
        matches = answer_pred == answer_target
        graph_matches = graph_answer_pred == answer_target
        correct_answer += int(matches.sum().item())
        correct_graph_answer += int(graph_matches.sum().item())
        for task, match in zip(batch["tasks"], matches.tolist()):
            task_total[task] += 1
            task_correct[task] += int(match)

    avg_lm = total_lm / max(total_examples, 1)
    return {
        "eval_total_loss": total_loss / max(total_examples, 1),
        "eval_lm_loss": avg_lm,
        "eval_answer_loss": total_answer / max(total_examples, 1),
        "eval_graph_answer_loss": total_graph_answer / max(total_examples, 1),
        "eval_graph_loss": total_graph / max(total_examples, 1),
        "eval_perplexity": float(np.exp(min(avg_lm, 20.0))),
        "eval_token_accuracy": token_correct / max(total_tokens, 1),
        "eval_answer_accuracy": correct_answer / max(total_examples, 1),
        "eval_graph_answer_accuracy": correct_graph_answer / max(total_examples, 1),
        "eval_passkey_answer_accuracy": task_correct["passkey"] / max(task_total["passkey"], 1),
        "eval_associative_answer_accuracy": task_correct["associative"] / max(task_total["associative"], 1),
        "eval_sequential_answer_accuracy": task_correct["sequential"] / max(task_total["sequential"], 1),
        "eval_compositional_answer_accuracy": task_correct["compositional"] / max(task_total["compositional"], 1),
    }


def save_language_results(results: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
