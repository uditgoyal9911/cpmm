"""Evaluation helpers for code reasoning and completion tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class CodeEvalSample:
    prompt: str
    expected: str
    task_type: str


def exact_match(prediction: str, expected: str) -> float:
    return 1.0 if prediction.strip() == expected.strip() else 0.0


def prefix_match(prediction: str, expected: str) -> float:
    prediction = prediction.strip()
    expected = expected.strip()
    if not expected:
        return 0.0
    matched = 0
    for left, right in zip(prediction, expected):
        if left != right:
            break
        matched += 1
    return matched / max(len(expected), 1)


def score_generation(prediction: str, expected: str, task_type: str) -> float:
    if task_type in {"definition_lookup", "symbol_trace", "import_reasoning"}:
        return exact_match(prediction, expected)
    return prefix_match(prediction, expected)


def aggregate_scores(samples: Sequence[CodeEvalSample], predictions: Sequence[str]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for sample, prediction in zip(samples, predictions):
        score = score_generation(prediction, sample.expected, sample.task_type)
        totals[sample.task_type] = totals.get(sample.task_type, 0.0) + score
        counts[sample.task_type] = counts.get(sample.task_type, 0) + 1
    return {
        task_type: totals[task_type] / max(counts[task_type], 1)
        for task_type in sorted(totals)
    }
