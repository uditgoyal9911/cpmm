"""Run the raw-text next-token plus graph-memory milestone."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.post_transformer_prototype.language_train import (
    LanguageMilestoneConfig,
    save_language_results,
    train_language_milestone,
)


def main() -> None:
    config = LanguageMilestoneConfig()
    results = train_language_milestone(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("results") / f"language_milestone_{timestamp}.json"
    save_language_results(results, output_path)
    final = results["history"][-1]
    print(f"Saved results to {output_path}")
    print(
        "language_milestone"
        f" | params={results['parameters']}"
        f" | ppl={final['eval_perplexity']:.3f}"
        f" | token_acc={final['eval_token_accuracy']:.3f}"
        f" | answer_acc={final['eval_answer_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
