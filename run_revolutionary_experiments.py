"""Run a harder benchmark pass for the post-transformer prototype."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.post_transformer_prototype.train import ExperimentConfig, run_experiment_suite, save_results


def main() -> None:
    config = ExperimentConfig(
        train_context_length=256,
        eval_context_lengths=(256, 512, 768, 1024),
        train_samples=4096,
        eval_samples=512,
        batch_size=64,
        epochs=6,
        tasks=("passkey", "associative", "sequential", "compositional"),
        num_workers=0,
        cpu_threads=10,
    )
    results = run_experiment_suite(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("results") / f"revolutionary_experiment_{timestamp}.json"
    save_results(results, output_path)
    print(f"Saved results to {output_path}")
    for model_name, model_results in results["models"].items():
        eval_1024 = model_results["evaluations"].get("1024")
        if eval_1024:
            print(
                f"{model_name:>18} | params={model_results['parameters']:>7} "
                f"| acc@1024={eval_1024['accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
