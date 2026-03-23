"""Run the post-transformer prototype experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.post_transformer_prototype.train import ExperimentConfig, run_experiment_suite, save_results


def main() -> None:
    config = ExperimentConfig()
    results = run_experiment_suite(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("results") / f"experiment_{timestamp}.json"
    save_results(results, output_path)
    print(f"Saved results to {output_path}")
    for model_name, model_results in results["models"].items():
        eval_512 = model_results["evaluations"].get("512")
        if eval_512:
            print(
                f"{model_name:>18} | params={model_results['parameters']:>7} "
                f"| acc@512={eval_512['accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
