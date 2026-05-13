"""
Results reporting and serialization.

Provides utilities for saving experiment results to JSON
and printing summary reports.
"""

import json
import os


def save_results(output, config, output_dir="Figure/data"):
    """
    Save training results to a JSON file.

    Automatically excludes non-serializable keys (e.g., model weights).

    Args:
        output: Dict of training results from strategy.train().
        config: Experiment configuration dict.
        output_dir: Directory to save the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    exclude_keys = ["best_weight"]
    filtered_output = {
        k: v for k, v in output.items() if k not in exclude_keys
    }

    filename = (
        f"HSFL_{config['dataset']}_iid:{config['iid']}_{config['model']}_"
        f"{config['num_users']}_users_t1:{config['t1']}_t2:{config['t2']}.json"
    )
    path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(filtered_output, f, indent=4)

    print(f"Results saved to: {path}")
    return path


def print_final_results(config, best_f1):
    """
    Print final training summary.

    Args:
        config: Experiment configuration dict.
        best_f1: Best validation F1 score achieved.
    """
    print(f'\nResults after {config["epochs"]} global rounds of training:')
    print(f"|---- Best Validation F1 Score: {100 * best_f1:.2f}%")
