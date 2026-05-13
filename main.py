#!/usr/bin/env python3
"""
H-SFP: Hierarchical Split-Federated Learning with Prototypes — ECCV 2026
Unified experiment runner.

Usage:
    python main.py --task <task> --method <method> --cfg <config>
    python main.py --list
    python main.py --list-configs [--task <task>] [--method <method>]

Examples:
    python main.py --task classification --method h-sfp \\
        --cfg configs/classification/h-sfp/ham10000_our_vgg_5_10.yaml

    python main.py --task classification --method federated \\
        --cfg configs/classification/federated/cifar_fedavg_resnet50.yaml

    python main.py --task segmentation --method hierfl \\
        --cfg configs/segmentation/hierfl/isic_hierfl_resnet50.yaml
"""

import argparse
import glob as _glob
import importlib.util
import os
import sys

try:
    import wandb
except ImportError:
    wandb = None

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Registry: task -> method -> relative path to method directory
REGISTRY = {
    "classification": {
        "h-sfp":      "classification/H-SFP",       # PRIMARY METHOD
        "federated":  "classification/Federated",
        "hierfl":     "classification/HierFL",
        "splitfl":    "classification/SplitFL",
        "hetero-sfl": "classification/HeteroSFL",
        "hsfl":       "classification/HSFL",
    },
    "segmentation": {
        "h-sfp":      "segmentation/H-SFP",       # PRIMARY METHOD
        "federated":  "segmentation/Federated",
        "hierfl":     "segmentation/HierFL",
        "splitfl":    "segmentation/SplitFL",
        "hetero-sfl": "segmentation/HeteroSFL",
        "hsfl":       "segmentation/HSFL",
    },
}


def list_methods():
    print("Available methods:\n")
    for task, methods in REGISTRY.items():
        print(f"  Task: {task}")
        for method, path in methods.items():
            tag = "[PRIMARY]" if method == "h-sfp" else "[baseline]"
            print(f"    --method {method:<12} {tag:<11}  {path}/")
        print()


def list_configs(task=None, method=None):
    print("Available configs:\n")
    for t, methods in REGISTRY.items():
        if task and t != task:
            continue
        for m in methods:
            if method and m != method:
                continue
            cfg_dir = os.path.join(ROOT_DIR, "configs", t, m)
            if not os.path.isdir(cfg_dir):
                continue
            yamls = sorted(
                y for y in _glob.glob(os.path.join(cfg_dir, "*.yaml"))
                if "default" not in os.path.basename(y)
            )
            if yamls:
                tag = "[PRIMARY]" if m == "h-sfp" else "[baseline]"
                print(f"  [{t}] {m} {tag}:")
                for y in yamls:
                    print(f"    {os.path.relpath(y, ROOT_DIR)}")
                print()


def _load_runner(method_dir: str):
    """Import runner.py from a method directory, adding it to sys.path."""
    runner_path = os.path.join(method_dir, "runner.py")
    if not os.path.isfile(runner_path):
        raise FileNotFoundError(f"runner.py not found in {method_dir}")

    sys.path.insert(0, method_dir)
    spec = importlib.util.spec_from_file_location("_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cleanup_runner(method_dir: str):
    """Remove method dir from sys.path and clear cached modules."""
    if method_dir in sys.path:
        sys.path.remove(method_dir)
    stale = [k for k in list(sys.modules) if k in {
        "_runner", "config", "data", "models", "hierarchy",
        "clients", "server", "servers", "FedServer", "strategies",
    }]
    for k in stale:
        del sys.modules[k]


def _init_wandb(task, method, cfg_path, wandb_project, wandb_entity):
    """Initialize a wandb run. Returns True if wandb was initialized."""
    if wandb is None:
        print("Warning: wandb not installed. Run: pip install wandb")
        return False

    # Parse YAML config to log as wandb config
    import yaml
    with open(cfg_path, "r") as f:
        exp_config = yaml.safe_load(f) or {}

    run_name = f"{task}/{method}/{os.path.basename(cfg_path).replace('.yaml', '')}"
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config={
            "task": task,
            "method": method,
            "cfg": os.path.relpath(cfg_path, ROOT_DIR),
            **exp_config,
        },
        tags=[task, method, exp_config.get("dataset", ""), exp_config.get("model", "")],
        reinit=True,
    )
    return True


def run(task, method, cfg, use_wandb=False, wandb_project="H-SFP", wandb_entity=None):
    task_methods = REGISTRY.get(task)
    if task_methods is None:
        print(f"Error: unknown task '{task}'. Available: {list(REGISTRY)}")
        sys.exit(1)

    method_rel = task_methods.get(method)
    if method_rel is None:
        print(f"Error: unknown method '{method}' for task '{task}'.")
        print(f"Available: {list(task_methods)}")
        sys.exit(1)

    cfg_path = cfg if os.path.isabs(cfg) else os.path.join(ROOT_DIR, cfg)
    if not os.path.isfile(cfg_path):
        print(f"Error: config not found: {cfg_path}")
        print(f"Tip: python main.py --list-configs --task {task} --method {method}")
        sys.exit(1)

    wandb_active = False
    if use_wandb:
        wandb_active = _init_wandb(task, method, cfg_path, wandb_project, wandb_entity)

    method_dir = os.path.join(ROOT_DIR, method_rel)
    print(f"[Runner] task={task}  method={method}")
    print(f"[Runner] cfg={os.path.relpath(cfg_path, ROOT_DIR)}\n")

    runner = _load_runner(method_dir)
    try:
        runner.run(cfg_path)
    finally:
        _cleanup_runner(method_dir)
        if wandb_active and wandb.run is not None:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="H-SFP Unified Experiment Runner (ECCV 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", choices=list(REGISTRY), help="Task to run")
    parser.add_argument("--method", help="Method (h-sfp, federated, hierfl, ...)")
    parser.add_argument("--cfg", help="Path to YAML config file")
    parser.add_argument("--list", action="store_true", help="List all methods")
    parser.add_argument("--list-configs", action="store_true", help="List all configs")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="H-SFP", help="W&B project name (default: H-SFP)")
    parser.add_argument("--wandb-entity", default=None, help="W&B team/entity name")

    args = parser.parse_args()

    if args.list:
        list_methods()
        return

    if args.list_configs:
        list_configs(task=args.task, method=args.method)
        return

    if not (args.task and args.method and args.cfg):
        parser.print_help()
        print("\nError: --task, --method, and --cfg are all required.")
        sys.exit(1)

    run(args.task, args.method, args.cfg,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity)


if __name__ == "__main__":
    main()
