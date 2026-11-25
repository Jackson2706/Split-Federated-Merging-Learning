#!/usr/bin/env python3
"""
E-HSFP: Episodic Hierarchical Split-Federated Prototyping
(extending H-SFP — ECCV 2026)
Unified experiment runner.

Usage:
    python main.py --task <task> --method <method> --cfg <config>
    python main.py --task <task> --method h-sfp --cfg <config> --ablation full_e_hsfp
    python main.py --list
    python main.py --list-configs [--task <task>] [--method <method>]

Examples:
    python main.py --task classification --method h-sfp \\
        --cfg configs/classification/h-sfp/ham10000_our_vgg_5_10.yaml

    python main.py --task classification --method h-sfp \\
        --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \\
        --ablation full_e_hsfp

    python main.py --task segmentation --method h-sfp \\
        --cfg configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml \\
        --ablation hsfp_memory_reliability_prc
"""

import argparse
import glob as _glob
import importlib.util
import os
import random
import sys

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Registry: task -> method -> relative path to method directory
REGISTRY = {
    "classification": {
        "h-sfp":      "classification/H-SFP",       # PRIMARY METHOD
        "misfp":      "classification/MiSFP",       # MiSFP (second journal extension)
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


def _set_seed(seed: int):
    """Set random seed for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _set_speed_flags(cfg):
    """Opt-in CUDA speed-ups (--bf16, --tf32, --cudnn-benchmark); results may change by seed noise."""
    if not torch.cuda.is_available():
        return
    if cfg.get("bf16"):
        torch.set_autocast_dtype("cuda", torch.bfloat16)  # every autocast(device_type='cuda') site
    if cfg.get("tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cfg.get("cudnn_benchmark"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


WEIGHTING = {"learned": "learnable_reliability", "count": "sample_count_weighted",
             "uniform": "average", "precision": "precision_weighted"}


def _revision_overrides(args):
    """Map the revision CLI flags onto config keys (None = flag not given)."""
    top = {"epochs": args.rounds, "staleness_tau": args.max_delay, "packet_loss": args.packet_loss,
           "sampler": args.sampler, "scatter_fix": args.scatter_fix, "prc_mode": args.prc,
           "oracle": args.oracle, "oracle_every": args.oracle_every, "gpu_data": args.gpu_data,
           "bf16": args.bf16, "tf32": args.tf32, "cudnn_benchmark": args.cudnn_benchmark}
    eh = {"memory_replay_ratio": args.alpha, "prototype_dropout_rate": args.p_train,
          "max_prototype_age": args.retention, "lambda_prc": args.lambda_prc,
          "aggregation_mode": WEIGHTING.get(args.weighting)}
    out = {k: v for k, v in top.items() if v is not None}
    if args.accum_gamma is not None:
        out["stat_accumulation_decay"] = args.accum_gamma
        if args.accum_gamma == 0:
            out["memory_mode"] = "records"  # memory_mode=statistics would re-enable gamma=0.8
    if args.out:
        out["revision_out"] = os.path.abspath(args.out)
        out["output_dir"] = os.path.join(out["revision_out"], "run")
    eh = {k: v for k, v in eh.items() if v is not None}
    if eh:
        out["ehsfp_overrides"] = eh
    return out


def _merge_overrides(a, b):
    out = {**a, **b}
    eh = {**(a.get("ehsfp_overrides") or {}), **(b.get("ehsfp_overrides") or {})}
    if eh:
        out["ehsfp_overrides"] = eh
    return out


def _coerce(value: str):
    """Coerce a CLI string to bool/int/float when possible, else keep as str."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def _parse_overrides(pairs):
    """Parse ['key=value', ...] into a dict with coerced values."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            print(f"Error: --set expects KEY=VALUE, got '{p}'")
            sys.exit(1)
        k, v = p.split("=", 1)
        k = k.strip()
        if k.startswith("ehsfp."):  # wins over --ablation presets (ehsfp/config.py)
            out.setdefault("ehsfp_overrides", {})[k[len("ehsfp."):]] = _coerce(v.strip())
        else:
            out[k] = _coerce(v.strip())
    return out


def run(task, method, cfg, use_wandb=False, wandb_project="H-SFP", wandb_entity=None,
        ablation=None, seed=None, overrides=None):
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

    # Inject ablation mode, seed, and/or arbitrary key=value overrides into a temp config YAML
    overrides = overrides or {}
    seed = 42 if seed is None else seed
    use_tmp_cfg = (ablation and method == "h-sfp") or (seed is not None) or bool(overrides)
    if use_tmp_cfg:
        import tempfile
        import yaml
        with open(cfg_path, "r") as f:
            cfg_data = yaml.safe_load(f) or {}
        if ablation and method == "h-sfp":
            cfg_data["ablation_mode"] = ablation
            print(f"[E-HSFP] Ablation mode: {ablation}")
        cfg_data["seed"] = seed
        print(f"[Runner] Seed: {seed}")
        if overrides:
            ehsfp_over = {**(cfg_data.get("ehsfp_overrides") or {}), **(overrides.pop("ehsfp_overrides", None) or {})}
            cfg_data.update(overrides)
            if ehsfp_over:
                cfg_data["ehsfp_overrides"] = ehsfp_over
            print(f"[Runner] Overrides: {overrides}")
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="exp_",
            dir=os.path.dirname(cfg_path), delete=False,
        )
        yaml.dump(cfg_data, tmp, default_flow_style=False)
        tmp.close()
        cfg_path = tmp.name

    # Set global seed before anything else
    _set_seed(seed)
    _set_speed_flags(overrides)

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
        # Clean up temp config
        if use_tmp_cfg:
            try:
                os.unlink(cfg_path)
            except OSError:
                pass


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
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config key (repeatable), e.g. --set partition=two_level_dirichlet "
             "--set alpha_edge=0.1 --set iid=false",
    )
    parser.add_argument(
        "--ablation",
        choices=[
            "baseline_hsfp", "hsfp_memory", "hsfp_memory_dropout",
            "hsfp_memory_reliability", "hsfp_memory_reliability_prc", "full_e_hsfp",
        ],
        default=None,
        help="E-HSFP ablation mode (only applies to h-sfp method)",
    )

    rev = parser.add_argument_group("revision experiments (all default to the config value)")
    rev.add_argument("--out", help="Run directory: config.yaml, meta.json, metrics.csv, ckpt.pt (resume), "
                                   "final.pt, summary.json, DONE")
    rev.add_argument("--rounds", type=int, help="Global rounds (paper: 200)")
    rev.add_argument("--alpha", type=float, help="Replay mixing coefficient; 1 disables replay")
    rev.add_argument("--max-delay", type=int, help="Packet delays ~ U{0..tau}")
    rev.add_argument("--p-train", type=float, help="Training-time source-mask rate")
    rev.add_argument("--packet-loss", type=float, help="External packet loss p (Assumption 8)")
    rev.add_argument("--retention", type=int, help="Record retention limit A_M (= M_M); 0 disables the bank")
    rev.add_argument("--weighting", choices=list(WEIGHTING))
    rev.add_argument("--sampler", choices=["gaussian", "mixture"])
    rev.add_argument("--scatter-fix", type=int, choices=[0, 1])
    rev.add_argument("--lambda-prc", type=float, help="PRC weight; 0 disables PRC")
    rev.add_argument("--accum-gamma", type=float, help="Edge accumulation decay gamma; 0 turns accumulation off")
    rev.add_argument("--prc", choices=["standard", "corrected"])
    rev.add_argument("--oracle", type=int, choices=[0, 1], help="Drift/coverage diagnostics (needs --out)")
    rev.add_argument("--oracle-every", type=int)
    rev.add_argument("--gpu-data", type=int, choices=[0, 1])
    rev.add_argument("--bf16", type=int, choices=[0, 1])
    rev.add_argument("--tf32", type=int, choices=[0, 1])
    rev.add_argument("--cudnn-benchmark", type=int, choices=[0, 1])

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
        wandb_entity=args.wandb_entity,
        ablation=args.ablation,
        seed=args.seed,
        overrides=_merge_overrides(_parse_overrides(args.overrides), _revision_overrides(args)))


if __name__ == "__main__":
    main()
