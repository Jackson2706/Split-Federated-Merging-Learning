import json
import os
import time

import torch
from ehsfp import (
    architecture_manifest,
    get_ehsfp_config,
    partition_hash,
    prepare_run_dir,
    resolved_config_hash,
    state_dict_hash,
    write_run_fingerprints,
)
from ehsfp.integrity import git_commit
from config import ConfigLoader
from data import get_dataset
from hierarchy import HierarchicalFL
from models import get_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def run(cfg_path: str):
    start_time = time.time()
    config_loader = ConfigLoader(cfg_path)
    config = config_loader.get_config()
    effective_ehsfp = get_ehsfp_config(config)
    config_hash = resolved_config_hash(config, effective_ehsfp)
    config["resolved_config_hash"] = config_hash
    print("Method: {}".format(config["strategy"]))

    use_cuda = bool(config.get("is_gpu", False)) and torch.cuda.is_available()
    if config.get("require_cuda", False) and not use_cuda:
        raise RuntimeError("CUDA is required by this config, but no CUDA device is available")
    if config.get("is_gpu", False) and not torch.cuda.is_available():
        print("[Device] CUDA requested but unavailable; falling back to CPU.")
    if use_cuda:
        torch.cuda.set_device(config["gpu"])
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    config["is_gpu"] = use_cuda
    print(f"[Device] training_device={device}")

    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(config)
    client_model, edge_model, cloud_model = get_model(config["model"], config["dataset"])
    # Optional deeper client split for the CIFAR ResNet path (default 2 = historical).
    split_depth = int(config.get("client_split_depth", 2) or 2)
    if split_depth != 2:
        if config["dataset"] != "cifar100" or config["model"] != "resnet50":
            raise ValueError("client_split_depth is implemented for cifar100/resnet50 only")
        client_model = client_model(split_depth=split_depth)
        d = client_model.SPLIT_DIMS[split_depth]
        residual = bool(config.get("edge_residual", False))
        edge_model = edge_model(in_dim=d, residual=residual)
        cloud_model = cloud_model({**config, "cloud_in_dim": d if residual else 256})
    else:
        client_model, edge_model, cloud_model = (
            client_model(), edge_model(), cloud_model(config)
        )
    arch = architecture_manifest((client_model, edge_model, cloud_model))
    configured_out_dir = config.get("output_dir")
    if configured_out_dir:
        out_base = configured_out_dir if os.path.isabs(configured_out_dir) else os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", configured_out_dir)
        )
    else:
        out_base = os.path.join(os.path.dirname(__file__), "Figure", "data", "runs")
    run_dir = prepare_run_dir(out_base, arch["architecture_id"], config_hash)
    deadlock_diagnostics = None
    if os.environ.get("EHSFP_DEADLOCK_DIAG") == "1":
        from ehsfp.deadlock_diagnostics import start_deadlock_diagnostics
        deadlock_diagnostics = start_deadlock_diagnostics(run_dir)
    fingerprint_path = os.path.join(run_dir, "run_fingerprints.json")
    config["runtime_metrics_path"] = os.path.join(run_dir, "metrics_unrounded.jsonl")
    config["runtime_counters_path"] = os.path.join(run_dir, "runtime_counters.jsonl")
    config["history_path"] = os.path.join(run_dir, "history.csv")
    config["run_fingerprints_path"] = fingerprint_path
    config["prediction_artifact_dir"] = run_dir
    fingerprints = {
        "seed": int(config["seed"]),
        "partition_hash": partition_hash(user_groups),
        "initial_model_hashes": {
            "client": state_dict_hash(client_model.state_dict()),
            "edge": state_dict_hash(edge_model.state_dict()),
            "cloud": state_dict_hash(cloud_model.state_dict()),
        },
    }
    write_run_fingerprints(fingerprint_path, fingerprints)
    metadata = {
        "resolved_config": config,
        "effective_ehsfp": effective_ehsfp,
        "resolved_config_hash": config_hash,
        "git_commit": git_commit(os.path.join(os.path.dirname(__file__), "..", "..")),
        "partition_hash": fingerprints["partition_hash"],
        "run_fingerprints_path": fingerprint_path,
        "architecture": arch,
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"[Integrity] resolved_config_hash={config_hash}")
    print(f"[Integrity] architecture_id={arch['architecture_id']} run_dir={run_dir}")

    hierarchical_fl = HierarchicalFL(
        args=config,
        client_model=client_model,
        client_weights=client_model.state_dict(),
        edge_model=edge_model,
        edge_weights=edge_model.state_dict(),
        cloud_model=cloud_model,
        cloud_weight=cloud_model.state_dict(),
        test_dataset=test_dataset,
    )
    hierarchical_fl.print_structure()

    output = hierarchical_fl.train_end_to_end(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        user_groups=user_groups,
        config=config,
        epochs=config["epochs"],
        checkpoint_path=os.path.join(run_dir, "checkpoint.pt"),
    )
    output["device"] = str(device)
    output["peak_vram_bytes"] = torch.cuda.max_memory_allocated() if use_cuda else 0
    metadata["device"] = str(device)
    metadata["peak_vram_bytes"] = output["peak_vram_bytes"]
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Test best model
    best_model = output.get("best_weight")
    if best_model is not None:
        best_model = best_model.to(device).eval()
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = best_model(data).argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        output["selected_checkpoint"].update({
            "test_accuracy": acc,
            "test_macro_f1": macro_f1,
        })
        print(f"\nResults after {config['epochs']} global rounds:")
        print("|---- Best Validation Acc: {:.2f}%".format(100 * output["best_val_top1"]))
        print("|---- Test Acc: {:.2f}%".format(100 * acc))
    else:
        print("\nNo best model was saved during training.")
        print("|---- Best Validation Acc: {:.2f}%".format(100 * output["best_val_top1"]))

    # Save metrics only after selected-checkpoint evaluation is attached.
    filtered_output = {k: v for k, v in output.items() if k not in ("best_weight",)}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(filtered_output, f, indent=4)

    if deadlock_diagnostics is not None:
        deadlock_diagnostics.stop()
    print("Total Run Time: {:.4f}s".format(time.time() - start_time))
