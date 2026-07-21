import json
import os
import time

import torch
from ehsfp import architecture_manifest, get_ehsfp_config, partition_hash, prepare_run_dir, resolved_config_hash
from ehsfp.integrity import git_commit
from config import ConfigLoader
from data import get_dataset
from hierarchy import HierarchicalFL
from models import get_model
from sklearn.metrics import accuracy_score
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
    config["runtime_metrics_path"] = os.path.join(run_dir, "metrics_unrounded.jsonl")
    config["runtime_counters_path"] = os.path.join(run_dir, "runtime_counters.jsonl")
    metadata = {
        "resolved_config": config,
        "effective_ehsfp": effective_ehsfp,
        "resolved_config_hash": config_hash,
        "git_commit": git_commit(os.path.join(os.path.dirname(__file__), "..", "..")),
        "partition_hash": partition_hash(user_groups),
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

    # Save metrics (exclude non-serializable model)
    filtered_output = {k: v for k, v in output.items() if k not in ("best_weight",)}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(filtered_output, f, indent=4)

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
        print(f"\nResults after {config['epochs']} global rounds:")
        print("|---- Best Validation Acc: {:.2f}%".format(100 * output["best_val_top1"]))
        print("|---- Test Acc: {:.2f}%".format(100 * acc))
    else:
        print("\nNo best model was saved during training.")
        print("|---- Best Validation Acc: {:.2f}%".format(100 * output["best_val_top1"]))

    print("Total Run Time: {:.4f}s".format(time.time() - start_time))
