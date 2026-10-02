"""MiSFP classification runner: ``python main.py --task classification --method misfp``.

Reuses the H-SFP data, models, and HierarchicalFL (added to sys.path); the
``hsfp`` variant runs the untouched H-SFP class as reference A so that every
variant shares one evaluation and provenance path.
"""

import json
import os
import platform
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_HSFP_DIR = os.path.join(_ROOT, "classification", "H-SFP")


def _git_state(root):
    def _run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                  timeout=30).stdout
        except Exception:
            return ""
    import hashlib
    paths = ["misfp", "classification/MiSFP", "classification/H-SFP", "ehsfp", "main.py"]
    diff = _run("diff", "HEAD", "--", *paths)
    return {
        "commit": _run("rev-parse", "HEAD").strip(),
        "dirty_paths": [l[3:] for l in _run("status", "--porcelain", "--", *paths).splitlines()],
        "diff_sha256_vs_head": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _versions():
    import numpy, sklearn, torch
    out = {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": numpy.__version__, "sklearn": sklearn.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def _evaluate(model, dataset, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    model = model.to(device).eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, t in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            targets.append(torch.as_tensor(t).cpu())
    return torch.cat(preds).numpy().astype(np.int64), torch.cat(targets).numpy().astype(np.int64)


def _metrics(preds, targets, num_classes):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    labels = list(range(num_classes))
    per_f1 = f1_score(targets, preds, labels=labels, average=None, zero_division=0)
    per_acc = [float((preds[targets == c] == c).mean()) if (targets == c).any() else None
               for c in labels]
    return {"accuracy": float(accuracy_score(targets, preds)),
            "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
            "per_class_accuracy": per_acc,
            "per_class_f1": [float(v) for v in per_f1],
            "worst_decile_class_accuracy": float(np.mean(sorted(
                [a for a in per_acc if a is not None])[: max(1, num_classes // 10)]))}


def run(cfg_path: str):
    if _HSFP_DIR not in sys.path:
        sys.path.insert(1, _HSFP_DIR)
    try:
        _run(cfg_path)
    finally:
        if _HSFP_DIR in sys.path:
            sys.path.remove(_HSFP_DIR)


def _run(cfg_path):
    import numpy as np
    import torch
    from config import ConfigLoader
    from data import get_dataset
    from models import get_model
    from hierarchy import HierarchicalFL
    from ehsfp import (architecture_manifest, get_ehsfp_config, partition_hash, prepare_run_dir,
                       resolved_config_hash, state_dict_hash, write_run_fingerprints)
    from misfp import get_misfp_config, public_view
    from hierarchy_misfp import MiSFPHierarchicalFL, current_pipeline

    start = time.time()
    config = ConfigLoader(cfg_path).get_config()
    effective_ehsfp = get_ehsfp_config(config)
    mcfg = get_misfp_config(config, effective_ehsfp)
    # Resolved MiSFP settings become part of the hashed config.
    config.update({k: v for k, v in public_view(mcfg).items() if k.startswith("misfp_")})
    config_hash = resolved_config_hash(config, effective_ehsfp)
    config["resolved_config_hash"] = config_hash
    variant, label = mcfg["misfp_variant"], mcfg["misfp_label"]
    print(f"[MiSFP] variant={variant} ({label})")

    use_cuda = bool(config.get("is_gpu", False)) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(config["gpu"])
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    config["is_gpu"] = use_cuda

    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(config)
    if valid_dataset is test_dataset:
        print("[MiSFP] WARNING: validation set IS the test set (val_from_train=0); "
              "best-val checkpoint selection is test-selected. Prefer val_from_train>0.")
    client_model, edge_model, cloud_model = get_model(config["model"], config["dataset"])
    split_depth = int(config.get("client_split_depth", 2) or 2)
    if split_depth != 2:
        client_model = client_model(split_depth=split_depth)
        edge_model = edge_model(in_dim=client_model.SPLIT_DIMS[split_depth])
        cloud_model = cloud_model(config)
    else:
        client_model, edge_model, cloud_model = client_model(), edge_model(), cloud_model(config)
    arch = architecture_manifest((client_model, edge_model, cloud_model))

    out_base = config.get("output_dir") or os.path.join("results", "misfp", "runs")
    if not os.path.isabs(out_base):
        out_base = os.path.join(_ROOT, out_base)
    run_dir = prepare_run_dir(out_base, arch["architecture_id"], config_hash)
    config["runtime_metrics_path"] = os.path.join(run_dir, "metrics_unrounded.jsonl")
    config["runtime_counters_path"] = os.path.join(run_dir, "runtime_counters.jsonl")
    config["history_path"] = os.path.join(run_dir, "history.csv")
    config["run_fingerprints_path"] = os.path.join(run_dir, "run_fingerprints.json")
    config["prediction_artifact_dir"] = run_dir
    status_path = os.path.join(run_dir, "status.json")

    def _status(state, **extra):
        with open(status_path, "w") as fh:
            json.dump({"state": state, "time": time.time(), **extra}, fh, indent=2)

    _status("running", pid=os.getpid())
    p_hash = partition_hash(user_groups)
    np.savez_compressed(os.path.join(run_dir, "client_partition.npz"),
                        **{f"client_{k}": np.asarray(v, dtype=np.int64) for k, v in user_groups.items()})
    write_run_fingerprints(config["run_fingerprints_path"], {
        "seed": int(config["seed"]), "partition_hash": p_hash,
        "initial_model_hashes": {"client": state_dict_hash(client_model.state_dict()),
                                 "edge": state_dict_hash(edge_model.state_dict()),
                                 "cloud": state_dict_hash(cloud_model.state_dict())},
    })

    common = dict(args=config, client_model=client_model, client_weights=client_model.state_dict(),
                  edge_model=edge_model, edge_weights=edge_model.state_dict(),
                  cloud_model=cloud_model, cloud_weight=cloud_model.state_dict(),
                  test_dataset=test_dataset)
    if variant == "hsfp":
        hfl = HierarchicalFL(**common)
    else:
        hfl = MiSFPHierarchicalFL(**common, misfp_cfg=mcfg,
                                  misfp_log_path=os.path.join(run_dir, "misfp_rounds.jsonl"))
    with open(os.path.join(run_dir, "client_to_edge.json"), "w") as fh:
        json.dump({str(k): int(v) for k, v in hfl.connectivity[-1].items()}, fh)

    metadata = {
        "method": "MiSFP" if variant != "hsfp" else "H-SFP (reference A via MiSFP runner)",
        "variant": variant, "label": label, "resolved_config": config,
        "resolved_misfp": public_view(mcfg), "effective_ehsfp": effective_ehsfp,
        "resolved_config_hash": config_hash, "partition_hash": p_hash,
        "seed": int(config["seed"]), "code_state": _git_state(_ROOT), "versions": _versions(),
        "architecture": arch, "run_dir": run_dir,
        "validation_source": "val_from_train" if valid_dataset is not test_dataset else "test (leaky)",
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)
    print(f"[MiSFP] run_dir={run_dir}")

    try:
        output = hfl.train_end_to_end(
            train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset,
            user_groups=user_groups, config=config, epochs=config["epochs"],
            checkpoint_path=os.path.join(run_dir, "checkpoint.pt"),
        )
    except BaseException as exc:
        _status("failed", error=repr(exc))
        raise

    num_classes = int(config["num_classes"])
    results = {}
    # Final-round composed model (headline: no checkpoint selection involved).
    final_model = current_pipeline(hfl)
    p, t = _evaluate(final_model, test_dataset, device)
    np.savez_compressed(os.path.join(run_dir, "test_final_predictions.npz"), preds=p, targets=t)
    results["test_final_round"] = _metrics(p, t, num_classes)
    best = output.get("best_weight")
    if best is not None:
        p, t = _evaluate(best, test_dataset, device)
        np.savez_compressed(os.path.join(run_dir, "test_best_val_predictions.npz"), preds=p, targets=t)
        results["test_best_val_checkpoint"] = _metrics(p, t, num_classes)
        results["test_best_val_checkpoint"]["selected_round"] = (output.get("best_validation") or {}).get("round")

    comm = dict(hfl.comm_tracker)
    totals = dict(getattr(hfl, "totals", {}))
    summary = {
        "method": metadata["method"], "variant": variant, "label": label,
        "seed": int(config["seed"]), "epochs": int(config["epochs"]),
        "iid": bool(config.get("iid")), "partition_hash": p_hash,
        "resolved_config_hash": config_hash, "run_dir": run_dir,
        "validation_accuracy": output.get("validation_accuracy"),
        "validation_f1": output.get("validation_f1"),
        "cloud_loss": output.get("cloud_loss"),
        **results,
        "communication_MB": comm,
        "prototype_payload_bytes": {
            "client_to_edge": totals.get("client_to_edge_payload_bytes"),
            "edge_to_cloud": totals.get("edge_to_cloud_payload_bytes"),
            "note": ("exact serialized MiSFP packet bytes (header+ids+counts+moments)"
                     if variant != "hsfp" else
                     "H-SFP counts tensor bytes of (mean,std) only; see communication_MB"),
        },
        "misfp_totals": totals,
        "wall_time_s": time.time() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if use_cuda else 0,
        "versions": metadata["versions"], "code_state": metadata["code_state"],
    }
    with open(os.path.join(run_dir, "misfp_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    filtered = {k: v for k, v in output.items() if k != "best_weight"}
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(filtered, fh, indent=2, default=float)
    _status("completed", wall_time_s=summary["wall_time_s"])
    fr = results["test_final_round"]
    print(f"[MiSFP] {label}: final-round test acc={fr['accuracy']*100:.2f}% "
          f"macro-F1={fr['macro_f1']*100:.2f}%  total_comm={comm['total_comm_MB']:.3f} MB")
    print(f"[MiSFP] summary -> {os.path.join(run_dir, 'misfp_summary.json')}")
