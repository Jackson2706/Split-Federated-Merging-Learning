import json
import os
import time

import torch
from ehsfp import architecture_manifest, get_ehsfp_config, partition_hash, prepare_run_dir, resolved_config_hash
from ehsfp.integrity import git_commit
from config.config_loader import ConfigLoader
from data import get_dataset
from hierarchy import HierarchicalFL
from models import get_model, get_decoder
from clients import test_inference
from torch.utils.data import DataLoader


def run(cfg_path: str):
    start_time = time.time()
    config_loader = ConfigLoader(cfg_path)
    config = config_loader.get_config()
    effective_ehsfp = get_ehsfp_config(config)
    config_hash = resolved_config_hash(config, effective_ehsfp)
    config["resolved_config_hash"] = config_hash
    print("Method: {}".format(config["strategy"]))

    if config["is_gpu"]:
        torch.cuda.set_device(config["gpu"])
    device = torch.device("cuda") if config["is_gpu"] else torch.device("cpu")

    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(config)

    # 3-tier models: client, edge, cloud (classifier on prototypes)
    client_cls, edge_cls, cloud_cls = get_model(config["model"], config["dataset"])
    client_model = client_cls()
    edge_model = edge_cls()
    cloud_model = cloud_cls(config)

    # Decoder for full-pipeline inference (spatial segmentation)
    decoder_cls = get_decoder(config["model"], config["dataset"])
    cloud_decoder = decoder_cls()
    arch = architecture_manifest((client_model, edge_model, cloud_model, cloud_decoder))
    out_base = os.path.join(os.path.dirname(__file__), "Figure", "data", "runs")
    run_dir = prepare_run_dir(out_base, arch["architecture_id"], config_hash)
    config["runtime_metrics_path"] = os.path.join(run_dir, "metrics_unrounded.jsonl")
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
        cloud_decoder=cloud_decoder,
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

    # Save metrics
    filtered_output = {k: v for k, v in output.items() if k not in ("best_weight",)}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(filtered_output, f, indent=4)

    # Test best model
    best_model = output.get("best_weight")
    if best_model is not None:
        best_model = best_model.to(device).eval()
        test_iou, test_dice, test_loss = test_inference(config, best_model, test_dataset)
        print(f"\nResults after {config['epochs']} global rounds:")
        print(f"|---- Best Validation IoU: {output['best_iou']*100:.2f}%")
        print(f"|---- Test IoU: {test_iou*100:.2f}%  Dice: {test_dice*100:.2f}%")
    else:
        print("\nNo best model was saved during training.")
        print(f"|---- Best Validation IoU: {output['best_iou']*100:.2f}%")

    print("Total Run Time: {:.4f}s".format(time.time() - start_time))
