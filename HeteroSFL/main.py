import argparse
import copy
import gc
import json
import os
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from tensorboardX import SummaryWriter

# --- Your Custom Imports ---
from config import ConfigLoader
from data import get_dataset
from models import get_model 

# -------------------------------------------------------------------------
# 1. HELPER CLASSES & FUNCTIONS
# -------------------------------------------------------------------------

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone(), torch.tensor(label)

def get_weight_size_mb(weights):
    return sum(torch.numel(v) for v in weights.values()) * 4 / (1024**2)

class HeteroServerAdapter(nn.Module):
    """
    Acts as the 'Decoder' mentioned in the HeteroSFL paper (Fig 3).
    It adapts the variable 'Wide' or 'Narrow' channels (e.g., 76, 16) 
    back to the fixed input size the Server Model expects (e.g., 64).
    """
    def __init__(self, server_model, wide_channels, server_expected_channels=64):
        super().__init__()
        # Decoder: Maps split_channels -> server_input_channels [cite: 217]
        self.decoder = nn.Conv2d(wide_channels, server_expected_channels, kernel_size=3, padding=1)
        self.server_model = server_model

    def forward(self, x):
        x = self.decoder(x)
        x = self.server_model(x)
        return x

def extract_narrow_model(wide_model, narrow_channels):
    """
    Extracts the 'Narrow' model weights from the 'Wide' model.
    The Narrow BL is a subnetwork of the Wide BL (first k channels)[cite: 260].
    """
    narrow_state = copy.deepcopy(wide_model.state_dict())
    
    # Identify the last layer (The Split Layer / Encoder)
    keys = list(narrow_state.keys())
    # Typically the last two keys are 'layer.weight' and 'layer.bias' of the final Conv layer
    weight_key = keys[-2]
    bias_key = keys[-1]
    
    # Slice the output channels to keep only the first 'narrow_channels'
    narrow_state[weight_key] = narrow_state[weight_key][:narrow_channels]
    narrow_state[bias_key] = narrow_state[bias_key][:narrow_channels]
    
    return narrow_state

def aggregate_hetero(global_wide_weights, local_updates, narrow_channels):
    """
    HeteroSFL Aggregation [cite: 263-265].
    - Shared channels (0 to narrow_channels): Aggregated from ALL clients.
    - Exclusive channels (narrow_channels to wide_channels): Aggregated ONLY from High-end clients.
    """
    update_acc = {k: torch.zeros_like(v) for k, v in global_wide_weights.items()}
    count_acc = {k: torch.zeros_like(v) for k, v in global_wide_weights.items()}
    
    # Identify split layer keys again
    keys = list(global_wide_weights.keys())
    weight_key = keys[-2]
    bias_key = keys[-1]

    for local_w in local_updates:
        # Check if this update is from a Narrow model (by checking shape)
        is_narrow = local_w[weight_key].shape[0] == narrow_channels
        
        for k in local_w.keys():
            if k in [weight_key, bias_key] and is_narrow:
                # Add to shared part only
                update_acc[k][:narrow_channels] += local_w[k]
                count_acc[k][:narrow_channels] += 1
            else:
                # Standard aggregation for everything else (or full wide updates)
                update_acc[k] += local_w[k]
                count_acc[k] += 1

    # Average and update
    avg_weights = copy.deepcopy(global_wide_weights)
    for k in avg_weights.keys():
        mask = count_acc[k] > 0
        avg_weights[k][mask] = update_acc[k][mask] / count_acc[k][mask]
        
    return avg_weights

def bdks_loss_function(pred_wide, pred_narrow, label, alpha=1.0):
    """
    Bidirectional Knowledge Sharing (BDKS) Loss[cite: 399].
    Total Loss = L_Task + alpha * L_N2W + L_W2N
    """
    criterion = nn.CrossEntropyLoss()
    
    # 1. Task Loss
    loss_task = criterion(pred_wide, label)
    
    # 2. Narrow-to-Wide (N2W) [cite: 368]
    log_probs_wide = F.log_softmax(pred_wide, dim=1)
    probs_narrow = F.softmax(pred_narrow, dim=1)
    loss_n2w = F.kl_div(log_probs_wide, probs_narrow, reduction='batchmean')
    
    # 3. Wide-to-Narrow (W2N) [cite: 391]
    probs_wide = F.softmax(pred_wide, dim=1)
    log_probs_narrow = F.log_softmax(pred_narrow, dim=1)
    loss_w2n = F.kl_div(log_probs_narrow, probs_wide, reduction='batchmean')

    return loss_task + (alpha * loss_n2w) + loss_w2n

# -------------------------------------------------------------------------
# 2. MAIN TRAINING LOOP
# -------------------------------------------------------------------------

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config_loader = ConfigLoader(args.cfg)
    config = config_loader.get_config()
    
    # --- HeteroSFL Parameters ---
    WIDE_CHANNELS = config.get("wide_channels", 76)     # High-end channel width
    NARROW_CHANNELS = config.get("narrow_channels", 4) # Low-end channel width
    SERVER_EXPECTED_CHANNELS = config.get("server_in_channels", 64) # Server input channels
        
    HIGH_END_RATIO = config.get("high_end_ratio", 0.3)
    BDKS_ALPHA = config.get("bdks_alpha", 1.0)
    
    print(f"HeteroSFL Setup: Wide={WIDE_CHANNELS}, Narrow={NARROW_CHANNELS}, ServerIn={SERVER_EXPECTED_CHANNELS}, Ratio={HIGH_END_RATIO}")

    # [IMPORTANT] Update config to force the correct channel width for Client Model Init
    config["client_out_channels"] = WIDE_CHANNELS 
    
    device = torch.device("cuda" if config["is_gpu"] else "cpu")
    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(config)
    model_tuple = get_model(config["model"], config["dataset"])

    # 1. Initialize Client Model (Wide Version)
    client_model_wide = model_tuple[0](config).to(device)
    
    # 2. Initialize Server Model with Adapter 
    original_server_model = model_tuple[1](config).to(device)
    main_server_model = HeteroServerAdapter(
        original_server_model, 
        wide_channels=WIDE_CHANNELS, 
        server_expected_channels=SERVER_EXPECTED_CHANNELS
    ).to(device)

    # 3. Assign Client Groups (Static Affiliation)
    num_users = config["num_users"]
    num_high = int(num_users * HIGH_END_RATIO)
    client_types = ["high"] * num_high + ["low"] * (num_users - num_high)
    np.random.shuffle(client_types)

    comm_cost_dict = {"upload_MB": 0, "download_MB": 0}
    best_f1 = 0.0

    # --- Training Loop ---
    for epoch in tqdm(range(config["epochs"])):
        idxs_users = np.random.choice(range(num_users), max(int(config["frac"] * num_users), 1), replace=False)
        
        local_weights = []
        epoch_losses = []
        
        # Server Optimizer
        server_optimizer = SGD(main_server_model.parameters(), lr=config["lr"], momentum=config["momentum"])

        for idx in idxs_users:
            client_type = client_types[idx]
            
            # --- A. Prepare Local Client Model ---
            if client_type == "high":
                local_client_model = copy.deepcopy(client_model_wide)
            else:
                local_client_model = copy.deepcopy(client_model_wide)
                # Slice weights for Narrow BL [cite: 260]
                narrow_state = extract_narrow_model(client_model_wide, NARROW_CHANNELS)
                
                # Apply sliced weights to last layer
                last_layer = list(local_client_model.children())[-1]
                if isinstance(last_layer, nn.Conv2d):
                    last_layer.out_channels = NARROW_CHANNELS
                    last_layer.weight = nn.Parameter(narrow_state[list(narrow_state.keys())[-2]])
                    last_layer.bias = nn.Parameter(narrow_state[list(narrow_state.keys())[-1]])

            local_client_model.to(device)
            local_client_model.train()
            
            client_optimizer = SGD(local_client_model.parameters(), lr=config["lr"], momentum=config["momentum"])
            
            # --- B. Local Training ---
            loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                                batch_size=config["local_bs"], shuffle=True)
            
            client_losses = []
            
            for _ in range(config["local_ep"]):
                for image, label in loader:
                    server_optimizer.zero_grad()
                    client_optimizer.zero_grad()
                    image, label = image.to(device), label.to(device)
                    
                    # 1. Client Forward
                    activation = local_client_model(image)
                    
                    # 2. Server Processing (Split Logic)
                    if client_type == "high":
                        # High-end Path (BDKS)
                        act_wide = activation.clone().detach().requires_grad_(True)
                        
                        # Simulate Narrow view (Zero out extra channels) for BDKS
                        act_narrow_sim = act_wide.clone()
                        mask = torch.ones_like(act_narrow_sim)
                        mask[:, NARROW_CHANNELS:, :, :] = 0
                        act_narrow_sim = act_narrow_sim * mask
                        
                        pred_wide = main_server_model(act_wide)
                        pred_narrow = main_server_model(act_narrow_sim)
                        
                        loss = bdks_loss_function(pred_wide, pred_narrow, label, alpha=BDKS_ALPHA)
                        loss.backward()
                        
                        activation_grad = act_wide.grad
                        
                    else:
                        # Low-end Path (Standard)
                        act_narrow = activation.clone().detach().requires_grad_(True)
                        
                        # Pad to match Wide size [B, 4] -> [B, 76] for Server Adapter
                        padding = torch.zeros(
                            act_narrow.shape[0], 
                            WIDE_CHANNELS - NARROW_CHANNELS, 
                            act_narrow.shape[2], 
                            act_narrow.shape[3]
                        ).to(device)
                        
                        act_input = torch.cat([act_narrow, padding], dim=1)
                        
                        pred = main_server_model(act_input)
                        loss = nn.CrossEntropyLoss()(pred, label)
                        loss.backward()
                        
                        # Use gradient from the leaf tensor (Narrow)
                        activation_grad = act_narrow.grad

                    # 3. Steps
                    server_optimizer.step()
                    activation.backward(activation_grad)
                    client_optimizer.step()
                    
                    client_losses.append(loss.item())
                    comm_cost_dict["upload_MB"] += (activation.numel() * 4 / (1024**2))
            
            local_weights.append(copy.deepcopy(local_client_model.state_dict()))
            epoch_losses.append(np.mean(client_losses))
            del local_client_model
            torch.cuda.empty_cache()

        # --- D. Aggregation ---
        avg_loss = np.mean(epoch_losses)
        new_weights = aggregate_hetero(client_model_wide.state_dict(), local_weights, NARROW_CHANNELS)
        client_model_wide.load_state_dict(new_weights)

        # --- E. Evaluation ---
        if (epoch + 1) % config["print_every"] == 0:
            client_model_wide.eval()
            main_server_model.eval()
            
            total_eval_loss = 0
            all_preds, all_labels = [], []
            criterion = nn.CrossEntropyLoss()
            
            # Evaluate using Global Client Wide + Server (The Inference Model) 
            eval_loader = DataLoader(valid_dataset, batch_size=config["local_bs"], shuffle=False)
            
            with torch.no_grad():
                for image, label in eval_loader:
                    image, label = image.to(device), label.to(device)
                    
                    # Pass through Global Client (Wide) -> Server Adapter -> Server
                    act = client_model_wide(image)
                    out = main_server_model(act)
                    
                    loss = criterion(out, label)
                    total_eval_loss += loss.item()
                    
                    _, predicted = torch.max(out.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())
            
            avg_eval_loss = total_eval_loss / len(eval_loader)
            eval_f1 = f1_score(all_labels, all_preds, average='macro')
            eval_acc = accuracy_score(all_labels, all_preds)
            
            print(f"| Epoch {epoch+1} | Loss: {avg_eval_loss:.4f} | Acc: {eval_acc:.4f} | F1: {eval_f1:.4f} |")
            
            # Save Best Model
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                print(f" -> New Best F1: {best_f1:.4f}")
                # Save weights if needed
                # torch.save(client_model_wide.state_dict(), "best_client.pth")
                # torch.save(main_server_model.state_dict(), "best_server.pth")

            client_model_wide.train()
            main_server_model.train()

    # --- F. Final Test ---
    print("\n--- Final Test Evaluation ---")
    client_model_wide.eval()
    main_server_model.eval()
    test_loader = DataLoader(test_dataset, batch_size=config["local_bs"], shuffle=False)
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            act = client_model_wide(image)
            out = main_server_model(act)
            _, predicted = torch.max(out.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(label.cpu().numpy())

    final_acc = accuracy_score(test_labels, test_preds)
    final_f1 = f1_score(test_labels, test_preds, average='macro')
    
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(f"Final Test F1 Score: {final_f1:.4f}")
    print(f"Total Upload Cost: {comm_cost_dict['upload_MB']:.2f} MB")
    print(f"Total Run Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()