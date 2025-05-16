# Centralized and Federated Learning Implementation

This repository contains implementations for both centralized and federated learning approaches using PyTorch. It supports MNIST, CIFAR-10, and ISIC datasets with various distribution strategies.

## Installation

```bash
pip install -r requirements.txt
```

## Centralized Learning
Run this command inside the `Centralized` directory

```bash
python main.py --config configs/isic_centralized.yaml
```

## Federated Learning

For federated learning, modify the above commands by adjusting these key parameters:

### Data Distribution Flags

1. `--num_clients N`: Number of clients (default: 10)
   - For centralized: set to 1
   - For federated: set to desired number of clients (e.g., 10, 20, 100)

2. `--iid X`: Data distribution type
   - 1: IID (Independent and Identically Distributed)
   - 0: Non-IID with balanced classes
   - -1: Non-IID with unbalanced classes
   - -2: One class per client

3. `--classes_per_client N`: Number of classes each client gets (default: 2)
   - Only used when `--iid` is 0 (Non-IID balanced)
   - Must be less than total number of classes

4. `--edgeiid X`: Edge server data distribution (only used when `--iid -2`)
   - 1: IID distribution within edges
   - 0: Non-IID distribution within edges

5. `--frac X`: Fraction of clients to use (between 0 and 1)
   - Controls what fraction of clients participate in each round

### Training Configuration

1. `--num_communication N`: Number of communication rounds
2. `--num_local_update N`: Number of local updates (τ₁)
3. `--num_edge_aggregation N`: Number of edge aggregations (τ₂)
4. `--batch_size N`: Batch size for client training

### Example Federated Configurations

1. IID with 10 clients:
```bash
python main.py --dataset mnist --num_clients 10 --iid 1
```

2. Non-IID balanced with 20 clients, 2 classes per client:
```bash
python main.py --dataset cifar10 --num_clients 20 --iid 0 --classes_per_client 2
```

3. One class per client with edge servers:
```bash
python main.py --dataset isic --num_clients 9 --iid -2 --num_edges 3 --edgeiid 1
```

4. Non-IID unbalanced with partial client participation:
```bash
python main.py --dataset mnist --num_clients 100 --iid -1 --frac 0.1
```

## Model Checkpoints

Best models are automatically saved in the `checkpoints` directory with the following information:
- Model state
- Optimizer state
- Training/Test accuracy
- Data distribution configuration

## Additional Features

1. `--use_imagenet_stats`: Use ImageNet normalization (1) or calculate dataset-specific stats (0)
2. `--show_dis`: Show data distribution across clients
3. `--verbose`: Print progress bars and detailed information
4. `--gpu`: Select GPU device (0, 1, 2, 3)

## Learning Rate Scheduling

- `--lr`: Initial learning rate
- `--lr_decay`: Learning rate decay factor
- `--lr_decay_epoch`: Epochs between learning rate updates
