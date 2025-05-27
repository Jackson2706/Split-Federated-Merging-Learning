# Centralized CNN Training on CIFAR-10

This is a centralized implementation for training a CNN model on the CIFAR-10 dataset using YAML configuration.

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python main.py --config configs/cifar_centralized.yaml
```

### Configuration Options

The training parameters can be configured in the YAML file. Here are the available options:

```yaml
dataset: cifar10
model: cnn
input_channels: 3
output_channels: 10
batch_size: 128
num_clients: 1  # For centralized training
lr: 0.01
num_communication: 100  # Number of epochs
momentum: 0.9
seed: 1
cuda: true
dataset_root: ./data
```

## Model Architecture

The model is a simple CNN with the following architecture:
- 2 convolutional layers with ReLU and max pooling
- 3 fully connected layers
- Output layer with softmax activation

## Training Process

The training process includes:
1. Data loading and preprocessing
   - Training set is split into 80% train and 20% validation
   - Test set is kept separate for final evaluation
2. Model training with validation
   - Model is trained on training set
   - Performance is evaluated on validation set
   - Best model is saved based on validation accuracy
3. Final evaluation
   - Best model is loaded and evaluated on test set
   - Test accuracy is reported

The best model will be saved in the `checkpoints` directory.
