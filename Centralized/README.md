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

## Results

The training results are saved in the following locations:

1. Model checkpoints:
   - `checkpoints/best_model.pth`: Best model based on validation accuracy

2. Training metrics:
   - `results/training_metrics.csv`: CSV file containing epoch-wise metrics
     - Training loss and accuracy
     - Validation loss and accuracy
   - `results/training_curves.png`: Plot of training and validation curves
     - Loss curves
     - Accuracy curves
   - `results/test_results.txt`: Final test results
     - Best validation accuracy
     - Test loss and accuracy

The best model will be saved in the `checkpoints` directory.
