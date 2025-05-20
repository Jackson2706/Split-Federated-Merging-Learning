import torch
import os
import yaml
from datetime import datetime
import torch.nn as nn
from datasets.datasets import get_dataset
from models.cnn_conv_layer import cnn_3conv
from utils.config import Config
import argparse

def test_model(model, test_loader, device):
    """
    Test the model on the test set
    Args:
        model: The model to test
        test_loader: DataLoader for the test set
        device: Device to run the model on
    Returns:
        accuracy: Test accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def read_checkpoint(checkpoint_path, config_path=None):
    """
    Read and display checkpoint information in a readable format
    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
        config_path: Path to the config file (.yaml)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config if provided
    if config_path and os.path.exists(config_path):
        config = Config(config_path)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("Warning: No config found in checkpoint or provided config file")
        config = None
    
    # Print checkpoint information
    print("\n=== Checkpoint Information ===")
    print(f"File: {os.path.basename(checkpoint_path)}")
    print(f"Last modified: {datetime.fromtimestamp(os.path.getmtime(checkpoint_path))}")
    print("\n--- Training Information ---")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Training Accuracy: {checkpoint['train_acc']:.2f}%")
    if 'val_acc' in checkpoint:
        print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    if 'test_acc' in checkpoint:
        print(f"Test Accuracy: {checkpoint['test_acc']:.2f}%")
    
    # Print model configuration
    if config:
        print("\n--- Model Configuration ---")
        if isinstance(config, Config):
            # Access the config dictionary directly
            config_dict = config.config
        else:
            config_dict = config
            
        for key, value in config_dict.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"{key}: {value}")
    
    # Print model state dict keys
    print("\n--- Model Architecture ---")
    state_dict = checkpoint['model_state_dict']
    print("Model layers:")
    for key in state_dict.keys():
        print(f"- {key}: {state_dict[key].shape}")
    
    return checkpoint, config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Read and test model checkpoints')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint file')
    args = parser.parse_args()

    # If specific checkpoint is provided, use it
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint file not found at {args.checkpoint}")
            return
        checkpoint_files = [args.checkpoint]
    else:
        # Get all checkpoint files in the checkpoints directory
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoints directory not found at {checkpoint_dir}")
            return

        # Find all .pth files
        checkpoint_files = []
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.pth'):
                    checkpoint_files.append(os.path.join(root, file))

        if not checkpoint_files:
            print("No checkpoint files found.")
            return

        # Print available checkpoints
        print("Available checkpoints:")
        for i, file in enumerate(checkpoint_files, 1):
            print(f"{i}. {file}")

    # Let user choose which checkpoint to read
    while True:
        try:
            if len(checkpoint_files) > 1:
                choice = int(input("\nEnter the number of the checkpoint to read (0 to exit): "))
                if choice == 0:
                    break
                if 1 <= choice <= len(checkpoint_files):
                    checkpoint_path = checkpoint_files[choice-1]
                else:
                    print("Invalid choice. Please try again.")
                    continue
            else:
                checkpoint_path = checkpoint_files[0]

            # Read checkpoint
            checkpoint, config = read_checkpoint(checkpoint_path, args.config)
            
            # Ask if user wants to test the model
            test_choice = input("\nDo you want to test the model on the test set? (y/n): ").lower()
            if test_choice == 'y':
                if not config:
                    print("Error: No configuration available for testing")
                    continue
                
                # Set device
                device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
                
                # Create model
                model = cnn_3conv(config.input_channels, config.output_channels)
                model = model.to(device)
                
                # Initialize FC layers by doing a forward pass with dummy input
                dummy_input = torch.randn(1, config.input_channels, config.isic_image_size, config.isic_image_size).to(device)
                with torch.no_grad():
                    model(dummy_input)
                
                # print(f"model_state_dict: {checkpoint['model_state_dict']}")
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Get test dataset
                _, _, _, _, _, v_test_loader = get_dataset(config.dataset_root, config.dataset, config)
                
                # Test on virtual test loader (all test data)
                print("\nTesting model on test set...")
                test_acc = test_model(model, v_test_loader, device)
                print(f"Test Accuracy: {test_acc:.2f}%")
            
            if len(checkpoint_files) == 1:
                break
                
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main() 