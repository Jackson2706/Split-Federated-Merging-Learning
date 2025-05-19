import torch
import os
import yaml
from datetime import datetime

def read_checkpoint(checkpoint_path):
    """
    Read and display checkpoint information in a readable format
    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print checkpoint information
    print("\n=== Checkpoint Information ===")
    print(f"File: {os.path.basename(checkpoint_path)}")
    print(f"Last modified: {datetime.fromtimestamp(os.path.getmtime(checkpoint_path))}")
    print("\n--- Training Information ---")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Training Accuracy: {checkpoint['train_acc']:.2f}%")
    print(f"Test Accuracy: {checkpoint['test_acc']:.2f}%")
    
    # Print model configuration
    if 'config' in checkpoint:
        print("\n--- Model Configuration ---")
        config = checkpoint['config']
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"{key}: {value}")
    
    # Print model state dict keys
    print("\n--- Model Architecture ---")
    state_dict = checkpoint['model_state_dict']
    print("Model layers:")
    for key in state_dict.keys():
        print(f"- {key}: {state_dict[key].shape}")

def main():
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
            choice = int(input("\nEnter the number of the checkpoint to read (0 to exit): "))
            if choice == 0:
                break
            if 1 <= choice <= len(checkpoint_files):
                read_checkpoint(checkpoint_files[choice-1])
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main() 