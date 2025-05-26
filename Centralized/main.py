import torch
import torch.nn as nn
import torch.optim as optim
from utils.config import Config
from datasets.datasets import get_dataset
from models.cnn_conv_layer import cnn_3conv
import os
from tqdm import tqdm
import argparse

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing', leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy

def main():
    # Parse command line arguments for config path
    parser = argparse.ArgumentParser(description='Train with YAML config')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to YAML config file')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                      help='Override config values: key1=value1 key2=value2')
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Handle command line overrides
    for override in args.override:
        key, value = override.split('=')
        try:
            # Try to convert to int or float if possible
            value = eval(value)
        except:
            pass
        config.update({key: value})
    
    # Set device
    device = config.device
    print(f'Using device: {device}')
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)
    
    # Load dataset
    print(f'Dataset: {config.dataset}')
    train_loaders, val_loaders, test_loaders, v_train_loader, v_val_loader, v_test_loader = get_dataset(config.dataset_root, config.dataset, config)
    
    # Print data distribution settings
    print(f'\nData Distribution Settings:')
    if config.iid == 1:
        print(f'Distribution: IID with equal size')
    elif config.iid == 0:
        print(f'Distribution: Non-IID with balanced classes ({config.classes_per_client} classes per client)')
    elif config.iid == -1:
        print(f'Distribution: Non-IID with unbalanced classes')
    elif config.iid == -2:
        print(f'Distribution: One class per client')
        print(f'Edge Distribution: {"IID" if config.edgeiid == 1 else "Non-IID"}')
    print(f'Number of clients: {config.num_clients}')
    print(f'Fraction of clients used: {config.frac}\n')
    
    # Initialize model
    model = cnn_3conv(config.input_channels, config.output_channels).to(device)
    print(f'Model architecture:\n{model}')
    
    # Set up optimizer and loss function
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset-specific directory for saving models
    checkpoint_dir = os.path.join('checkpoints', config.dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a unique identifier for this run based on parameters
    run_id = f"clients{config.num_clients}_iid{config.iid}_lr{config.lr}_batch{config.batch_size}"
    if config.iid == 0:
        run_id += f"_classes{config.classes_per_client}"
    elif config.iid == -2:
        run_id += f"_edges{config.num_edges}_edgeiid{config.edgeiid}"
    
    # Training loop
    best_val_acc = 0
    epochs = config.num_communication*config.num_clients
    print(f'Starting training for {epochs} epochs...')
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train(model, v_train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = test(model, v_val_loader, criterion, device)
        
        # Learning rate decay
        if (epoch + 1) % config.lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= config.lr_decay
                print(f'Learning rate decayed to: {param_group["lr"]}')
        
        # Print metrics
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(checkpoint_dir, f'best_model_{run_id}.pth')
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': None,
                'config': config.config  # Save full configuration
            }
            
            torch.save(checkpoint, save_path)
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
            print(f'Saved to: {save_path}')
            
            # Also save the configuration separately
            config_save_path = os.path.join(checkpoint_dir, f'config_{run_id}.yaml')
            config.save(config_save_path)
    
    print(f'\nTraining completed!')
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    
    # Load best model and evaluate on test set
    print('\nEvaluating best model on test set...')
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = test(model, v_test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

    # Save model checkpoint
    checkpoint['test_acc'] = test_acc
    torch.save(checkpoint, save_path)
    print(f"=========Done training and testing=========")
    
if __name__ == '__main__':
    main() 