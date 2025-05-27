import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import yaml
import argparse
from models.CNNCifar import CNNCifar

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    return val_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    return test_loss, accuracy

def main():
    # Parse command line arguments for config path
    parser = argparse.ArgumentParser(description='Train with YAML config')
    parser.add_argument('--config', type=str, default='configs/cifar_centralized.yaml',
                      help='Path to YAML config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    if config['cuda']:
        torch.cuda.manual_seed(config['seed'])
    
    # Set device
    device = torch.device("cuda" if config['cuda'] else "cpu")
    
    # Data loading and preprocessing
    apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=config['dataset_root'], train=True, download=True, transform=apply_transform)
    
    # Split training dataset into train and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    
    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=config['dataset_root'], train=False, download=True, transform=apply_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Initialize model
    model = CNNCifar({"num_classes": config['output_channels']}).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    
    # Training loop
    best_val_acc = 0
    for epoch in range(config['num_communication']):
        print(f'\nEpoch: {epoch+1}/{config["num_communication"]}')
        
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f'Best model saved with validation accuracy: {best_val_acc:.2f}%')
    
    # Load best model and evaluate on test set
    print('\nEvaluating best model on test set...')
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

if __name__ == '__main__':
    main() 