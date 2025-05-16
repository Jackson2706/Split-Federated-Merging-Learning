import torch
import torch.nn as nn
import torch.optim as optim
from options import args_parser
from datasets.datasets import get_dataset
from models.cnn_conv_layer import cnn_3conv
import os
from tqdm import tqdm

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
    # Get arguments
    args = args_parser()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Load dataset
    _, _, train_loader, test_loader = get_dataset(args.dataset_root, args.dataset, args)
    print(f'Dataset: {args.dataset}')
    
    # Print data distribution settings
    print(f'\nData Distribution Settings:')
    if args.iid == 1:
        print(f'Distribution: IID with equal size')
    elif args.iid == 0:
        print(f'Distribution: Non-IID with balanced classes ({args.classes_per_client} classes per client)')
    elif args.iid == -1:
        print(f'Distribution: Non-IID with unbalanced classes')
    elif args.iid == -2:
        print(f'Distribution: One class per client')
        print(f'Edge Distribution: {"IID" if args.edgeiid == 1 else "Non-IID"}')
    print(f'Number of clients: {args.num_clients}')
    print(f'Fraction of clients used: {args.frac}\n')
    
    # Initialize model
    model = cnn_3conv(args.input_channels, args.output_channels).to(device)
    print(f'Model architecture:\n{model}')
    
    # Set up optimizer and loss function
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Create directory for saving models if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_acc = 0
    print(f'Starting training for {args.num_communication} epochs...')
    
    for epoch in range(args.num_communication):
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # Learning rate decay
        if (epoch + 1) % args.lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay
                print(f'Learning rate decayed to: {param_group["lr"]}')
        
        # Print metrics
        print(f'Epoch: {epoch+1}/{args.num_communication}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join('checkpoints', 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'data_distribution': {
                    'iid': args.iid,
                    'edgeiid': args.edgeiid,
                    'num_clients': args.num_clients,
                    'frac': args.frac,
                    'classes_per_client': args.classes_per_client
                }
            }, save_path)
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
    
    print(f'\nTraining completed!')
    print(f'Best Test Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main() 