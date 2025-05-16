"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""
import os
from tqdm import tqdm
from sklearn import metrics
import numpy as np

import torch
import torch.backends.cudnn as cudnn
cudnn.banchmark = True

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from options import args_parser

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target

def gen_ran_sum(_sum, num_users):
    base = 100*np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100*num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)
    print(p.sum())
    p = p[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0]
    size_users = size_users + base
    print(size_users.sum())
    return size_users

def get_mean_and_std(dataset):
    """
    compute the mean and std value of dataset
    """
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("=>compute mean and std")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def iid_esize_split(dataset, args, kwargs, is_shuffle = True):
    """
    split the dataset to users
    Return:
        dict of the data_loaders
    """
    sum_samples = len(dataset)
    num_samples_per_client = int(sum_samples / args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_clients):
        dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace = False)
        #dict_users[i] = dict_users[i].astype(int)
        #dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)

    return data_loaders

def iid_nesize_split(dataset, args, kwargs, is_shuffle = True):
    sum_samples = len(dataset)
    num_samples_per_client = gen_ran_sum(sum_samples, args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for (i, num_samples_client) in enumerate(num_samples_per_client):
        dict_users[i] = np.random.choice(all_idxs, num_samples_client, replace = False)
        #dict_users[i] = dict_users[i].astype(int)
        #dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)

    return data_loaders

def niid_esize_split(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0] * args.num_clients
    
    # Get labels
    if hasattr(dataset, 'train_labels'):  # MNIST/CIFAR10 style
        if is_shuffle:
            labels = dataset.train_labels
        else:
            labels = dataset.test_labels
    elif isinstance(dataset, ImageFolder):  # ImageFolder style
        labels = np.array(dataset.targets)
    else:
        raise ValueError(f"Dataset type {type(dataset)} not supported")

    # Create index array for all samples
    all_idxs = np.arange(len(dataset))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Group indices by label
    label_indices = {label: all_idxs[labels == label] for label in unique_labels}
    
    # Print distribution information
    print("\nClass distribution in dataset:")
    for label in unique_labels:
        print(f"Class {label}: {len(label_indices[label])} samples")
    
    # Ensure we can assign at least 2 classes per client
    if num_classes < 2:
        raise ValueError("Need at least 2 classes for non-IID distribution")
    
    # Calculate how many clients should get each class
    # Each client needs 2 classes, so each class needs to go to (2 * num_clients) / num_classes clients
    clients_per_class = (2 * args.num_clients) // num_classes
    if clients_per_class == 0:
        clients_per_class = 1
    
    # Create class assignments for each client
    client_class_assignments = []
    for i in range(args.num_clients):
        # Select 2 different classes for this client
        available_classes = list(unique_labels)
        class1 = np.random.choice(available_classes)
        available_classes.remove(class1)
        class2 = np.random.choice(available_classes)
        client_class_assignments.append((class1, class2))
    
    # Calculate samples per class per client
    min_samples_per_class = min(len(indices) for indices in label_indices.values())
    samples_per_class_per_client = min_samples_per_class // clients_per_class
    
    # Keep track of how many times each class has been assigned
    class_assignment_counts = {label: 0 for label in unique_labels}
    
    # Create data loaders for each client
    for i in range(args.num_clients):
        class1, class2 = client_class_assignments[i]
        
        # Get indices for both classes
        indices1 = label_indices[class1]
        indices2 = label_indices[class2]
        
        # Calculate start and end indices for each class
        start1 = class_assignment_counts[class1] * samples_per_class_per_client
        end1 = start1 + samples_per_class_per_client
        start2 = class_assignment_counts[class2] * samples_per_class_per_client
        end2 = start2 + samples_per_class_per_client
        
        # Update assignment counts
        class_assignment_counts[class1] += 1
        class_assignment_counts[class2] += 1
        
        # Combine indices from both classes
        client_indices = np.concatenate([
            indices1[start1:end1],
            indices2[start2:end2]
        ])
        
        # Create dataloader
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, client_indices),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
        
        # Print distribution for this client
        client_labels = labels[client_indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        print(f"\nClient {i} distribution:")
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} samples")
    
    return data_loaders

def niid_esize_split_train(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0]* args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
#     no need to judge train ans test here
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
#     divide and assign
#     and record the split patter
    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace= False)
        split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern

def niid_esize_split_test(dataset, args, kwargs, split_pattern,  is_shuffle = False ):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
#     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i][0]
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None

def niid_esize_split_train_large(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0]* args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace= False)
        # split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
            # store the label
            split_pattern[i].append(dataset.__getitem__(idxs[rand * num_imgs])[1])
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern

def niid_esize_split_test_large(dataset, args, kwargs, split_pattern, is_shuffle = False ):
    """
    :param dataset: test dataset
    :param args:
    :param kwargs:
    :param split_pattern: split pattern from trainloaders
    :param test_size: length of testloader of each client
    :param is_shuffle: False for testloader
    :return:
    """
    data_loaders = [0] * args.num_clients
    # for mnist and cifar 10, only 10 classes
    num_shards = 10
    num_imgs = int (len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(len(dataset))
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
#     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i]
        # idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None

def niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0] * args.num_clients
    
    # Get labels
    if hasattr(dataset, 'train_labels'):  # MNIST/CIFAR10 style
        if is_shuffle:
            labels = dataset.train_labels
        else:
            labels = dataset.test_labels
    elif isinstance(dataset, ImageFolder):  # ImageFolder style
        labels = np.array(dataset.targets)
    else:
        raise ValueError(f"Dataset type {type(dataset)} not supported")

    # Create index array for all samples
    all_idxs = np.arange(len(dataset))
    
    # Group indices by label
    label_indices = {label: all_idxs[labels == label] for label in np.unique(labels)}
    
    # Calculate samples per client to ensure even distribution
    min_samples = min(len(indices) for indices in label_indices.values())
    samples_per_client = min_samples // (args.num_clients // len(label_indices))
    
    # Create shards (one per client)
    shards = []
    for label, indices in label_indices.items():
        # Shuffle indices for this label
        np.random.shuffle(indices)
        # Split into roughly equal shards
        n_clients_for_label = args.num_clients // len(label_indices)
        for i in range(n_clients_for_label):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            shards.append((label, indices[start_idx:end_idx]))
    
    # Shuffle shards
    np.random.shuffle(shards)
    
    # Assign shards to clients
    for i in range(args.num_clients):
        label, indices = shards[i]
        # Create dataloader for this client
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, indices),
            batch_size=args.batch_size,
            shuffle=is_shuffle,
            **kwargs
        )
    
    return data_loaders

def split_data(dataset, args, kwargs, is_shuffle = True):
    """
    Split dataset according to the specified distribution type
    Args:
        dataset: The dataset to split
        args: Arguments containing iid, num_clients, etc.
        kwargs: Additional arguments for DataLoader
        is_shuffle: Whether to shuffle the data within each client's dataset
    Returns:
        List of DataLoaders, one for each client
    """
    if args.iid == 1:  # IID with equal size
        return iid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == 0:  # Non-IID with balanced classes
        return niid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -1:  # Non-IID with unbalanced classes
        return niid_esize_split_train(dataset, args, kwargs, is_shuffle)
    elif args.iid == -2:  # One class per client
        if args.edgeiid == 1:  # IID within edges
            return niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle)
        else:  # Non-IID within edges
            return niid_esize_split_train_large(dataset, args, kwargs, is_shuffle)
    else:
        raise ValueError(f"Invalid iid value: {args.iid}. Must be one of: 1 (IID), 0 (Non-IID balanced), -1 (Non-IID unbalanced), -2 (One-class)")

def get_dataset(dataset_root, dataset, args):
    trains, train_loaders, tests, test_loaders = {}, {}, {}, {}
    if dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar10(dataset_root, args)
    elif dataset == 'isic':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_isic(dataset_root, args)
    elif dataset == 'femnist':
        raise ValueError('CODING ERROR: FEMNIST dataset should not use this file')
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader

def get_mnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = True,
                            download = True, transform = transform)
    test =  datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = False,
                            download = True, transform = transform)
    #note: is_shuffle here also is a flag for differentiating train and test
    train_loaders = split_data(train, args, kwargs, is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs, is_shuffle = False)
    #the actual batch_size may need to change.... Depend on the actual gradient...
    #originally written to get the gradient of the whole dataset
    #but now it seems to be able to improve speed of getting accuracy of virtual sequence
    v_train_loader = DataLoader(train, batch_size = args.batch_size * args.num_clients,
                                shuffle = True, **kwargs)
    v_test_loader = DataLoader(test, batch_size = args.batch_size * args.num_clients,
                                shuffle = False, **kwargs)
    return  train_loaders, test_loaders, v_train_loader, v_test_loader

def get_cifar10(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory':True} if is_cuda else{}
    if args.model == 'cnn_complex':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.model == 'resnet18':
        transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding = 4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError("this nn for cifar10 not implemented")
    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train = True,
                        download = True, transform = transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root,'cifar10'), train = False,
                        download = True, transform = transform_test)
    v_train_loader = DataLoader(train, batch_size = args.batch_size,
                                shuffle = True, **kwargs)
    v_test_loader = DataLoader(test, batch_size = args.batch_size,
                                shuffle = False, **kwargs)
    train_loaders = split_data(train, args, kwargs, is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs, is_shuffle = False)
    return  train_loaders, test_loaders, v_train_loader, v_test_loader

def get_isic(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

    # Construct full path to ISIC dataset
    isic_path = os.path.join(dataset_root, args.isic_dirname)
    print(f'Looking for ISIC dataset in: {isic_path}')

    # Check if dataset exists
    train_dir = os.path.join(isic_path, 'Train')
    test_dir = os.path.join(isic_path, 'Test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise ValueError(f'ISIC dataset not found at {isic_path}. Please ensure the dataset is properly organized with Train and Test folders.')

    # Define normalization stats
    if args.use_imagenet_stats:
        print('Using ImageNet normalization statistics')
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    else:
        print('Calculating dataset-specific normalization statistics...')
        # Create temporary dataset to compute statistics
        transform_initial = transforms.Compose([
            transforms.Resize((args.isic_image_size, args.isic_image_size)) if hasattr(args, 'isic_image_size') else transforms.Compose([]),
            transforms.ToTensor(),
        ])
        temp_dataset = ImageFolder(root=train_dir, transform=transform_initial)
        mean, std = get_mean_and_std(temp_dataset)
        norm_mean = mean.tolist()
        norm_std = std.tolist()
        print(f'Dataset statistics - Mean: {norm_mean}, Std: {norm_std}')

    # Define transforms with selected normalization stats
    transform_train = transforms.Compose([
        transforms.Resize((args.isic_image_size, args.isic_image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.isic_image_size, args.isic_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Load the datasets
    train_dataset = ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = ImageFolder(root=test_dir, transform=transform_test)

    # Update output channels based on number of classes
    args.output_channels = len(train_dataset.classes)
    print(f'ISIC dataset loaded with {len(train_dataset.classes)} classes')
    print(f'Class mapping: {train_dataset.class_to_idx}')

    # Create main dataloaders with same batch size pattern as CIFAR10/MNIST
    v_train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, **kwargs)
    v_test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, **kwargs)

    # Split data for federated learning
    train_loaders = split_data(train_dataset, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test_dataset, args, kwargs, is_shuffle=False)

    return train_loaders, test_loaders, v_train_loader, v_test_loader

def show_distribution(dataloader, args):
    """
    show the distribution of the data on certain client with dataloader
    return:
        percentage of each class of the label
    """
    if args.dataset == 'mnist':
        try:
            labels = dataloader.dataset.dataset.train_labels.numpy()
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels.numpy()
    elif args.dataset == 'cifar10':
        try:
            labels = dataloader.dataset.dataset.train_labels
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels
    elif isinstance(dataloader.dataset.dataset, ImageFolder):  # ImageFolder style
        labels = np.array(dataloader.dataset.dataset.targets)
    elif hasattr(dataloader.dataset, 'labels'):  # FSDD style
        labels = dataloader.dataset.labels
    else:
        raise ValueError(f"Dataset type not supported for distribution visualization")

    num_samples = len(dataloader.dataset)
    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    for idx in idxs:
        img, label = dataloader.dataset[idx]
        distribution[label] += 1
    distribution = np.array(distribution)
    distribution = distribution / num_samples
    return distribution

if __name__ == '__main__':
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loaders, test_loaders, _, _ = get_dataset(args.dataset_root, args.dataset, args)
    print(f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way")
    for i in range(args.num_clients):
        train_loader = train_loaders[i]
        print(len(train_loader.dataset))
        distribution = show_distribution(train_loader, args)
        print("dataloader {} distribution".format(i))
        print(distribution)

