
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

# import time
# import numpy as np
# from LSTM_utilities import dataset, utility
# from options import args_parser
# from models import Basic_LSTM_2
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import random_split
from torchvision import datasets, transforms

from .sampling import *
from .utils import (CIFAR10PairDataset, CIFAR100PairDataset, SkinCancerDataset,
                    SkinCancerPairDataset)


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args["dataset"] == 'cifar10':
        data_dir = args["dataset_root"]
        

        train_dataset = CIFAR10PairDataset(
            data_root=data_dir,
            phase=True,
            num_pairs=25000,
        )
        train_len = int(0.9 * len(train_dataset))
        valid_len = len(train_dataset) - train_len
        
        train_dataset, valid_dataset = random_split(train_dataset, [train_len, valid_len])
        print(len(train_dataset)/len(valid_dataset))
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=valid_transform)

        # sample training data amongst users
        if args["iid"]:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args["num_users"])
        else:
            # Sample Non-IID user data from Mnist
            # if args.unequal:
            #     # Chose uneuqal splits for every user
            #     raise NotImplementedError()
            # else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args["num_users"])
    elif args["dataset"] == 'cifar100':
        data_dir = args["dataset_root"]
        train_dataset = CIFAR100PairDataset(
            data_root=data_dir,
            phase=True,
            num_pairs=25000,
        )
        train_len = int(0.9 * len(train_dataset))
        valid_len = len(train_dataset) - train_len
        
        train_dataset, valid_dataset = random_split(train_dataset, [train_len, valid_len])
        print(len(train_dataset)/len(valid_dataset))
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=valid_transform)

        # sample training data amongst users
        if args["iid"]:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args["num_users"])
        else:
            user_groups = cifar_noniid(train_dataset, args["num_users"])

    elif args["dataset"] == 'ham10000':
        import pandas as pd
        from sklearn.model_selection import train_test_split

        from .utils.ham10000 import SkinCancerDataset 
        metadata = pd.read_csv(args["metadata_path"])
        metadata['age'] = metadata['age'].fillna(metadata['age'].mean())
        metadata['sex'] = metadata['sex'].fillna('unknown')
        metadata['age'] = metadata['age'].clip(lower=0, upper=100)
            
        image_dirs = args["image_dirs"]
        apply_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_df, val_df = train_test_split(metadata, test_size=0.2, stratify=metadata['dx'], random_state=42)
        train_dataset = SkinCancerPairDataset(train_df, image_dirs, transform=apply_transform)
        test_dataset = SkinCancerDataset(val_df, image_dirs, transform=apply_transform)
        valid_dataset = SkinCancerDataset(val_df, image_dirs, transform=apply_transform)
        if args["iid"]:
            # Sample IID user data from HAM10000
            user_groups = ham10000_iid(train_dataset, args["num_users"])
        else:
            user_groups = ham10000_noniid(train_dataset, args["num_users"])
    return train_dataset, valid_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return







