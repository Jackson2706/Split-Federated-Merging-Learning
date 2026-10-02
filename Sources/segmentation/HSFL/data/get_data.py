#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os

import torch

from .sampling import *
from .utils import ISICSegmentationDataset


def get_dataset(args):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args["dataset"] == "isic-2018":
        train_dataset = ISICSegmentationDataset(
            image_data_folder_path=args["train_image_data_folder_path"],
            mask_data_folder_path=args["train_mask_data_folder_path"],
            phase="train",
        )
        print(f"Number of training images: {len(train_dataset)}")
        val_dataset = ISICSegmentationDataset(
            image_data_folder_path=args["val_image_data_folder_path"],
            mask_data_folder_path=args["val_mask_data_folder_path"],
            phase="val",
        )
        test_dataset = ISICSegmentationDataset(
            image_data_folder_path=args["test_image_data_folder_path"],
            mask_data_folder_path=args["test_mask_data_folder_path"],
            phase="test",
        )
        user_groups = isic_iid(train_dataset, args["num_users"])
    else:
        raise ValueError("Unsupported dataset type.")
    return train_dataset, val_dataset, test_dataset, user_groups


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
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")

    print("    Federated parameters:")
    if args.iid:
        print("    IID")
    else:
        print("    Non-IID")
    print(f"    Fraction of users  : {args.frac}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.local_ep}\n")
    return
