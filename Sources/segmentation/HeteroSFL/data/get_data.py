import copy

import torch

from .sampling import isic_iid
from .utils import ISICSegmentationDataset


def get_dataset(args):
    """Returns train and test datasets and a user group dict."""
    if args["dataset"] == "isic-2018":
        train_dataset = ISICSegmentationDataset(
            image_data_folder_path=args["train_image_data_folder_path"],
            mask_data_folder_path=args["train_mask_data_folder_path"],
            phase="train",
        )
        print(f"Number of training images: {len(train_dataset)}")
        test_dataset = ISICSegmentationDataset(
            image_data_folder_path=args["test_image_data_folder_path"],
            mask_data_folder_path=args["test_mask_data_folder_path"],
            phase="test",
        )
        user_groups = isic_iid(train_dataset, args["num_users"])
    else:
        raise ValueError("Unsupported dataset type.")
    return train_dataset, test_dataset, user_groups
