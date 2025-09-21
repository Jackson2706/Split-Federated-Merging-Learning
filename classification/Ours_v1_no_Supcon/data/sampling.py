#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(all_idxs, num_items, replace=False)
        )
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                axis=0,
            )
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(
        min_shard, max_shard + 1, size=num_users
    )
    random_shard_size = np.around(
        random_shard_size / sum(random_shard_size) * num_shards
    )
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (
                        dict_users[i],
                        idxs[rand * num_imgs : (rand + 1) * num_imgs],
                    ),
                    axis=0,
                )

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(
                np.random.choice(idx_shard, shard_size, replace=False)
            )
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (
                        dict_users[i],
                        idxs[rand * num_imgs : (rand + 1) * num_imgs],
                    ),
                    axis=0,
                )
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(
                np.random.choice(idx_shard, shard_size, replace=False)
            )
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (
                        dict_users[i],
                        idxs[rand * num_imgs : (rand + 1) * num_imgs],
                    ),
                    axis=0,
                )

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(
                np.random.choice(idx_shard, shard_size, replace=False)
            )
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (
                        dict_users[k],
                        idxs[rand * num_imgs : (rand + 1) * num_imgs],
                    ),
                    axis=0,
                )

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(all_idxs, num_items, replace=False)
        )
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


from collections import defaultdict

import numpy as np
from torch.utils.data import Subset


def get_targets_from_dataset(dataset):
    """Extract targets from raw or Subset dataset."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    elif isinstance(dataset, Subset):
        return np.array([dataset.dataset.targets[i] for i in dataset.indices])
    else:
        raise AttributeError("Dataset has no 'targets' attribute.")


def get_pair_list(dataset):
    """Extract pair list from raw or Subset dataset."""
    if hasattr(dataset, "pair_list"):
        return dataset.pair_list
    elif isinstance(dataset, Subset):
        return dataset.dataset.pair_list
    else:
        raise AttributeError("Dataset has no 'pair_list' attribute.")


def cifar_noniid(pair_dataset, num_users, num_classes=10, shards_per_user=25):
    image_labels = get_targets_from_dataset(pair_dataset)
    full_pair_list = get_pair_list(pair_dataset)
    total_images = len(image_labels)

    # Step 1: Create label-based shards
    num_shards = num_users * shards_per_user
    num_imgs_per_shard = total_images // num_shards
    sorted_indices = np.argsort(image_labels)

    shards = [
        sorted_indices[i * num_imgs_per_shard : (i + 1) * num_imgs_per_shard]
        for i in range(num_shards)
    ]

    # Step 2: Randomly assign shards to users
    shard_ids = np.arange(num_shards)
    np.random.shuffle(shard_ids)

    user_img_indices = defaultdict(set)
    for user_id in range(num_users):
        assigned = shard_ids[
            user_id * shards_per_user : (user_id + 1) * shards_per_user
        ]
        for shard_id in assigned:
            user_img_indices[user_id].update(shards[shard_id])

    # Step 3: For each user, collect pairs (i, j) where both images are owned
    dict_users = {}
    dataset_len = len(pair_dataset)

    for user_id in range(num_users):
        owned_imgs = user_img_indices[user_id]
        user_pairs = []

        for pair_idx, (i, j) in enumerate(full_pair_list):
            if pair_idx >= dataset_len:
                continue
            if (
                i in owned_imgs and j in owned_imgs
            ):  # ensure pair fully belongs to user
                user_pairs.append(pair_idx)

        dict_users[user_id] = Subset(pair_dataset, user_pairs)

    return dict_users


def ham10000_iid(dataset, num_users):
    """
    Sample I.I.D. client data from HAM10000 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(all_idxs, num_items, replace=False)
        )
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def ham10000_noniid(pair_dataset, num_users, num_classes=7, shards_per_user=2):
    image_labels = get_targets_from_dataset(pair_dataset)
    full_pair_list = get_pair_list(pair_dataset)
    total_images = len(image_labels)

    # Step 1: Create label-based shards
    num_shards = num_users * shards_per_user
    num_imgs_per_shard = total_images // num_shards
    sorted_indices = np.argsort(image_labels)

    shards = [
        sorted_indices[i * num_imgs_per_shard : (i + 1) * num_imgs_per_shard]
        for i in range(num_shards)
    ]

    # Step 2: Randomly assign shards to users
    shard_ids = np.arange(num_shards)
    np.random.shuffle(shard_ids)

    user_img_indices = defaultdict(set)
    for user_id in range(num_users):
        assigned = shard_ids[
            user_id * shards_per_user : (user_id + 1) * shards_per_user
        ]
        for shard_id in assigned:
            user_img_indices[user_id].update(shards[shard_id])

    # Step 3: For each user, collect pairs (i, j) where both images are owned
    dict_users = {}
    dataset_len = len(pair_dataset)

    for user_id in range(num_users):
        owned_imgs = user_img_indices[user_id]
        user_pairs = []

        for pair_idx, (i, j) in enumerate(full_pair_list):
            if pair_idx >= dataset_len:
                continue
            if (
                i in owned_imgs and j in owned_imgs
            ):  # ensure pair fully belongs to user
                user_pairs.append(pair_idx)

        dict_users[user_id] = Subset(pair_dataset, user_pairs)

    return dict_users
