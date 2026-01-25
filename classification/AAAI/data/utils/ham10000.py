import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

# Define label classes
classes = ['bkl', 'bcc', 'akiec', 'vasc', 'nv', 'mel', 'df']

class SkinCancerPairDataset(Dataset):
    def __init__(self, metadata, image_dirs, transform=None, num_pairs=10000):
        """
        Args:
            metadata: Pandas DataFrame with 'image_id' and 'dx'
            image_dirs: List or single directory where images are stored
            transform: Torchvision transforms to apply
            num_pairs: Number of unique unordered pairs to generate
        """
        self.metadata = metadata.reset_index(drop=True)
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.transform = transform
        self.label_map = {c: i for i, c in enumerate(classes)}
        self.num_pairs = num_pairs
        self.pair_list = self._generate_unique_pairs(len(self.metadata), num_pairs)

    def _generate_unique_pairs(self, dataset_size, num_pairs):
        seen = set()
        pairs = []

        while len(pairs) < num_pairs:
            i = random.randint(0, dataset_size - 1)
            j = random.randint(0, dataset_size - 1)
            if i == j:
                continue
            a, b = sorted((i, j))
            if (a, b) not in seen:
                seen.add((a, b))
                pairs.append((a, b))

        return pairs

    def _load_image_by_id(self, image_id):
        for dir_path in self.image_dirs:
            potential_path = os.path.join(dir_path, f"{image_id}.jpg")
            if os.path.isfile(potential_path):
                return Image.open(potential_path).convert('RGB')
        raise FileNotFoundError(f"Image {image_id}.jpg not found in any of: {self.image_dirs}")

    def __getitem__(self, index):
        idx1, idx2 = self.pair_list[index]
        row1 = self.metadata.iloc[idx1]
        row2 = self.metadata.iloc[idx2]

        image1 = self._load_image_by_id(row1['image_id'])
        image2 = self._load_image_by_id(row2['image_id'])

        label1 = self.label_map[row1['dx']]
        label2 = self.label_map[row2['dx']]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), (torch.tensor(label1), torch.tensor(label2))

    def __len__(self):
        return self.num_pairs


class SkinCancerDataset(Dataset):
    def __init__(self, metadata, image_dirs, transform=None):
        """
        Args:
            metadata: Pandas DataFrame containing image_id and dx (diagnosis/label)
            image_dirs: List of directories to search for images
            transform: Torchvision transforms to apply to the image
        """
        self.metadata = metadata.reset_index(drop=True)  # ensure consistent indexing
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.transform = transform
        self.label_map = {c: i for i, c in enumerate(classes)}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        label_str = row['dx']

        # Search for image in provided directories
        img_path = None
        for dir_path in self.image_dirs:
            potential_path = os.path.join(dir_path, f"{image_id}.jpg")
            if os.path.isfile(potential_path):
                img_path = potential_path
                break

        if img_path is None:
            raise FileNotFoundError(f"Image {image_id}.jpg not found in any of: {self.image_dirs}")

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Get label index
        label = self.label_map.get(label_str)
        if label is None:
            raise ValueError(f"Unknown label '{label_str}' in row: {row}")

        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
