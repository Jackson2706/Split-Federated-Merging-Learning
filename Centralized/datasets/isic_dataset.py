import os

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Optional: Map labels to integer indices
        self.label_map = {label: idx for idx, label in enumerate(self.annotations['dx'].unique())}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]['image_id'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label_str = self.annotations.iloc[idx]['dx']
        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label
