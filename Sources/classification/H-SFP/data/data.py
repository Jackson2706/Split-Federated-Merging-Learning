from torch.utils.data import DataLoader, Dataset
import torch

# --- Dataset definition ---
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        # Ensure a tensor is returned; some older datasets return a PIL Image
        return image.clone(), torch.tensor(label)