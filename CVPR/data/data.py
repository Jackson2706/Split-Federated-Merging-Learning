from torch.utils.data import DataLoader, Dataset
import torch

# --- Định nghĩa Dataset ---
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        # Đảm bảo trả về tensor, một số dataset cũ trả về PIL Image
        return image.clone(), torch.tensor(label)