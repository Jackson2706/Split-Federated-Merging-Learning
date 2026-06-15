mean = [0.7084, 0.5821, 0.5361]
std = [0.0967, 0.1118, 0.1261]
from torchvision import transforms

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
}

mask_transforms = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)


import glob

from PIL import Image
from torch.utils.data import Dataset


class ISICSegmentationDataset(Dataset):
    def __init__(
        self,
        image_data_folder_path="/media/jackson/Data/ISIC2018/ISIC2018_Task1-2_Training_Input",
        mask_data_folder_path="/media/jackson/Data/ISIC2018/ISIC2018_Task1_Training_GroundTruth",
        phase="train",
    ):
        self.image_data_folder_path = image_data_folder_path
        self.mask_data_folder_path = mask_data_folder_path
        self.phase = phase
        self.img_files = sorted(glob.glob(self.image_data_folder_path + "/*.jpg"))
        self.mask_imgs = sorted(glob.glob(self.mask_data_folder_path + "/*.png"))
        self.data_transforms = data_transforms[phase]
        self.mask_transforms = mask_transforms
        self.datalen = len(self.img_files)

    def __getitem__(self, index):
        img = self.data_transforms(Image.open(self.img_files[index]))
        mask = self.mask_transforms(Image.open(self.mask_imgs[index]))
        return img, mask

    def __len__(self):
        assert self.datalen == len(self.mask_imgs)
        return self.datalen
