import os

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
        image_files = glob.glob(self.image_data_folder_path + "/*.jpg")
        mask_files = glob.glob(self.mask_data_folder_path + "/*.png")
        images_by_id = {
            os.path.splitext(os.path.basename(path))[0]: path for path in image_files
        }
        masks_by_id = {
            os.path.splitext(os.path.basename(path))[0].removesuffix("_segmentation"): path
            for path in mask_files
        }
        if images_by_id.keys() != masks_by_id.keys():
            missing_masks = sorted(images_by_id.keys() - masks_by_id.keys())
            missing_images = sorted(masks_by_id.keys() - images_by_id.keys())
            raise ValueError(
                "ISIC image/mask identifier mismatch: "
                f"missing masks={missing_masks[:5]}, missing images={missing_images[:5]}"
            )
        identifiers = sorted(images_by_id)
        self.img_files = [images_by_id[identifier] for identifier in identifiers]
        self.mask_imgs = [masks_by_id[identifier] for identifier in identifiers]
        self.data_transforms = data_transforms[phase]
        self.mask_transforms = mask_transforms
        self.datalen = len(self.img_files)

        if self.datalen == 0:
            raise FileNotFoundError(
                f"ISICSegmentationDataset ({phase}): found 0 images.\n"
                f"  images: {self.image_data_folder_path} "
                f"({'missing' if not os.path.isdir(self.image_data_folder_path) else 'no *.jpg'})\n"
                f"  masks:  {self.mask_data_folder_path} "
                f"({'missing' if not os.path.isdir(self.mask_data_folder_path) else 'no *.png'})\n"
                f"Download the ISIC-2018 Task 1 segmentation data and set the "
                f"train/test image & mask folder paths in your config to point at it."
            )
        if self.datalen != len(self.mask_imgs):
            raise ValueError(
                f"ISICSegmentationDataset ({phase}): image/mask count mismatch "
                f"({self.datalen} images vs {len(self.mask_imgs)} masks)."
            )

    def __getitem__(self, index):
        img = self.data_transforms(Image.open(self.img_files[index]))
        mask = self.mask_transforms(Image.open(self.mask_imgs[index]))
        return img, mask

    def __len__(self):
        assert self.datalen == len(self.mask_imgs)
        return self.datalen
