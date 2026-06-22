# Datasets

All configs reference datasets through a portable, repo-relative `data/` layout:

```
data/
├── cifar/        # CIFAR-10 / CIFAR-100 (auto-downloaded by torchvision)
├── HAM10000/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
├── ISIC2018/
│   ├── ISIC2018_Task1-2_Training_Input/
│   ├── ISIC2018_Task1_Training_GroundTruth/
│   ├── ISIC2018_Task1-2_Validation_Input/
│   └── ISIC2018_Task1_Validation_GroundTruth/
└── ImageNet/     # standard torchvision ImageNet layout (val split used)
```

`data/` is git-ignored. Populate it in either of two ways.

## Option A — symlink existing copies (recommended if data already on disk)

```bash
mkdir -p data
ln -sfn /path/to/cifar     data/cifar
ln -sfn /path/to/HAM10000  data/HAM10000
ln -sfn /path/to/ISIC2018  data/ISIC2018
ln -sfn /path/to/imagenet  data/ImageNet
```

## Option B — download into `data/`

| Dataset | How to obtain |
|---------|---------------|
| **CIFAR-10/100** | Auto-downloaded by torchvision on first run into `data/cifar/`. No action needed. |
| **HAM10000** | Download from the [ISIC archive / Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000); place `HAM10000_metadata.csv` and the two image folders under `data/HAM10000/`. |
| **ISIC-2018** | Download Task 1 (segmentation) inputs + ground truth from the [ISIC 2018 challenge](https://challenge.isic-archive.com/data/#2018); place the four folders under `data/ISIC2018/`. |
| **ImageNet** | Provide a standard torchvision ImageNet directory at `data/ImageNet/` (validation split is used). |

## Overriding paths without editing configs

Every dataset path is a config key, so you can override per-run instead of moving data:

```bash
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --set dataset_root=/abs/path/to/cifar/
```

Relevant keys: `dataset_root` (CIFAR/ImageNet), `metadata_path` + `image_dirs`
(HAM10000), `train_image_data_folder_path` / `train_mask_data_folder_path` /
`test_image_data_folder_path` / `test_mask_data_folder_path` (ISIC-2018).
