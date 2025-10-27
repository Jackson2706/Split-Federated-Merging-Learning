from torchvision import transforms
from torch.nn import functional as F

def get_simclr_transforms(input_size: int = 32):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], 
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

