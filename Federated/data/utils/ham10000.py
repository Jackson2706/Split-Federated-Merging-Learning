import os
from PIL import Image
from torch.utils.data import Dataset

# Define label classes
classes = ['bkl', 'bcc', 'akiec', 'vasc', 'nv', 'mel', 'df']

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
