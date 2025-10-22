"""
Prepares dataset in ./recognition/s4696725_siamese/train/*.jpg and ./recognition/s4696725_siamese/metadata/*.csv.
Data loaders for training and single-image inference.
"""

import os
import torch
import random
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


"""
Produces pairs for siamese training.

image_dir: Folder with images.
csv_path: CSV file for the training metadata.
transform: Torchvision transforms applied to images.
pairs_per_epoch: Number of pairs to generate per epoch.
"""
class SiameseDataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str, transform=None, pairs_per_epoch: int = 15000):
        super().__init__()
        self.image_dir = image_dir
        self.metadata = pd.read_csv(csv_path)
        self.metadata['image_name'] += '.jpg'

        # Build lists by class
        self.class0 = self.metadata[self.metadata['target'] == 0]['image_name'].tolist()
        self.class1 = self.metadata[self.metadata['target'] == 1]['image_name'].tolist()

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.pairs_per_epoch = pairs_per_epoch

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        # Randomly produce a positive (same class) or negative pair
        same = random.random() < 0.5

        if same:
            # pick a class then two distinct images from that class
            cls = 0 if random.random() < 0.5 else 1
            pool = self.class0 if cls == 0 else self.class1

            if len(pool) < 2:
                # fallback to the other class
                pool = self.class1 if cls == 0 else self.class0

            a, b = random.sample(pool, 2)
            label = 1

        else:
            a = random.choice(self.class0)
            b = random.choice(self.class1)
            label = 0

        img1 = Image.open(os.path.join(self.image_dir, a)).convert('RGB')
        img2 = Image.open(os.path.join(self.image_dir, b)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)


"""
Deterministic utility function to generate dataloaders.

train_images: Folder of images for training.
train_csv: CSV file with metadata for images.
"""
def generate_dataloaders(train_images: str, train_csv: str):
    seed = 137
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator().manual_seed(seed)

    dataset = SiameseDataset(train_images, train_csv, pairs_per_epoch=15000)

    train = int(len(dataset) * 0.75)
    val = int(len(dataset) * 0.15)
    test = len(dataset) - train - val 
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train, val, test], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader