"""
Prepares dataset in /recognition/s4696725_siamese/train/*.jpg and /recognition/s4696725_siamese/metadata/*.csv.
Data loaders for training and single-image inference.
"""

import os
import random
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset


class SiamesePairDataset(Dataset):
    """
    Produces pairs for siamese training.
    - image_dir: folder with images
    - csv_path: metadata csv that contains at least columns: image_name (or image id) and melanoma label (0/1)
    - transform: torchvision transforms applied to images
    - pairs_per_epoch: approximate number of pairs to generate per epoch
    """

    def __init__(self, image_dir: str, csv_path: str, transform=None, pairs_per_epoch: int = 20000):
        super().__init__()
        self.image_dir = image_dir
        self.metadata = pd.read_csv(csv_path)
        self.metadata['image_name'] += '.jpg'

        # Build lists by class
        self.class0 = self.metadata[self.metadata['target'] == 0]['filename'].tolist()
        self.class1 = self.metadata[self.metadata['target'] == 1]['filename'].tolist()

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
    

class SingleImageDataset(Dataset):
    """
    Dataset for running inference on single images in the test set.
    """

    def __init__(self, image_dir: str, csv_path: str = None, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.filenames = [f for f in os.listdir(image_dir)]
        self.filenames.sort()
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.metadata = pd.read_csv(csv_path)
        self.metadata['image_name'] += '.jpg'

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img = Image.open(os.path.join(self.image_dir, fn)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        meta_row = None

        if self.meta is not None:
            row = self.meta[self.meta['filename'] == fn]

            if not row.empty:
                meta_row = row.iloc[0].to_dict()

        return img, fn, meta_row
    

"""
Utility function to generate dataLoaders.
"""
def make_loaders(train_image_dir: str, train_csv: str, batch_size: int = 32, val_split: float = 0.1, pairs_per_epoch: int = 20000, num_workers: int = 4):
    dataset = SiamesePairDataset(train_image_dir, train_csv, pairs_per_epoch=pairs_per_epoch)
    n_val = int(len(dataset) * val_split) if val_split > 0 else 0

    if n_val == 0:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_loader, None
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader