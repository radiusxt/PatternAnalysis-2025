"""
Prepares dataset in /recognition/s4696725_siamese/train/*.jpg and /recognition/s4696725_siamese/metadata/*.csv.
Data loaders for training and single-image inference.
"""

import os
import torch
import random
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


"""
Produces pairs for siamese training.
- image_dir: folder with images
- csv_path: metadata csv that contains at least columns: image_name (or image id) and melanoma label (0 or 1)
- transform: torchvision transforms applied to images
- pairs_per_epoch: approximate number of pairs to generate per epoch
"""
class SiamesePairDataset(Dataset):
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
Dataset for running inference on single images in the test set.
"""
class SingleImageDataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str = None, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
        self.metadata = pd.read_csv(csv_path)
        self.metadata['image_name'] += '.jpg'
        self.metadata.set_index('image_name', inplace=True)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = int(self.metadata.loc[filename]['target'])
        return img, label
    

"""
Utility function to generate dataloaders.
"""
def generate_dataloaders(train_images: str, train_csv: str, split=0.15):
    dataset = SiamesePairDataset(train_images, train_csv, pairs_per_epoch=15000)
    n_val = int(len(dataset) * split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    return train_loader, val_loader