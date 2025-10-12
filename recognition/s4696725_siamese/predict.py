"""
Example usage of the trained model: load checkpoints, compute prototypes from training set,
run inference on test images, print basic metrics if ground truth available and show some visualizations.
"""

import os
import random
import torch
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from modules import SiameseNet
from dataset import SingleImageDataset
from train import load_model, compute_class_prototypes, predict_on_test


def show_samples(test_dir, predictions_csv, n=8, out='sample_predictions.png'):
    df = pd.read_csv(predictions_csv)
    sample = df.sample(min(n, len(df)))
    plt.figure(figsize=(12, 6))

    for i, (_, row) in enumerate(sample.iterrows()):
        fn = row['image_name']
        p = row['melanoma_prob']
        img = Image.open(os.path.join(test_dir, fn)).convert('RGB')
        plt.subplot(2, (n+1)//2, i+1)
        plt.imshow(img)
        plt.title(f"{fn}\nP(melanoma)={p:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(out)
    print('Saved sample predictions to', out)


if __name__ == '__main__':
    model_dir = './recognition/s4696725_siamese/model'
    model_path = os.path.join(model_dir, 'siamese.pth')
    train_images = './recognition/s4696725_siamese/train'
    train_csv = './recognition/s4696725_siamese/metadata/train.csv'
    test_images = './recognition/s4696725_siamese/test'
    test_csv = './recognition/s4696725_siamese/metadata/test.csv'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    prototypes = compute_class_prototypes(model, device, train_images, train_csv, num_per_class=200)
    filenames, probs = predict_on_test(model, device, test_images, test_csv, prototypes)
    out_csv = os.path.join(model_dir, 'test_predictions.csv')
    pd.DataFrame({'image_name': filenames, 'prob': probs}).to_csv(out_csv, index=False)
    show_samples(test_images, out_csv)