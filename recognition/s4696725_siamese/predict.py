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

"""
Compute mean embedding per class using a subset of training images.
Returns dict {0: tensor, 1: tensor}

def compute_class_prototypes(model, device, image_dir, csv_path, num_per_class: int = 200):
    dataset = SingleImageDataset(image_dir, csv_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    embeddings, proto = {0: [], 1: []}, {}

    with torch.no_grad():
        for img, _, metadata in loader:
            img = img.to(device)
            emb = model.embedding_net(img)
            
            for i, m in enumerate(metadata):
                label = int(m['target'])
                embeddings[label].append(emb[i])

            if len(embeddings[0]) >= num_per_class and len(embeddings[1]) >= num_per_class:
                break

    for k in [0, 1]:
        if embeddings[k]:
            proto[k] = torch.stack(embeddings[k])[:num_per_class].mean(dim=0).cpu()

        else:
            proto[k] = torch.zeros(model.embedding_net.fc[0].out_features)

    return proto



Runs Siamese model on the test set using similarity.

def predict_on_test(model, device, test_image_dir, test_csv=None, prototypes=None, batch_size=32):
    ds = SingleImageDataset(test_image_dir, test_csv)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    filenames, probs = [], []

    with torch.no_grad():
        proto0 = prototypes[0].unsqueeze(0).to(device)
        proto1 = prototypes[1].unsqueeze(0).to(device)

        for imgs, fns, _ in loader:
            imgs = imgs.to(device)
            emb = model.embedding_net(imgs)

            # distances to prototypes
            d0 = torch.norm(emb - proto0, dim=1)
            d1 = torch.norm(emb - proto1, dim=1)

            # convert to probability of class 1 (melanoma) by softmax over negative distance
            probs_batch = torch.softmax(torch.stack([-d0, -d1], dim=1), dim=1)[:, 1]
            filenames.extend(fns)
            probs.extend(probs_batch.cpu().numpy().tolist())

    return filenames, probs
"""

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



    # load of junk
    """
    test_images = './recognition/s4696725_siamese/test'
    test_csv = './recognition/s4696725_siamese/metadata/test.csv'
    # after training compute prototypes on train set and run on test
    print('Computing prototypes...')
    prototypes = compute_class_prototypes(model, device, train_images, train_csv, num_per_class=200)
    print('Predicting on test set...')
    filenames, probs = predict_on_test(model, device, test_images, test_csv, prototypes)

    # save predictions
    os.makedirs(model_dir, exist_ok=True)
    out_df = {'image_name': filenames, 'prob': probs}
    
    pd.DataFrame(out_df).to_csv(os.path.join(model_dir, 'test_predictions.csv'), index=False)
    """