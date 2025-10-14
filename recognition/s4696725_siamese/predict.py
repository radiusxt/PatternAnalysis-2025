"""
Example usage of the trained model: load checkpoints, compute prototypes from training set,
run inference on test images, print basic metrics if ground truth available and show some visualizations.
"""

import torch
import pandas as pd
from modules import SiameseNet
from dataset import SingleImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


"""
Loads a model from a given path.
"""
def load_model(path: str, device, embedding_size=512):
    model = SiameseNet(embedding_size=embedding_size, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


"""
Compute mean embedding per class using a subset of training images.
Returns dict {0: tensor, 1: tensor}
"""
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
        if len(embeddings[k]) > 0:
            proto[k] = torch.stack(embeddings[k])[:num_per_class].mean(dim=0).cpu()

        else:
            proto[k] = torch.zeros_like(next(model.embedding_net.parameters())).mean(dim=0)

    return proto


"""
Runs Siamese model on the test set using similarity.
"""
def predict_on_test(model, device, test_image_dir, test_csv=None, prototypes=None):
    ds = SingleImageDataset(test_image_dir, test_csv)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
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


def main():
    model_path = './recognition/s4696725_siamese/model/siamese.pth'
    train_images = './recognition/s4696725_siamese/train'
    train_csv = './recognition/s4696725_siamese/metadata/train.csv'
    test_images = './recognition/s4696725_siamese/test'
    test_csv = './recognition/s4696725_siamese/metadata/test.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print('Using ', device)

    model = load_model(model_path, device)

    print("Computing class prototypes")
    prototypes = compute_class_prototypes(model, device, train_images, train_csv, num_per_class=200)

    print("Running inference on test set")
    _, probs = predict_on_test(model, device, test_images, test_csv, prototypes=prototypes)

    df = pd.read_csv(test_csv)
    df['pred'] = [1 if p >= 0.5 else 0 for p in probs]
    acc = accuracy_score(df['target'], df['pred'])
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()