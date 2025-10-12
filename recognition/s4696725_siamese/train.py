"""
Train, validate, test and save the model.
Produces training plots and saves model to ./recognition/s4696725_siamese/model.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import SiameseNet
from dataset import SiamesePairDataset, SingleImageDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


"""
Main training loop
"""
def train_loop(model, device, loader, optimizer, criterion, epoch, log_interval=50):
    model.train()
    running_loss = 0.0
    preds, trues = [], []

    for batch_idx, (x1, x2, y) in enumerate(tqdm(loader, desc=f'Training {batch_idx}')):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        _, _, logits = model(x1, x2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        with torch.no_grad():
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            pred = (prob > 0.5).astype(int)
            preds.extend(pred.tolist())
            trues.extend(y.detach().cpu().numpy().astype(int).tolist())

        if (batch_idx + 1) % log_interval == 0:
            print(f"Epoch {epoch} Batch {batch_idx+1}/{len(loader)} Loss {running_loss/(batch_idx+1):.4f}")

    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    return running_loss / len(loader), acc, prec, rec, f1


"""
Main evaluation loop
"""
def eval_loop(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds, trues = [], []
    
    with torch.no_grad():
        for x1, x2, y in tqdm(loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            _, _, logits = model(x1, x2)
            loss = criterion(logits, y)
            running_loss += loss.item()
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob > 0.5).astype(int)
            preds.extend(pred.tolist())
            trues.extend(y.detach().cpu().numpy().astype(int).tolist())

    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    return running_loss / len(loader), acc, prec, rec, f1


"""
Saves the trained model to <path>/siamese.pth.
"""
def save_model(model, path: str):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'siamese.pth'))


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
                embeddings[label].append(emb[i].cpu())

            if len(embeddings[0]) >= num_per_class and len(embeddings[1]) >= num_per_class:
                break

    for k in [0, 1]:
        if len(embeddings[k]) == 0:
            proto[k] = torch.zeros(model.embedding_net.fc[0].out_features)

        else:
            proto[k] = torch.stack(embeddings[k])[:num_per_class].mean(dim=0)

    return proto


def predict_on_test(model, device, test_image_dir, test_csv=None, prototypes=None, batch_size=32):
    ds = SingleImageDataset(test_image_dir, test_csv)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    filenames = []
    probs = []

    with torch.no_grad():
        for imgs, fns, meta in loader:
            imgs = imgs.to(device)
            emb = model.embedding_net(imgs).cpu()
            # distances to prototypes
            d0 = torch.norm(emb - prototypes[0].unsqueeze(0), dim=1)
            d1 = torch.norm(emb - prototypes[1].unsqueeze(0), dim=1)
            # convert to probability of class 1 (melanoma) by softmax over negative distance
            scores = torch.stack([ -d0, -d1 ], dim=1)
            probs_batch = torch.softmax(scores, dim=1)[:,1].numpy()
            filenames.extend(fns)
            probs.extend(probs_batch.tolist())

    return filenames, probs


"""

"""
def plot_metrics(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history['train_loss']) + 1))

    plt.figure()
    plt.plot(epochs, history['train_loss'], label='train_loss')

    if history.get('val_loss') is not None:
        plt.plot(epochs, history['val_loss'], label='val_loss')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, history['train_f1'], label='train_f1')

    if history.get('val_f1') is not None:
        plt.plot(epochs, history['val_f1'], label='val_f1')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('f1')
    plt.savefig(os.path.join(out_dir, 'f1.png'))
    plt.close()


"""

"""
def main():
    train_images = './recognition/s4696725_siamese/train'
    train_csv = './recognition/s4696725_siamese/metadata/train.csv'
    test_images = './recognition/s4696725_siamese/test'
    test_csv = './recognition/s4696725_siamese/metadata/test.csv'
    model_dir = './recognition/s4696725_siamese/model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print('Using ', device)

    # hyperparams
    epochs = 10
    batch_size = 32
    lr = 1e-4

    model = SiameseNet(embedding_size=512, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()


    # loaders
    train_dataset = SiamesePairDataset(train_images, train_csv, pairs_per_epoch=20000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # optional small val split
    val_loader = None


    history = {'train_loss': [], 'train_f1': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_loop(model, device, train_loader, optimizer, criterion, epoch)
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        print(f"Epoch {epoch} train_loss={train_loss:.4f} f1={train_f1:.4f} time={(time.time()-t0):.1f}s")


        # optional validation
        if val_loader is not None:
            val_loss, val_acc, val_prec, val_rec, val_f1 = eval_loop(model, device, val_loader, criterion)
            history.setdefault('val_loss', []).append(val_loss)
            history.setdefault('val_f1', []).append(val_f1)
            print(f" val_loss={val_loss:.4f} val_f1={val_f1:.4f}")


        # checkpoint each epoch
        save_model(model, model_dir)


    # after training compute prototypes on train set and run on test
    print('Computing prototypes...')
    prototypes = compute_class_prototypes(model, device, train_images, train_csv, num_per_class=200)
    print('Predicting on test set...')
    filenames, probs = predict_on_test(model, device, test_images, test_csv, prototypes)


    # save predictions
    os.makedirs(model_dir, exist_ok=True)
    out_df = {'image_name': filenames, 'prob': probs}
    pd.DataFrame(out_df).to_csv(os.path.join(model_dir, 'test_predictions.csv'), index=False)
    plot_metrics(history, model_dir)

    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history, f)

    print('Training complete. Model and artifacts saved to', model_dir)


if __name__ == '__main__':
    main()