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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


"""
Training loop on training set.
"""
def train_loop(model, device, loader, optimizer, criterion, epoch, log_interval=50):
    model.train()
    running_loss = 0.0
    preds, trues = [], []

    for x1, x2, y in tqdm(loader, desc=f'Epoch {epoch}'):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        _, _, logits = model(x1, x2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        with torch.no_grad():
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob > 0.5).astype(int)
            preds.extend(pred)
            trues.extend(y.cpu().numpy().astype(int))

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, zero_division=0)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    return running_loss / len(loader), acc, f1, prec, rec

"""
Evaluation loop on training set.
"""
def eval_loop(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds, trues = [], []
    
    with torch.no_grad():
        for x1, x2, y in tqdm(loader, desc='Validation'):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            _, _, logits = model(x1, x2)
            loss = criterion(logits, y)
            running_loss += loss.item()
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob > 0.5).astype(int)
            preds.extend(pred)
            trues.extend(y.cpu().numpy().astype(int))

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, zero_division=0)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    return running_loss / len(loader), acc, f1, prec, rec


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
Plots metrics for model.
"""
def plot_metrics(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history['loss']) + 1))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Plot train loss on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, history['loss'], color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot train accuracy on right y-axis
    color = 'tab:orange'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, history['train_acc'], color=color, label='Train Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Training Loss and Accuracy')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_results.jpg'))
    plt.show()

    print(f"F1 Score:   {history['f1'][-1]:.4f}")
    print(f"Precision:  {history['prec'][-1]:.4f}")
    print(f"Recall:     {history['rec'][-1]:.4f}")


"""
Main training loop
"""
def main():
    train_images = './recognition/s4696725_siamese/train'
    train_csv = './recognition/s4696725_siamese/metadata/train.csv'
    model_dir = './recognition/s4696725_siamese/model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print('Using ', device)

    # hyperparameters
    epochs = 10
    batch_size = 32

    model = SiameseNet(embedding_size=512, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # dataset and dataloaders
    dataset = SiamesePairDataset(train_images, train_csv, pairs_per_epoch=20000)
    n_total = len(dataset)
    n_val = int(0.15 * n_total)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    best_f1 = 0.0
    history = {
        'loss': [],
        'acc': [],
        'f1': [], 
        'prec': [],
        'rec': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_prec': [],
        'val_rec': []
    }

    for epoch in range(1, epochs + 1):
        loss, acc, f1, prec, rec = train_loop(model, device, train_loader, optimizer, criterion, epoch)
        history['loss'].append(loss)
        history['acc'].append(acc)
        history['f1'].append(f1)
        history['prec'].append(prec)
        history['rec'].append(rec)

        print(f"Epoch {epoch}, Loss = {loss:.4f}, Acc = {acc:.4f}, F1 = {f1:.4f}")

        # validate every 5 epochs
        if epoch % 5 == 0:
            val_loss, val_acc, val_f1, val_prec, val_rec = eval_loop(model, device, val_loader, criterion)
            history.setdefault('val_loss', []).append(val_loss)
            history.setdefault('val_f1', []).append(val_f1)
            print(f"Validation: Loss = {val_loss:.4f}, Acc = {val_acc:.4f}, F1 = {val_f1:.4f}")

            # save best model
            if f1 > best_f1:
                best_f1 = f1
                save_model(model, model_dir)

    plot_metrics(history, model_dir)

    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history, f, default=float)

    print('Training complete.')


if __name__ == '__main__':
    main()