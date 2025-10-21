"""
Train, validate, test and save the model.
Produces training plots and saves model to ./recognition/s4696725_siamese/model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import SiameseNet
from dataset import generate_dataloaders
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


"""
Training loop on training set.
"""
def train_loop(model, device, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    preds, trues = [], []

    for x1, x2, y in tqdm(loader, desc=f'Epoch {epoch}', leave=False):
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
    auc = roc_auc_score(trues, preds)
    return running_loss / len(loader), acc, f1, prec, rec, auc


"""
Evaluation loop on validation set.
"""
def eval_loop(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds, trues = [], []
    
    with torch.no_grad():
        for x1, x2, y in tqdm(loader, desc='Validation', leave=False):
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
    auc = roc_auc_score(trues, preds)
    return running_loss / len(loader), acc, f1, auc


"""
Plots loss and accuracy metrics on a graph.
"""
def plot_metrics(history):
    plt.figure(figsize=(18, 9))

    plt.plot(history['loss'], label='Loss')
    plt.plot(history['acc'], label='Accuracy')
    plt.plot(history['f1'], label='F1 Score')
    plt.plot(history['prec'], label='Precision')
    plt.plot(history['rec'], label='Recall')
    plt.plot(history['auc'], label='ROC AUC')

    plt.title('Training Metrics')
    plt.xlabel('Epoch #')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./recognition/s4696725_siamese/training_results.png')
    plt.close()


"""
Trains a Siamese network on the ISIC dataset using training and validation splits.
Saves the best-performing model (based on validation F1) and logs metrics.
"""
def main():
    train_images = './recognition/s4696725_siamese/train'
    train_csv = './recognition/s4696725_siamese/metadata/train.csv'
    model_dir = './recognition/s4696725_siamese/model'

    epochs = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print('Using', device, '\n')

    model = SiameseNet(embedding_size=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    train_loader, val_loader, _ = generate_dataloaders(train_images, train_csv)
    
    best_f1 = 0.0
    history = {'loss': [], 'acc': [], 'f1': [], 'prec': [], 'rec': [], 'auc': []}

    for epoch in range(1, epochs + 1):
        loss, acc, f1, prec, rec, auc = train_loop(model, device, train_loader, optimizer, criterion, epoch)
        history['loss'].append(loss)
        history['acc'].append(acc)
        history['f1'].append(f1)
        history['prec'].append(prec)
        history['rec'].append(rec)
        history['auc'].append(auc)
        print(f"Epoch {epoch}:      Loss = {loss:.4f}, Acc = {acc:.4f}, F1 = {f1:.4f}, AUC = {auc:.4f}")

        # validate every 2 epochs and final epoch
        if epoch % 2 == 0 or epoch == epochs:
            val_loss, val_acc, val_f1, val_auc = eval_loop(model, device, val_loader, criterion)
            print(f"Validation: Loss = {val_loss:.4f}, Acc = {val_acc:.4f}, F1 = {val_f1:.4f}, AUC = {val_auc.f4}")

            # save best model as .pth file
            if val_f1 >= best_f1:
                best_f1 = val_f1
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_dir, 'siamese.pth'))

    plot_metrics(history)

    print(f"Loss:       {history['loss'][-1]:.4f}")
    print(f"Accuracy:   {history['acc'][-1]:.4f}")
    print(f"F1 Score:   {history['f1'][-1]:.4f}")
    print(f"Precision:  {history['prec'][-1]:.4f}")
    print(f"Recall:     {history['rec'][-1]:.4f}")
    print(f"ROC AUC:     {history['auc'][-1]:.4f}\n")
    print('Training Complete.')


if __name__ == '__main__':
    main()