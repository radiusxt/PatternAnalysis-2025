"""
Loads trained model and runs inference on test images.
"""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import SiameseNet
from dataset import generate_dataloaders
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


"""
Compute mean embedding per class using a subset of training images.
Returns dict {0: tensor, 1: tensor}
"""
def compute_class_prototypes(model, device, loader, num_per_class):
    embeddings, proto = {0: [], 1: []}, {}

    with torch.no_grad():
        for img1, img2, labels in tqdm(loader, leave=False):
            img1, img2 = img1.to(device), img2.to(device)
            e1, _, _ = model(img1, img2)

            for i, label in enumerate(labels):
                embeddings[int(label.item())].append(e1[i])

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
def evaluate(model, device, test_loader, prototypes):
    proto0 = prototypes[0].unsqueeze(0).to(device)
    proto1 = prototypes[1].unsqueeze(0).to(device)

    preds, targets = [], []

    with torch.no_grad():
        for img1, _, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            img1 = img1.to(device)
            emb = model.embedding_net(img1)

            d0 = torch.norm(emb - proto0, dim=1)
            d1 = torch.norm(emb - proto1, dim=1)

            probs = torch.softmax(torch.stack([-d0, -d1], dim=1), dim=1)[:, 1]
            preds.extend((probs >= 0.5).long().cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    return acc, targets, preds


"""
Runs a sample test set on the siamese neural network.
"""
def main():
    model_path = './recognition/s4696725_siamese/model/siamese.pth'
    train_images = './recognition/s4696725_siamese/train'
    train_csv = './recognition/s4696725_siamese/metadata/train.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print('Using', device, '\n')

    # Load dataloaders
    train_loader, _, test_loader = generate_dataloaders(train_images, train_csv)

    # Load model
    model = SiameseNet(embedding_size=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Compute prototypes using training set then run inference
    prototypes = compute_class_prototypes(model, device, train_loader, 3000)
    test_acc, y_true, y_pred = evaluate(model, device, test_loader, prototypes)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Benign (0)", "Malignant (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('./recognition/s4696725_siamese/confusion_matrix.png')
    plt.close()


if __name__ == '__main__':
    main()