"""
Siamese model components with backbone, embedding head, and comparator/classifier head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


"""
Pre-trained ResNet backbone producing a fixed-length embedding from an input image.

embedding_size: Dimension of output embedding vector for each image in feature space.
"""
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size: int = 512):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)

        # remove final fc
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        feature = self.encoder(x) # (B, 512, 1, 1)
        feature = feature.view(feature.size(0), -1)
        emb = self.fc(feature)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


"""
Two-tower siamese that returns embeddings for both inputs and a similarity score.
For training we provide pairs (img1, img2) and label = 1 if same class, else 0.
The classification head takes |e1 - e2| and predicts same/different.

embedding_size: Dimension of output embedding vector for each image in feature space.
"""
class SiameseNet(nn.Module):
    def __init__(self, embedding_size: int = 512):
        super().__init__()
        self.embedding_net = EmbeddingNet(embedding_size=embedding_size)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x1, x2):
        e1 = self.embedding_net(x1)
        e2 = self.embedding_net(x2)
        diff = torch.abs(e1 - e2)
        logit = self.classifier(diff).squeeze(1)
        return e1, e2, logit