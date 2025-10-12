"""
Siamese model components with backbone, embedding head, and comparator/classifier head.
"""

import torch
import torch.nn as nn
import torchvision.models as models


"""
Backbone producing a fixed-length embedding from an input image.
"""
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size: int = 512, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)

        # remove final fc
        modules = list(resnet.children())[:-1] # remove fc
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        feature = self.encoder(x) # (B, 512, 1, 1)
        feature = feature.view(feature.size(0), -1)
        emb = self.fc(feature)
        return emb
    
    
"""
Two-tower siamese that returns embeddings for both inputs and a similarity score.
For training we provide pairs (img1, img2) and label = 1 if same class, else 0.
The classification head takes |e1 - e2| and predicts same/different.
"""
class SiameseNet(nn.Module):
    def __init__(self, embedding_size: int = 512, pretrained: bool = True):
        super().__init__()
        self.embedding_net = EmbeddingNet(embedding_size=embedding_size, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_size // 2, 1),
        )

    def forward(self, x1, x2):
        e1 = self.embedding_net(x1)
        e2 = self.embedding_net(x2)
        diff = torch.abs(e1 - e2)
        logit = self.classifier(diff).squeeze(1)
        return e1, e2, logit