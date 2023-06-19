import torch
import torch.nn as nn
import torch.optim as optim


class RecommendationModel(nn.Module):
    def __init__(self, initial_features):
        super().__init__()
        self.initial_features = initial_features
        self.layer1 = nn.Linear(in_features=self.initial_features, out_features=128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=128, out_features=64)
        self.layer3 = nn.Linear(in_features=64, out_features=32)
        self.layer4 = nn.Linear(in_features=32, out_features=16)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

