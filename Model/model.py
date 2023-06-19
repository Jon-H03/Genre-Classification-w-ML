import torch
import torch.nn as nn
import torch.optim as optim


class RecommendationModel(nn.Module):
    def __init__(self, initial_features):
        self.initial_features = initial_features
        super().__init__()
        # Create nn.Layers capable of handling the shapes of the data
        self.layer1 = nn.Linear(in_features=self.initial_features, out_features=64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=64, out_features=16)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

