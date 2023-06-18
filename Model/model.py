import torch
import torch.nn as nn
import torch.optim as optim

class RecommendationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create nn.Layers capable of handling the shapes of the data
        self.layer1 = nn.Linear(in_features=12, out_features=64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))  # x -> layer1 -> relu -> layer2 -> output

