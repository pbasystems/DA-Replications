import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,feature_size,hidden_size=32,dropout=0.3):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, 1) 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x