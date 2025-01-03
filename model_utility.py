import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.distributions as dist 

class SpectrumModel(nn.Module):
    def __init__(self,in_channels=1, base_channels=512):
  
 
        return 0
    def forward(self):

        return 0

class Traffic_dataset(Dataset):
    def __init__(self,data):
        self.features = data['features']
        self.adjacency = data['adjacency']

    def __len__(self):
        return [len(self.adjacency), len(self.features)]

    def __getitem__(self, idx):
        return self.features[idx], self.adjacency[idx]