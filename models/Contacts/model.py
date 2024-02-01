import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


import warnings
warnings.filterwarnings('ignore')

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()

        self.l1 = nn.Linear(400,20)
        self.l2 = nn.Linear(20,20)
        self.l3 = nn.Linear(20,1)

    def forward(self,x):
        x = self.l1(x)
        x = F.dropout(x,training=self.training)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.dropout(x,training=self.training)
        x = F.leaky_relu(x)
        x = self.l3(x)
        return x.squeeze()

class dataset(Dataset):
    def __init__(self,x,y):
        self.x = np.array(x).astype('float')
        self.y = np.array(y).astype('float')
    def __len__(self):
        return len(self.x)
    def __getitem__(self,i):
        return torch.tensor(self.x[i],dtype=torch.float32),torch.tensor(self.y[i],dtype=torch.float32)
