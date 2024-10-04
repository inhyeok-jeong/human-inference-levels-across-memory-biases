import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import numpy as np

class BehaviorDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class NN(nn.Module):
    def __init__(self, layer_num = 5, input_size = 10):
        super(NN, self).__init__()
        self.network = nn.ModuleList()
        prev_node_num = input_size
        for i in range(layer_num):
            # node_num = random.randint()
            node_num = 16
            self.network.append(nn.Linear(prev_node_num, node_num))
            prev_node_num = node_num
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(prev_node_num, 1))
    
    def forward(self, x):
        x = torch.from_numpy(np.array(x, dtype=np.float64)).float()
        for layer in self.network:
            x = layer(x)
        return x

class ratio_NN(nn.Module):
    def __init__(self, layer_num = 5, input_size = 11):
        super(ratio_NN, self).__init__()
        self.network = nn.ModuleList()
        prev_node_num = input_size
        for i in range(layer_num):
            # node_num = random.randint()
            node_num = 16
            self.network.append(nn.Linear(prev_node_num, node_num))
            prev_node_num = node_num
            self.network.append(nn.ReLU())

        self.network.append(nn.Linear(prev_node_num, 1))
    
    def forward(self, x):
        if not x is torch.Tensor:
            x = torch.from_numpy(np.array(x, dtype=np.float64)).float()
        for layer in self.network:
            x = layer(x)
        return x

class meta_LSTM(nn.Module):
    def __init__(self, input_size = 9, hidden_size = 9, seq_len = 4, output_size = 9, num_layers=1):
        super(meta_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.input_size, self.hidden_size),
            torch.zeros(self.num_layers, self.input_size, self.hidden_size)
        )

    def forward(self, x):
        #print("LSTM forward x: ")
        #print(x)
        x = x.to(torch.float32)
        #print(x.shape)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class meta_NN(nn.Module):
    def __init__(self, layer_num = 5, input_size = 10):
        super(meta_NN, self).__init__()
        self.network = nn.ModuleList()
        prev_node_num = input_size
        for i in range(layer_num):
            # node_num = random.randint()
            node_num = 16
            self.network.append(nn.Linear(prev_node_num, node_num))
            prev_node_num = node_num
            self.network.append(nn.ReLU())

        self.network.append(nn.Linear(prev_node_num, 1))
    
    def forward(self, x):
        #print(x)
        x = torch.from_numpy(np.array(x, dtype=np.float64)).float()
        #print(x)
        for layer in self.network:
            x = layer(x)
        return x


class meta_ratio_LSTM(nn.Module):
    def __init__(self, input_size = 9, hidden_size = 9, seq_len = 4, output_size = 9, num_layers=1):
        super(meta_ratio_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.input_size, self.hidden_size),
            torch.zeros(self.num_layers, self.input_size, self.hidden_size)
        )

    def forward(self, x):
        #print("LSTM forward x: ")
        #print(x)
        x = x.to(torch.float32)
        #print(x.shape)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class meta_ratio_NN(nn.Module):
    def __init__(self, layer_num = 5, input_size = 10):
        super(meta_ratio_NN, self).__init__()
        self.network = nn.ModuleList()
        prev_node_num = input_size
        for i in range(layer_num):
            # node_num = random.randint()
            node_num = 16
            self.network.append(nn.Linear(prev_node_num, node_num))
            prev_node_num = node_num
            self.network.append(nn.ReLU())

        self.network.append(nn.Linear(prev_node_num, 1))
    
    def forward(self, x):
        #print(x)
        x = torch.from_numpy(np.array(x, dtype=np.float64)).float()
        #print(x)
        for layer in self.network:
            x = layer(x)
        return x