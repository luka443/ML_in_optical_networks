import torch
import torch.nn as nn
import argparse
import numpy as np
import time


# neural_network_models/rnn.py

import torch
import torch.nn as nn

# neural_network_models/rnn_model.py

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=120, hidden_size=128, num_layers=2, num_classes=6):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through RNN layer
        out, _ = self.rnn(x, h0)
        
        # Decode hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out
