import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(200, 3), stride=(50, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 10, (20, 2), (4, 1), 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Flatten layer output for LSTM
        self.flatten_dim = 10 * 10 * 4  # The output shape after conv2
        self.lstm_input_dim = 10 * 4    # Flattened dimension of CNN features
        self.sequence_length = 10       # Define LSTM input
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=128, num_layers=2, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(128, 6)    

    def forward(self, x):
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Prepare input for LSTM
        batch_size = x.size(0)
        x = x.view(batch_size, self.sequence_length, -1)  # Reshape for LSTM
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the last output of the LSTM (sequence output for classification)
        x = lstm_out[:, -1, :]
        
        # Fully connected layer
        feature = x
        output = self.fc(x)
        
        return feature, output
    
    #Input to CNN: (batch_size, 1, 10000, 12)
    #1st layer (batch_size, 5, 99, 7)
    #2nd (batch_size, 10, 10, 4)
    #reshapeforlstm  (batch_size, 10, 40)
    #output do 6 klas mapuje