import torch
import torch.nn as nn

from blitz.modules import BayesianLSTM, BayesianLinear
from blitz.utils import variational_estimator

class SimpleLSTM(nn.Module):
    def __init__(self, output_lstm=128, dropout=0.5, len_features = 25, noisy=False):
        super().__init__()
        self.noisy = noisy
        self.lstm = nn.LSTM(len_features, output_lstm, num_layers=1, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(len_features)
        
        self.fc1 = torch.nn.Linear(output_lstm, output_lstm)
        self.drop1 = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(output_lstm, output_lstm//2)
        self.drop2 = torch.nn.Dropout(dropout)
        if self.noisy:
            self.fc3 = torch.nn.Linear(output_lstm//2, 4)
        else:
            self.fc3 = torch.nn.Linear(output_lstm//2, 2)
        
        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.drop1,
            self.relu,
            self.fc2,
            self.drop2,
            self.relu,
            self.fc3
        )
        
    def forward(self, x):
        x = self.ln1(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        if self.noisy:
            mean, sigma = x.split(split_size=2, dim=-1)
            return mean, sigma
        else:
            return x


@variational_estimator
class BBBLSTM(nn.Module):
    def __init__(self, output_lstm=128, dropout=0.5, len_features=25, noisy=False):
        super().__init__()
        self.noisy = noisy
        self.ln1 = torch.nn.LayerNorm(len_features)
        self.lstm = BayesianLSTM(in_features=len_features, out_features=output_lstm, sharpen=False)
        self.fc1 = BayesianLinear(in_features=output_lstm, out_features=output_lstm)
        self.drop1 = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.fc2 = BayesianLinear(in_features=output_lstm, out_features=output_lstm//2)
        self.drop2 = torch.nn.Dropout(dropout)
        if self.noisy:
            self.fc3 = BayesianLinear(in_features=output_lstm//2, out_features=4)
        else:
            self.fc3 = BayesianLinear(in_features=output_lstm//2, out_features=2)
        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            # self.drop1,
            self.relu,
            self.fc2,
            # self.drop2,
            self.relu,
            self.fc3
        )

    def forward(self, x, loss=None):
        x = self.ln1(x)
        x_, _ = self.lstm(x)

        # gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.fc_nn(x_)
        if self.noisy:
            mean, sigma = x_.split(split_size=2, dim=-1)
            return mean, sigma
        else:
            return x_
