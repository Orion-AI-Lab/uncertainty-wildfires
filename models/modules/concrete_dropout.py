import torch
import torch.nn as nn
import numpy as np


class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer: float = 1e-6,
                 dropout_regularizer: float = 1e-5, init_min: float = 0.1, init_max: float = 0.1):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.regularization = 0.0

    def forward(self, x: torch.Tensor, layer: nn.Module):
        p = torch.sigmoid(self.p_logit)

        out = layer(self._concrete_dropout(x, p))

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        input_dimensionality = x[0].numel()  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        self.regularization = weights_regularizer + dropout_regularizer

        return out

    def _concrete_dropout(self, x: torch.Tensor, p: torch.Tensor):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x
