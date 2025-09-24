import torch.nn as nn
import torch


def nll_loss(output, target, positive_weight, device, train=True):
    weights = [1 - positive_weight, positive_weight]
    class_weights = torch.FloatTensor(weights).to(device)
    if train:
        red = 'mean'
    else:
        red = 'none'
    criterion = nn.NLLLoss(weight=class_weights, reduction=red)
    return criterion(output, target)


def bbb_loss(outputs, targets, train=False):
    # weights = [1 - positive_weight, positive_weight]
    # class_weights = torch.FloatTensor(weights).to(device)
    if train:
        red = 'mean'
    else:
        red = 'none'
    criterion = nn.NLLLoss(reduction=red)
    return criterion(outputs, targets)
