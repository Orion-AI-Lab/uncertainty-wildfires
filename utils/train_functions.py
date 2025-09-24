import torch
import torch.nn as nn

def enable_dropout(mod):
    """ Function to enable the dropout layers during test-time """
    for m in mod.modules():
        if 'Dropout' in m.__class__.__name__:
            m.train()


def uncertainties(outputs_list, e):
    outputs = torch.stack(outputs_list, dim=1)
    m = nn.Softmax(dim=2)
    outputs = m(outputs)
    mean = outputs.mean(1)
    epistemic = (outputs ** 2).mean(1) - (outputs.mean(1)) ** 2
    aleatoric = (outputs * (1 - outputs)).mean(1)
    entropy = -torch.sum(mean * torch.log(mean + e), dim=-1)
    mi = entropy + (torch.sum(outputs * torch.log(outputs + e), dim=[-1, -2]))/outputs.shape[1]
    # aleatoric = entropy - epistemic
    return outputs, mean, epistemic, aleatoric, mi, entropy

def uncertainties_noisy(outputs_list, means_list, variances_list, e):
    outputs = torch.stack(outputs_list, dim=1)
    means = torch.stack(means_list, dim=1)
    variances = torch.stack(variances_list, dim=1)**2
    mean = outputs.mean(1)
    epistemic = (outputs ** 2).mean(1) - (outputs.mean(1)) ** 2
    aleatoric = variances.mean(1)
    entropy = -torch.sum(mean * torch.log(mean + e), dim=-1)
    mi = entropy + (torch.sum(outputs * torch.log(outputs + e), dim=[-1, -2]))/outputs.shape[1]
    return outputs, mean, epistemic, aleatoric, mi, entropy