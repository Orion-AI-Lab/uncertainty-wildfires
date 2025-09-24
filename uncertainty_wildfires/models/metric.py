import torch
import torch.nn as nn


def accuracy(pred, target):
    with torch.no_grad():
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target)

def mean_epistemics(epistemic):
    epistemics = 0
    for i in range(epistemic.shape[0]):
        epistemics += epistemic[i].item()
    return epistemics, epistemic.shape[0]

def mean_mis(mi):
    mis = 0
    for i in range(mi.shape[0]):
        mis += mi[i].item()
    return mis, mi.shape[0]

def mean_aleatorics(aleatoric):
    aleatorics = 0
    for i in range(aleatoric.shape[0]):
        aleatorics += aleatoric[i].item()
    return aleatorics, aleatoric.shape[0]


def mean_entropies(entropy):
    entropies = 0
    for i in range(entropy.shape[0]):
        entropies += entropy[i].item()
    return entropies, entropy.shape[0]


def precision(output, labels):
    true_positives_fire = 0
    false_positives_fire = 0
    for j in range(output.size()[0]):
        if output[j] == 1 and labels[j] == 1:
            true_positives_fire += 1
        if output[j] == 1 and labels[j] == 0:
            false_positives_fire += 1
    if false_positives_fire + true_positives_fire == 0:
        true_positives_fire += 1
    return true_positives_fire, false_positives_fire + true_positives_fire


def recall(output, labels):
    true_positives_fire = 0
    false_negatives_fire = 0
    for j in range(output.size()[0]):
        if output[j] == 1 and labels[j] == 1:
            true_positives_fire += 1
        if output[j] == 0 and labels[j] == 1:
            false_negatives_fire += 1
    if false_negatives_fire + true_positives_fire == 0:
        true_positives_fire += 1
    return true_positives_fire, false_negatives_fire + true_positives_fire


def f1_score(output, labels):
    true_positives_fire = 0
    false_negatives_fire = 0
    false_positives_fire = 0
    for j in range(output.size()[0]):
        if output[j] == 1 and labels[j] == 1:
            true_positives_fire += 1
        if output[j] == 0 and labels[j] == 1:
            false_negatives_fire += 1
        if output[j] == 1 and labels[j] == 0:
            false_positives_fire += 1
    return true_positives_fire, true_positives_fire + (1/2)*(false_positives_fire + false_negatives_fire)


def auc(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return preds, labels


def aucpr(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return preds, labels


def ece(preds, labels):
    # labels_oneh = nn.functional.one_hot(labels, num_classes=2)
    labels_oneh = labels.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    return preds, labels_oneh

def spread_skill(skill, uncertainties):
    return skill.detach().cpu().numpy(), uncertainties.detach().cpu().numpy()
