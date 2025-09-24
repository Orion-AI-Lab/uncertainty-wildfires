import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt


def calc_bins(preds, labels_oneh):
    # Assign each prediction to a bin
    num_bins = 19
    bins = np.linspace(0.1, 1, num_bins)

    inds = preds.argmax(axis=1)
    preds = preds.max(axis=1)

    binned = np.digitize(preds, bins)
    
    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (inds[binned == bin] == labels_oneh[binned == bin]).sum() / bin_sizes[bin]
            # bin_accs[bin] = (labels_oneh[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned == bin]).sum() / bin_sizes[bin]
    # print(bin_sizes)

    return bins, binned, bin_accs, bin_confs, bin_sizes


def ece(preds, labels):

    labels = np.array(labels)
    preds = np.array(preds)

    ECE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif

    return ECE


def draw_reliability_graph(preds, labels_oneh, name):

    ECE = ece(preds, labels_oneh)

    bins, _, bin_accs, _, bin_sizes = calc_bins(preds, labels_oneh)

    path_accs = f'eces/{name}.npy'
    path_sizes = f'eces/{name}_sizes.npy'

    np.save(path_accs, bin_accs)
    np.save(path_sizes, bin_sizes)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    # MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch])

    plt.show()

    plt.savefig(f"plots/calibration/calibration_{name}.png")

    return