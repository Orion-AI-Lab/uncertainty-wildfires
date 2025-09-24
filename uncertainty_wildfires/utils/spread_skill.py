import numpy as np
from matplotlib import pyplot as plt


def calc_bins(skill, uncertainties):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.01, max(uncertainties)-0.01, num_bins)
    binned = np.digitize(uncertainties, bins)

    # Save the accuracy, confidence and size of each bin
    bin_skills = np.zeros(num_bins)
    bin_uncs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(uncertainties[binned == bin])
        if bin_sizes[bin] > 0:
            bin_skills[bin] = np.sqrt((skill[binned == bin]).sum() / bin_sizes[bin])
            bin_uncs[bin] = (uncertainties[binned == bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_skills, bin_uncs, bin_sizes


def spread_skill_reliability(skills, uncertainties):

    skills = np.array(skills).flatten()
    uncertainties = np.array(uncertainties).flatten()

    REL = 0
    bins, _, bin_skill, bin_uncs, bin_sizes = calc_bins(skills, uncertainties)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_uncs[i] - bin_skill[i])
        REL += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif

    return REL

def draw_spread_skill(skills, uncertainties):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    bins, _, bin_skills, bin_uncs, bin_sizes = calc_bins(skills, uncertainties)

    # # x/y limits
    ax.set_xlim(-0.0, 0.57)
    ax.set_ylim(-0.0, 0.57)

    # x/y labels
    plt.xlabel('Uncertainty')
    plt.ylabel('Loss')

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    plt.bar(bins, bin_sizes/bin_sizes.sum(), width=0.018, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Error bars
    plt.plot(bin_uncs, bin_skills)
    plt.scatter(bin_uncs, bin_skills)
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')


    # # ECE and MCE legend
    # ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    # # MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    # plt.legend(handles=[ECE_patch])

    return plt.show()