import numpy as np
from matplotlib import pyplot as plt


def discard_test(losses, uncertainties, targets, predictions):
    discard_fractions = np.linspace(0.0, 1, 10)
    num_fractions = len(discard_fractions)

    discard_fractions = np.sort(discard_fractions)
    discard_error_values = np.full(num_fractions, np.nan)
    discard_example_fractions = np.full(num_fractions, np.nan)
    discard_targets = np.full(num_fractions, np.nan)
    discard_predictions = np.full(num_fractions, np.nan)

    use_example_flags = np.full(uncertainties.shape, 1, dtype=bool)

    for k in range(num_fractions):
        percentile = 100 * (1 - discard_fractions[k])
        this_inverted_mask = (
                uncertainties >
                np.percentile(uncertainties, percentile)
        )
        use_example_flags[this_inverted_mask] = False

        discard_example_fractions[k] = np.mean(use_example_flags)
        discard_targets[k] = np.count_nonzero(targets[use_example_flags == True] == 1)/\
                             len(targets[use_example_flags == True])
        discard_predictions[k] = np.count_nonzero(predictions[use_example_flags == True] == 1) / \
                             len(predictions[use_example_flags == True])

        discard_error_values[k] = np.mean(
            losses[use_example_flags == True]
        )

    mf = np.mean(np.diff(discard_error_values) < 0)
    di = -np.mean(np.diff(discard_error_values))



    return discard_fractions, discard_error_values, discard_example_fractions, discard_targets, discard_predictions, mf, di

def draw_discard_test(losses, uncertainties, targets, predictions, name, unc_type):

    losses = np.concatenate(losses, axis=0)
    uncertainties = np.concatenate(uncertainties, axis=0)
    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    if name == 'noisy-ensembles':
        n = 'DEs + Aleatoric'
    elif name == 'noisy-mcd':
        n = 'MCD + Aleatoric'
    elif name == 'noisy-vi':
        n = "BBB + Aleatoric"
    elif name == 'Noisy-Final':
        n = "Aleatoric"

    discard_fractions, discard_error_values, discard_example_fractions, discard_targets, discard_predictions,\
        monotonicity_fraction, di = \
        discard_test(losses, uncertainties, targets, predictions)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # # x/y limits
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.6)

    # x/y labels
    plt.xlabel('Discard Fraction', fontsize=24)
    plt.ylabel('Loss', fontsize=24)

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.plot(discard_fractions, discard_error_values)
    plt.scatter(discard_fractions, discard_error_values)
    # plt.plot([0, 1], [0, 0.5], '--', color='gray', linewidth=2)
    # plt.bar(discard_fractions, discard_targets, width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Equally spaced axes
    # plt.gca().set_aspect('equal', adjustable='box')

    title_string = '{0}, MF = {1:.2f}, DI = {2:.2f}'.format(n, monotonicity_fraction, di)
    ax.set_title(title_string, fontsize=24)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset_axes = inset_axes(ax,
                            width="30%",
                            height="30%",
                            bbox_to_anchor=(-0.6, -0.6, 1, 1),
                            bbox_transform=ax.transAxes)

    inset_axes.set_title('Test')
    plt.plot(discard_fractions, discard_targets, label='Percentage of positive targets')
    plt.plot(discard_fractions, discard_predictions, label='Percentage of positive predictions')
    plt.scatter(discard_fractions, discard_targets)
    plt.scatter(discard_fractions, discard_predictions)
    plt.xlabel('Discard Fraction')

    inset_axes.legend(bbox_to_anchor=(0.6, 0.9))
    plt.show()

    plt.savefig(f"plots/{unc_type}/discard_{name}.png")

    return

