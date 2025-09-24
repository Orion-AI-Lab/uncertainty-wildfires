import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def draw_uncertainties_densities(predictions, targets, uncertainties, name, unc_type):

    uncertainties = np.concatenate(uncertainties, axis=0)
    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)


    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    num_classes = 2
    classes = list(range(num_classes)) + [list(range(num_classes))]

    for l, ax in zip(classes, axs):
        correct = uncertainties[(predictions == targets) & np.isin(predictions, l)]
        incorrect = uncertainties[(predictions != targets) & np.isin(predictions, l)]
        
        n = name
        # # # x/y limits
        # ax.set_xlim(-0.0, 0.57)
        # ax.set_ylim(-0.0, 0.57)

        sns.kdeplot(data=correct, shade=True, color='green', label='Correctly classified ' + str(len(correct)), ax=ax)
        sns.kdeplot(data=incorrect, shade=True, color='red', label='Incorrectly classified ' + str(len(incorrect)), ax=ax)
        if isinstance(l, int):
            ax.set_title(f'{n} for class ' + str(l), fontsize=24)
        else:
            ax.set_title(f'{n} for all classes', fontsize=24)
        ax.set_xlabel('Uncertainty', fontsize=24)
        ax.set_ylabel('Density', fontsize=24)

        q_value = np.quantile(correct, 0.5)
        ax.axvline(x=q_value, linestyle='--', color='green', label=f'median of all')

        q_value = np.quantile(incorrect, 0.5)
        ax.axvline(x=q_value, linestyle='--', color='red', label=f'median of incorrect')

        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.show()
    plt.savefig(f"plots/densities_{unc_type}_{name}.png")

    # # Create grid
    # ax.set_axisbelow(True)
    # ax.grid(color='gray', linestyle='dashed')


    # # Equally spaced axes
    # plt.gca().set_aspect('equal', adjustable='box')

    return