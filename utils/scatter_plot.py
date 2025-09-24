import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

def scatter_plot_aleatoric_epistemic(aleatorics, epistemics, labels):
    # Create a scatter plot
    aleatorics = np.concatenate(aleatorics, axis=0)
    epistemics = np.concatenate(epistemics, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(10, 8))
    plt.scatter(aleatorics, epistemics, c=aleatorics+epistemics, 
                cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(label='Sum of Uncertainties')
    plt.xlabel('Aleatoric Uncertainty')
    plt.ylabel('Epistemic Uncertainty')
    plt.title('Scatter Plot of Aleatoric vs Epistemic Uncertainty')
    plt.grid(True)
    plt.show()
    plt.savefig(f"plots/scatter_plot.png")