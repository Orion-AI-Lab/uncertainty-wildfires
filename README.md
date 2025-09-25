# Uncertainty-Aware Deep Learning for Wildfire Danger Forecasting

This repository contains the code and pretrained models used in the paper **"Uncertainty-Aware Deep Learning for Wildfire Danger Forecasting"**. 

The project implements both deterministic and uncertainty-aware deep learning models to predict wildfire danger.  

## Table of Contents
- [Installation](#installation)  
- [Usage](#usage)  
- [Configuration](#configuration)  
- [Pretrained Models](#pretrained-models)  
- [Citation](#citation)  

---

## Installation

Clone this repository and install the required dependencies:  

```bash
git clone <repository_url>
cd uncertainty-wildfires
pip install -r requirements.txt
```

---

## Dataset

You can download the dataset from: [Zenodo](https://zenodo.org/records/17201754)

## Usage

The repository includes configuration files for all models in the `configs` and `configs_test` directories.

### Training

To train a model, specify the corresponding configuration file. For example, to train a Variational Inference based Bayesian Neural Network:

```bash
python train.py --config configs/config_bnn.json
```

You can modify the configuration files to change hyperparameters, model type, or enable aleatoric uncertainty (via the `noisy` parameter, check Aleatoric Uncertainty section).

### Testing

To evaluate a trained model, use the configuration in `configs_test`.

For example, to test a Variational Inference based Bayesian Neural Network:

```bash
python test.py --config configs_test/config_bnn.json
```

Before running any scripts, update the following paths in the configuration files:
- `dataset_root`: Path to your stored downloaded dataset.
- `save_dir`: Path where trained models will be saved.

### Aleatoric Uncertainty

The `noisy` variable in the configuration files determines whether the model accounts for aleatoric uncertainty during training.  
- Set `noisy: true` to include aleatoric noise in predictions.  
- Set `noisy: false` to ignore aleatoric uncertainty.

### Deep Ensembles

Deep Ensembles combine multiple trained models to improve predictive performance and quantify uncertainty. To run a Deep Ensemble:

1. Train the deterministic model multiple times using `config_det.json`.  
2. Use all trained models together for ensemble predictions. For this, you have to specify the number of models you have trained in `configs_test/config_des.json` via the `num_models` variable.  

---

## Pretrained Models

All pretrained checkpoints for the models that have been used in the paper are available in the `trained_models` directory.

---

## Citation

If you use our work, please cite: