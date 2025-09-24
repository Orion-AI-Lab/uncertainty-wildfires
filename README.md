# Uncertainty-Aware Deep Learning for Wildfire Danger Forecasting

This repository contains the code and pretrained models used in the paper **"Uncertainty-Aware Deep Learning for Wildfire Danger Forecasting"**. The project implements both deterministic and uncertainty-aware deep learning models to predict wildfire danger while quantifying predictive uncertainties.  

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

Ensure you have the necessary packages for PyTorch, geospatial processing, and other dependencies specified in requirements.txt.

## Usage

The repository includes configuration files for all models in the `configs` and `configs_test` directories.

### Training

To train a model, specify the corresponding configuration file. For example, to train a Bayesian Neural Network (BNN):

```bash
python train.py --configs/config_bnn.json

You can modify the configuration files to change hyperparameters, model type, or enable aleatoric uncertainty.

### Testing

To evaluate a trained model, use the configuration in `configs_test`.

To test a Bayesian Neural Network (BNN), for example:

```bash
python test.py --configs_test/config_bnn.json

Before running any scripts, update the following paths in the configuration files:
- `dataset_root`: Path to your dataset.
- `save_dir`: Path where trained models will be saved. This should be the same path used in `configs_test` to load models.

### Aleatoric Uncertainty

The `noisy` variable in the configuration files determines whether the model accounts for aleatoric uncertainty during training.  
- Set `noisy: true` to include aleatoric noise in predictions.  
- Set `noisy: false` to ignore aleatoric uncertainty.

---

### Deep Ensembles

Deep Ensembles combine multiple trained models to improve predictive performance and quantify uncertainty. To run a Deep Ensemble:

1. Train the deterministic model multiple times using `config_det.json`.  
2.  Use all trained models together for ensemble predictions. For this, you have to specify the number of models you have trained in `configs_test/config_des.json` via the `num_models` variable.  


## Citation