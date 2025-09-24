import argparse
import collections
import torch
import numpy as np
import random
import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
import models.loss as module_loss
import models.metric as module_metric
import models.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import optimizers.sgld

def main(config):
    # fix random seeds for reproducibility
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    logger = config.get_logger('train')

    # setup datasets instances
    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]

    dataset = {}
    dataloader = {}
    for mode in ['train', 'val']:
        dataset[mode] = config.init_obj('dataset', module_data,
                                        dynamic_features=dynamic_features, static_features=static_features,
                                        train_val_test=mode)
        dataloader[mode] = config.init_obj('dataloader', module_dataloader, dataset=dataset[mode]).dataloader()

    device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])
    # build models architecture, then print to console
    model = config.init_obj('arch', module_arch, len_features=len(dynamic_features) + len(static_features),
                                noisy=config['noisy'])

    logger.info(model)

    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    train_metrics = [getattr(module_metric, met) for met in config['train_metrics']]
    valid_metrics = [getattr(module_metric, met) for met in config['valid_metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if config['optimizer']['type'] != 'SGLD':
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    else:
        optimizer = config.init_obj('optimizer', optimizers.sgld, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, train_metrics, valid_metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=dataloader['train'],
                      valid_data_loader=dataloader['val'],
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;args;batch_size'),
        CustomArgs(['--tau', '--tau'], type=float, target='temperature_scale'),
        CustomArgs(['--name', '--name'], type=str, target='name'),
        CustomArgs(['--seed', '--seed'], type=int, target='seed'),
        CustomArgs(['--temporal_gap', '--temporal_gap'], type=int, target='dataset;args;temporal_gap'),

    ]
    config = ConfigParser.from_args(args, options)
    main(config)
