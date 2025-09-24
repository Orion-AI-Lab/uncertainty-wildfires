import argparse
import torch
import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
import models.loss as module_loss
import models.metric as module_metric
import models.model as module_arch
from parse_config import ConfigParser
from utils import MetricTracker
from logger import TensorboardWriter
from pathlib import Path
from glob import glob
from utils.train_functions import enable_dropout, uncertainties, uncertainties_noisy
from utils.calibration_metrics import draw_reliability_graph
from utils.discard_test import draw_discard_test
from utils.densities import draw_uncertainties_densities
from utils.scatter_plot import scatter_plot_aleatoric_epistemic
import numpy as np
import torch.nn as nn
import collections
import os 


def main(config):
    logger = config.get_logger('test')

    # setup datasets instances
    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]

    dataset = config.init_obj('dataset', module_data,
                                      dynamic_features=dynamic_features, static_features=static_features,
                                      train_val_test='test')
    dataloader = config.init_obj('dataloader', module_dataloader, dataset=dataset).dataloader()

    # device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])
    device = 'cpu'

    # logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    train_metric_ftns = [getattr(module_metric, met) for met in config['train_metrics']]
    test_metric_ftns = [getattr(module_metric, met) for met in config['valid_metrics']]

    ts = config['temperature_scale']
    ts = "%.2f" % ts
    
    models = []
    if 'noisy' in config["name"]:
        folders = [os.path.join(config['model_path'], f) for f in os.listdir(config['model_path']) if config['name'] in f]
    else:
        folders = [os.path.join(config['model_path'], f) for f in os.listdir(config['model_path']) if config['name'] in f and 'noisy' not in f]
    print(folders)
    # f = folders[0]
    for i in range(int(config["num_models"])):
        # # build models architecture
        model = config.init_obj('arch', module_arch, len_features=len(dynamic_features) + len(static_features),
                                noisy=config['noisy'])
        f = folders[i]
        path = Path(glob(f + '/*/checkpoint-epoch30.pth')[0])
        # path = Path(glob(config['model_path'].format(config["name"] + last))[0])
        # path = Path(glob(config['model_path'])[0])
        logger.info('Loading checkpoint: {} ...'.format(path))
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model.eval()
        if config["dropout"]:
            enable_dropout(model)
        models.append(model)

    # prepare models for testing
    # model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    e = 0.000001
    cfg_trainer = config['trainer']
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    writer = TensorboardWriter(config.log_dir, logger, cfg_trainer['tensorboard'])
    test_metrics = MetricTracker('loss', *[m.__name__ for m in test_metric_ftns], writer=writer)
    test_metrics.reset()

    epistemics = []
    aleatorics = []
    losses = []
    targets = []
    predictions = []
    entropies = []
    preds = []
    means = []
    xs = []
    ys = []
    bas = []

    with torch.no_grad():
        for batch_idx, (dynamic, static, bas_size, labels, x, y) in enumerate(dataloader):
            static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
            labels = labels.to(device, dtype=torch.long)
            input_ = torch.cat([dynamic, static], dim=2)
            input_ = input_.to(device, dtype=torch.float32)
            bas_size = bas_size.to(device, dtype=torch.float32)

            outputs_list = []
            if config["noisy"]:
                m = nn.Softmax(dim=2)
                sigmas_list = []
                means_list = []
                for model in models:
                    for _ in range(config['forward_passes']):
                        mean, sigma = model(input_)
                        epsilon = torch.randn((1000,) + sigma.size()).to(device, dtype=torch.float32)
                        f = ((mean.unsqueeze(0) + epsilon * torch.abs(sigma).unsqueeze(0)) / config['temperature_scale'])
                        probs = m(f)
                        output = probs.mean(0)
                        outputs_list.append(output)
                        means_list.append(mean)
                        sigmas_list.append(probs.std(0))
                outputs, mean, epistemic, aleatoric, mi, entropy = uncertainties_noisy(outputs_list, means_list,
                                                                                   sigmas_list, e)
            else:
                for model in models:
                    for _ in range(config['forward_passes']):
                        output = model(input_)
                        outputs_list.append(output)
                outputs, mean, epistemic, aleatoric, mi, entropy = uncertainties(outputs_list, e)

            if 'bbb' not in config['loss']:
                loss = criterion(torch.log(mean + e), labels, config['positive_weight'], device, train=False)
            else:
                loss = model.sample_elbo(outputs=torch.log(mean + e), labels=labels,
                                              criterion=criterion, sample_nbr=1,
                                              complexity_cost_weight=1 / len(dataloader.dataset))[0]
            loss = loss * bas_size
            output = torch.argmax(mean, dim=1)
            if 'bbb' not in config['loss']:
                loss = loss*2

            writer.set_step(batch_idx)
            test_metrics.update('loss', loss.mean(0).item() * dynamic.size(0), dynamic.size(0))

            for met in test_metric_ftns:
                if met in train_metric_ftns:
                    if met.__name__ not in ['auc', 'aucpr']:
                        test_metrics.update(met.__name__, met(output, labels)[0], met(output, labels)[1])
                    else:
                        if met.__name__ == 'auc':
                            test_metrics.auc_update(met.__name__, met(mean[:, 1], labels)[0],
                                                          met(mean[:, 1], labels)[1])
                        elif met.__name__ == 'aucpr':
                            test_metrics.aucpr_update(met.__name__, met(mean[:, 1], labels)[0],
                                                            met(mean[:, 1], labels)[1])
                else:
                    if met.__name__ == 'mean_epistemics':
                        test_metrics.update(met.__name__, met(epistemic[:, 1])[0], met(epistemic[:, 1])[1])
                    elif met.__name__ == 'mean_mis':
                        test_metrics.update(met.__name__, met(mi)[0], met(mi)[1])
                    elif met.__name__ == 'mean_aleatorics':
                        test_metrics.update(met.__name__, met(aleatoric[:, 1])[0], met(aleatoric[:, 1])[1])
                    elif met.__name__ == 'mean_entropies':
                        test_metrics.update(met.__name__, met(entropy)[0], met(entropy)[1])
                    elif met.__name__ == 'ece':
                        test_metrics.ece_update(met.__name__, met(mean, labels)[0], met(mean, labels)[1])
                    elif met.__name__ == 'spread_skill':
                        test_metrics.spread_skill_update(met.__name__, met((mean[:, 1] - labels)**2, aleatoric[:, 1])[0],
                                                         met((mean[:, 1] - labels)**2, aleatoric[:, 1])[1])


            losses.append(loss.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())
            predictions.append(output.detach().cpu().numpy())
            means.append(mean.detach().cpu().numpy()[:,1])
            epistemics.append(epistemic.detach().cpu().numpy()[:,1])
            aleatorics.append(aleatoric.detach().cpu().numpy()[:,1])


    log = test_metrics.result()
    logger.info(log)

    preds, labels = np.array(test_metrics._data.total["ece"]), np.array(test_metrics._data.counts["ece"])

    draw_reliability_graph(preds, labels, config["name"])

    #Plots for epistemic uncertainties only
    draw_discard_test(losses, epistemics, targets, predictions, config["name"], unc_type = 'epistemic')
    draw_uncertainties_densities(predictions, targets, epistemics, config["name"], unc_type = 'epistemic')

    #Plots for aleatoric uncertainties only
    draw_discard_test(losses, aleatorics, targets, predictions, config["name"], unc_type = 'aleatoric')
    draw_uncertainties_densities(predictions, targets, aleatorics, config["name"], unc_type = 'aleatoric')

    #Plots for sum of aleatoric and epistemic uncertainties
    draw_discard_test(losses, [x + y for x, y in zip(epistemics, aleatorics)], targets, predictions, config["name"], unc_type = 'epistemic+aleatoric')
    draw_uncertainties_densities(predictions, targets, [x + y for x, y in zip(epistemics, aleatorics)], config["name"], unc_type = 'epistemic+aleatoric')

    scatter_plot_aleatoric_epistemic(aleatorics, epistemics, predictions)


    return

    # n_samples = len(dataloader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: test_metrics[i].item() / n_samples for i, met in enumerate(test_metric_fns)
    # })
    # logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--tau', '--tau'], type=float, target='temperature_scale'),
        CustomArgs(['--model_path', '--model_path'], type=str, target='model_path'),
        CustomArgs(['--name', '--name'], type=str, target='name'),
        CustomArgs(['--temporal_gap', '--temporal_gap'], type=int, target='dataset;args;temporal_gap'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
