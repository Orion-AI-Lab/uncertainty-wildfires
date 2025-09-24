import numpy as np
import torch
import torch.nn as nn
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.train_functions import enable_dropout, uncertainties, uncertainties_noisy

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, train_metric_ftns, valid_metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, train_metric_ftns, valid_metric_ftns, optimizer, config)
        self.e = 0.000001
        self.config = config
        self.positive_weight = self.config["positive_weight"]
        self.forward_passes = self.config["forward_passes"]
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))*2

        self.gamma=-0.55
        self.b = self.config["trainer"]["epochs"] / ((0.1)**(1/self.gamma) - 1)
        self.a = self.config['optimizer']['args']['lr'] / (self.b ** self.gamma)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.train_metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.valid_metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (dynamic, static, bas_size, labels) in enumerate(self.data_loader):
            static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
            labels = labels.to(self.device, dtype=torch.long)
            input_ = torch.cat([dynamic, static], dim=2)
            input_ = input_.to(self.device, dtype=torch.float32)
            bas_size = bas_size.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()

            if self.config['noisy']:
                m = nn.Softmax(dim=2)
                mean, sigma = self.model(input_)
                epsilon = torch.randn((1000,) + sigma.size()).to(self.device, dtype=torch.float32)
                f = ((mean.unsqueeze(0) + epsilon * torch.abs(sigma).unsqueeze(0)) / self.config['temperature_scale'])
                probs = m(f)
                outputs = probs.mean(0)
            else:
                m = nn.Softmax(dim=1)
                outputs = self.model(input_)
                outputs = m(outputs)

            if 'bbb' not in self.config['loss']:
                loss = self.criterion(torch.log(outputs + self.e), labels, self.positive_weight,
                                      self.device, train=True)
            else:
                loss = self.model.sample_elbo(outputs=torch.log(outputs + self.e), labels=labels,
                                              criterion=self.criterion, sample_nbr=1,
                                              complexity_cost_weight=1/len(self.data_loader.dataset))[0]
            loss = torch.mean(loss*bas_size)
            loss.backward()
            self.optimizer.step()

            output = torch.argmax(outputs, dim=1)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item()*dynamic.size(0), dynamic.size(0))

            for met in self.train_metric_ftns:
                if met.__name__ not in ['auc', 'aucpr']:
                    self.train_metrics.update(met.__name__, met(output, labels)[0], met(output, labels)[1])
                else:
                    if met.__name__ == 'auc':
                        self.train_metrics.auc_update(met.__name__, met(outputs[:, 1], labels)[0],
                                                      met(outputs[:, 1], labels)[1])
                    elif met.__name__ == 'aucpr':
                        self.train_metrics.aucpr_update(met.__name__, met(outputs[:, 1], labels)[0],
                                                        met(outputs[:, 1], labels)[1])

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            if batch_idx == self.len_epoch:
                break

        # self.optimizer.param_groups[0]['lr'] = self.a*((self.b + epoch)**(self.gamma))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        if self.config["dropout"]:
            enable_dropout(self.model)

        with torch.no_grad():
            for batch_idx, (dynamic, static, bas_size, labels) in enumerate(self.valid_data_loader):
                static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
                labels = labels.to(self.device, dtype=torch.long)
                input_ = torch.cat([dynamic, static], dim=2)
                input_ = input_.to(self.device, dtype=torch.float32)
                bas_size = bas_size.to(self.device, dtype=torch.float32)
            
                outputs_list = []

                if self.config['noisy']:
                    m = nn.Softmax(dim=2)
                    sigmas_list = []
                    means_list = []
                    for _ in range(self.forward_passes):
                        mean, sigma = self.model(input_)
                        epsilon = torch.randn((1000,) + sigma.size()).to(self.device, dtype=torch.float32)
                        f = ((mean.unsqueeze(0) + epsilon * torch.abs(sigma).unsqueeze(0)) / self.config['temperature_scale'])
                        probs = m(f)
                        output = probs.mean(0)
                        outputs_list.append(output)
                        means_list.append(mean)
                        sigmas_list.append(probs.std(0))
                    outputs, mean, epistemic, aleatoric, mi, entropy = uncertainties_noisy(outputs_list, means_list,
                                                                                       sigmas_list, self.e)
                else:
                    m = nn.Softmax(dim=1)
                    for _ in range(self.forward_passes):
                        output = self.model(input_)
                        outputs_list.append(output)
                    outputs, mean, epistemic, aleatoric, mi, entropy = uncertainties(outputs_list, self.e)
                if 'bbb' not in self.config['loss']:
                    loss = self.criterion(torch.log(mean + self.e), labels, self.positive_weight,
                                          self.device, train=True)
                else:
                    loss = self.model.sample_elbo(outputs=torch.log(mean + self.e), labels=labels,
                                                  criterion=self.criterion, sample_nbr=1,
                                                  complexity_cost_weight=1 / len(self.data_loader.dataset))[0]
                
                loss = torch.mean(loss*bas_size)
                output = torch.argmax(mean, dim=1)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item()*dynamic.size(0), dynamic.size(0))
                for met in self.valid_metric_ftns:
                    if met in self.train_metric_ftns:
                        if met.__name__ not in ['auc', 'aucpr']:
                            self.valid_metrics.update(met.__name__, met(output, labels)[0], met(output, labels)[1])
                        else:
                            if met.__name__ == 'auc':
                                self.valid_metrics.auc_update(met.__name__, met(mean[:, 1], labels)[0],
                                                              met(mean[:, 1], labels)[1])
                            elif met.__name__ == 'aucpr':
                                self.valid_metrics.aucpr_update(met.__name__, met(mean[:, 1], labels)[0],
                                                                met(mean[:, 1], labels)[1])
                    else:
                        if met.__name__ == 'mean_epistemics':
                            self.valid_metrics.update(met.__name__, met(epistemic[:, 1])[0], met(epistemic[:, 1])[1])
                        elif met.__name__ == 'mean_mis':
                            self.valid_metrics.update(met.__name__, met(mi)[0], met(mi)[1])
                        elif met.__name__ == 'mean_aleatorics':
                            self.valid_metrics.update(met.__name__, met(aleatoric[:, 1])[0], met(aleatoric[:, 1])[1])
                        elif met.__name__ == 'mean_entropies':
                            self.valid_metrics.update(met.__name__, met(entropy)[0], met(entropy)[1])
                        elif met.__name__ == 'ece':
                            self.valid_metrics.ece_update(met.__name__, met(mean, labels)[0], met(mean, labels)[1])

        # add histogram of models parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
