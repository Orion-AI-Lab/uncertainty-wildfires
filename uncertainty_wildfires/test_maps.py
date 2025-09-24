import argparse
import torch
from utils.maps import combine_dynamic_static_inputs, input_reshape, plot_maps, plot_maps_exp, save_tif
import utils.maps as module_data
import dataloaders.dataloader as module_dataloader
import models.model as module_arch
from parse_config import ConfigParser
from pathlib import Path
from glob import glob
from utils.train_functions import enable_dropout, uncertainties, uncertainties_noisy
import json
import xarray as xr
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import torch.nn as nn
import os
from utils import prepare_device

def main(config):
    logger = config.get_logger('test maps')

    # setup datasets instances
    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]

    dc_path = Path.home() / config['datacube_path']
    input_ds = xr.open_zarr(dc_path)    

    dataset_root = Path.home() / config['dataset_root']
    min_max_file = dataset_root / 'norms.json'

    with open(min_max_file) as f:
        min_max_dict = json.load(f) 


    input_date = datetime.strptime(config['input_date'], '%Y/%m/%d %H:%M:%S')
    inp_ds = input_ds.sel(time=slice(input_date - pd.Timedelta('44 days'), input_date)).isel(x=slice(684-35, 684+35), y =slice(660-35, 660+35))
    slope = xr.open_dataset('/home/skondylatos/jh-shared/iprapas/uc3/MED/dem_products.nc')['dem_slope_radians'].isel(x=slice(684-35, 684+35), y =slice(660-35, 660+35))
    inp_ds['slope'] = (['y', 'x'], slope.values)
    inp_ds = inp_ds.load()

    main_var_filled = inp_ds.ffill(dim="x", limit=20).ffill(dim="y",limit=20).bfill(dim="x",limit=20).bfill(dim="y",limit=20)
    mask = (inp_ds['slope']!=0)
    inp_ds = inp_ds.where(~mask, main_var_filled)

    print('Dataset loaded')

    dataset = config.init_obj('dataset', module_data, ds=input_ds, inp_ds=inp_ds,
                              min_max_dict=min_max_dict, dynamic_features=dynamic_features,
                              static_features=static_features)

    dataloader = config.init_obj('dataloader', module_dataloader, dataset=dataset).dataloader()

    
    device = 'cpu'
    # # build models architecture
    ts = config['temperature_scale']
    ts = "%.2f" % ts
    models = []
    folders = [os.path.join(config['model_path'], f) for f in os.listdir(config['model_path']) if config['name'] in f]
    print(folders)
    for i in range(int(config["num_models"])):
        # # build models architecture
        model = config.init_obj('arch', module_arch, len_features=len(dynamic_features) + len(static_features),
                                noisy=config['noisy'])
        f = folders[i]
        path = Path(glob(f + f'/*/checkpoint-epoch30.pth')[0])
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

    # lc = (input_ds['lc_wetland'].sel(time=slice('01-01-2022', '01-01-2022')).squeeze() + 
    # input_ds['lc_water_bodies'].sel(time=slice('01-01-2022', '01-01-2022')).squeeze() +
    # input_ds['lc_settlement'].sel(time=slice('01-01-2022', '01-01-2022'))).values

    #.isel(x=slice(110, 1200), y =slice(380, 1170))

    lc = (input_ds['lc_wetland'].sel(time=slice('01-01-2022', '01-01-2022')).isel(x=slice(684-35, 684+35), y =slice(660-35, 660+35)).squeeze() + 
    input_ds['lc_water_bodies'].sel(time=slice('01-01-2022', '01-01-2022')).isel(x=slice(684-35, 684+35), y =slice(660-35, 660+35)).squeeze() +
    input_ds['lc_settlement'].sel(time=slice('01-01-2022', '01-01-2022')).isel(x=slice(684-35, 684+35), y =slice(660-35, 660+35)).squeeze()).values
    lc=lc.astype(int)

    fin_outputs = []
    entropies = []
    epistemics = []
    aleatorics = []
    total_unc = []
    e = 0.000001

    dt = inp_ds['time'][-1].dt.strftime('%d-%m-%Y').values
    path = config['maps_path']

    for batch_idx, (dynamic, static) in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ = combine_dynamic_static_inputs(dynamic, static)
        input_ = input_.to(device)
        
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

        
        fin_outputs.append(mean[:, 1].detach().numpy())
        entropies.append(entropy.detach().numpy())
        epistemics.append(epistemic[:, 1].detach().numpy())
        aleatorics.append(aleatoric[:, 1].detach().numpy())
        total_unc.append(epistemic[:, 1].detach().numpy() + aleatoric[:, 1].detach().numpy())

    fin_outputs, entropies, epistemics, aleatorics, total_unc = input_reshape(fin_outputs, inp_ds), input_reshape(entropies, inp_ds), \
                                                 input_reshape(epistemics, inp_ds), input_reshape(aleatorics, inp_ds), input_reshape(total_unc, inp_ds)
    
    fin_outputs[lc==1] = float('Nan')
    epistemics[lc==1] = float('Nan')
    aleatorics[lc==1] = float('Nan')
    total_unc[lc==1] = float('Nan')

    # _ = plot_maps(inp_ds, fin_outputs, dt, path, config['name'], 'predictions')
    # _ = plot_maps(inp_ds, aleatorics, dt, path, config['name'], 'aleatorics')
    # _ = plot_maps(inp_ds, epistemics, dt, path, config['name'], 'epistemics')
    # _ = plot_maps(inp_ds, total_unc, dt, path, config['name'], 'total uncertainty')
    # _ = plot_maps_exp(outputs, aleatorics, epistemics, dt, path, config['name'])

    _ = save_tif(inp_ds, fin_outputs, epistemics, aleatorics, total_unc, dt, path, config['name'])

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
