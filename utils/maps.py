import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import warnings
import xarray as xr
import matplotlib
import matplotlib
import matplotlib.offsetbox
from matplotlib.lines import Line2D

from mpl_toolkits.basemap import Basemap
import matplotlib.font_manager as fm
import pandas as pd
from datetime import datetime


# def input_reshape(inputs_list, input_ds):
#     input_ = np.concatenate(inputs_list)
#     len_x, len_y = len(input_ds['x']), len(input_ds['y'])
#     y = input_.reshape(len_y, len_x)
#     clc_to_include = [2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
#     clc = np.isin(input_ds['CLC_2018'].values, clc_to_include, invert=True)
#     y[clc] = float('Nan')
#     return y


# def combine_dynamic_static_inputs(dynamic, static, clc):
#     bsize, timesteps, _ = dynamic.shape
#     static = static.unsqueeze(dim=1)
#     repeat_list = [1 for _ in range(static.dim())]
#     repeat_list[1] = timesteps
#     static = static.repeat(repeat_list)
#     input_list = [dynamic, static]
#     if clc is not None:
#         clc = clc.unsqueeze(dim=1).repeat(repeat_list)
#         input_list.append(clc)
#     inputs = torch.cat(input_list, dim=2).float()
#     return inputs


# def get_pixel_feature_ds(the_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0):
#     assert access_mode in ['spatial', 'temporal', 'spatiotemporal']
#     assert lag >= 0 and patch_size >= 0 and t >= 0 and x >= 0 and y >= 0
#     patch_half = patch_size // 2
#     assert x >= patch_half and x + patch_half < the_ds.dims['x']
#     assert y >= patch_half and y + patch_half < the_ds.dims['y']
#     #     len_x = ds.dims['x'] - patch_size
#     #     len_y = ds.dims['y'] - patch_size
#     if access_mode == 'spatiotemporal':
#         block = the_ds.isel(time=slice(t + 1 - lag, t + 1), x=slice(x - patch_half, x + patch_half + 1),
#                             y=slice(y - patch_half, y + patch_half + 1))  # .reset_index(['x', 'y', 'time'])
#     elif access_mode == 'temporal':
#         block = the_ds.isel(time=slice(0, t + 1), x=x, y=y).reset_index(['time'])
#     elif access_mode == 'spatial':
#         block = the_ds.isel(x=slice(x - patch_half, x + patch_half + 1),
#                             y=slice(y - patch_half, y + patch_half + 1))  # .reset_index(['x', 'y'])

#     return block


# def min_max_scaling(chunk, feat_name, access_mode, feature_renaming, min_max_dict, clip=True):
#     '''
#     (x - min)/(max - min)
#     '''
#     feat_name_old = feature_renaming[feat_name]
#     minimum = min_max_dict['min'][access_mode][feat_name_old]
#     maximum = min_max_dict['max'][access_mode][feat_name_old]
#     feat = chunk[feat_name]
#     if feat_name == 'ndvi':
#         feat = feat * 100000000
#     if clip:
#         feat = np.clip(feat, a_min=minimum, a_max=maximum)
#     return (feat - minimum) / (maximum - minimum)


# def get_pixel_feature_vector(the_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0, feature_renaming=None,
#                              min_max_dict=None, dynamic_features=None,
#                              static_features=None, override_whole=False, scaling='minmax',
#                              clc='vec'):
#     if override_whole:
#         chunk = the_ds
#     else:
#         chunk = get_pixel_feature_ds(the_ds, t=t, x=x, y=y, access_mode=access_mode, patch_size=patch_size, lag=lag)

#     if scaling == 'minmax':
#         dynamic = np.stack([min_max_scaling(chunk, feature, access_mode, feature_renaming, min_max_dict) for feature in dynamic_features])
#         static = np.stack([min_max_scaling(chunk, feature, access_mode, feature_renaming, min_max_dict) for feature in static_features])

#     if 'temp' in access_mode:
#         dynamic = np.moveaxis(dynamic, 0, 1)
#     clc_vec = 0
#     if clc == 'vec':
#         clc_vec = np.stack([chunk[f'CLC_2018_{i}'] for i in range(10)])

#     return dynamic, static, clc_vec


# class FireDatasetWholeDay(Dataset):
#     def __init__(self, ds, access_mode='temporal', problem_class='classification', patch_size=0, lag=10,
#                  feature_renaming=None, min_max_dict=None, dynamic_features=None,
#                  static_features=None, nan_fill=-1.0, clc=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             dynamic_transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         assert access_mode in ['temporal', 'spatial', 'spatiotemporal']
#         assert problem_class in ['classification', 'segmentation']
#         self.problem_class = problem_class
#         self.override_whole = problem_class == 'segmentation'
#         self.ds = ds
#         self.ds = self.ds.load()
#         print("Dataset loaded...")
#         pixel_range = patch_size // 2
#         self.pixel_range = pixel_range
#         self.len_x = self.ds.dims['x']
#         self.len_y = self.ds.dims['y']
#         self.patch_size = patch_size
#         self.lag = lag
#         self.access_mode = access_mode
#         self.nan_fill = nan_fill
#         self.feature_renaming = feature_renaming
#         self.min_max_dict = min_max_dict
#         self.dynamic_features = dynamic_features
#         self.static_features = static_features
#         self.clc = clc

#     def __len__(self):
#         if self.problem_class == 'segmentation':
#             return 1
#         return self.len_x * self.len_y

#     def __getitem__(self, idx):
#         y = idx // self.len_x + self.pixel_range
#         x = idx % self.len_x + self.pixel_range

#         dynamic, static, clc = get_pixel_feature_vector(self.ds, self.lag, x,
#                                                         y, self.access_mode, self.patch_size,
#                                                         self.lag, self.feature_renaming,
#                                                         self.min_max_dict,
#                                                         self.dynamic_features,
#                                                         self.static_features,
#                                                         self.override_whole, clc=self.clc)
#         if self.access_mode == 'temporal':
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore", category=RuntimeWarning)
#                 feat_mean = np.nanmean(dynamic, axis=0)
#                 # Find indices that you need to replace
#                 inds = np.where(np.isnan(dynamic))
#                 # Place column means in the indices. Align the arrays using take
#                 dynamic[inds] = np.take(feat_mean, inds[1])

#         dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
#         static = np.nan_to_num(static, nan=self.nan_fill)

#         return dynamic, static, clc


def input_reshape(inputs_list, input_ds):
    input_ = np.concatenate(inputs_list)
    len_x, len_y = len(input_ds['x']), len(input_ds['y'])
    y = input_.reshape(len_y, len_x)
    # clc_to_include = [2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    # clc = np.isin(input_ds['CLC_2018'].values, clc_to_include, invert=True)
    # y[clc] = float('Nan')
    return y


def combine_dynamic_static_inputs(dynamic, static):
    bsize, timesteps, _ = dynamic.shape
    static = static.unsqueeze(dim=1)
    repeat_list = [1 for _ in range(static.dim())]
    repeat_list[1] = timesteps
    static = static.repeat(repeat_list)
    input_list = [dynamic, static]
    inputs = torch.cat(input_list, dim=2).float()
    return inputs


def get_pixel_feature_ds(the_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0):
    block = the_ds.isel(time=slice(0, t + 1), x=x, y=y).reset_index(['time'])
    return block


def mean_scaling(chunk, feat_name, min_max_dict):
    '''
    (x - mean)/(std)
    '''
    mean = min_max_dict['mean'][feat_name]
    std = min_max_dict['std'][feat_name]
    feat = chunk[feat_name]
    return (feat - mean) / std


def get_pixel_feature_vector(the_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0, feature_renaming=None,
                             min_max_dict=None, dynamic_features=None,
                             static_features=None, override_whole=False, scaling='mean'):
            
    if override_whole:
        chunk = the_ds
    else:
        chunk = get_pixel_feature_ds(the_ds, t=t, x=x, y=y, access_mode=access_mode, patch_size=patch_size, lag=lag)

    if scaling == 'mean':
        dynamic = np.stack([mean_scaling(chunk, feature, min_max_dict) for feature in dynamic_features])
        static = np.stack([mean_scaling(chunk, feature, min_max_dict) for feature in static_features])

    if 'temp' in access_mode:
        dynamic = np.moveaxis(dynamic, 0, 1)

    return dynamic, static


class FireDatasetWholeDay(Dataset):
    def __init__(self, ds, inp_ds, access_mode='temporal', problem_class='classification', patch_size=0, lag=45,
                 nan_fill=0.0, feature_renaming=None, min_max_dict=None, dynamic_features=None,
                 static_features=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            dynamic_transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert access_mode in ['temporal', 'spatial', 'spatiotemporal']
        assert problem_class in ['classification', 'segmentation']
        self.problem_class = problem_class
        self.override_whole = problem_class == 'segmentation'
        self.ds = inp_ds
        # self.ds = self.ds.load()
        # print("Dataset loaded...")
        pixel_range = patch_size // 2
        self.pixel_range = pixel_range
        self.len_x = self.ds.dims['x']
        self.len_y = self.ds.dims['y']
        self.patch_size = patch_size
        self.lag = lag
        self.access_mode = access_mode
        self.nan_fill = nan_fill
        self.feature_renaming = feature_renaming
        self.min_max_dict = min_max_dict
        self.dynamic_features = dynamic_features
        self.static_features = static_features
        
        ds_vars = list(self.ds.keys())  
        year = str(pd.DatetimeIndex([self.ds['time'][-1].values]).year[0])

        for var in ds_vars:
            if var == 'population' or 'lc' in var:
                del self.ds[var]
                dt = str(year) + '/01/01'
                self.ds[var] = ds[var].sel(time=slice(dt, dt))[0].isel(x=slice(110, 1200), y =slice(380, 1170)).load()
                # self.ds[var] = ds[var].sel(time=slice(dt, dt))[0].load()

    def __len__(self):
        if self.problem_class == 'segmentation':
            return 1
        return self.len_x * self.len_y

    def __getitem__(self, idx):
        y = idx // self.len_x + self.pixel_range
        x = idx % self.len_x + self.pixel_range

        dynamic, static = get_pixel_feature_vector(self.ds, self.lag, x,
                                                        y, self.access_mode, self.patch_size,
                                                        self.lag, self.feature_renaming,
                                                        self.min_max_dict,
                                                        self.dynamic_features,
                                                        self.static_features,
                                                        self.override_whole)
        if self.access_mode == 'temporal':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                feat_mean = np.nanmean(dynamic, axis=0)
                # Find indices that you need to replace
                inds = np.where(np.isnan(dynamic))
                # Place column means in the indices. Align the arrays using take
                dynamic[inds] = np.take(feat_mean, inds[1])

        dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
        static = np.nan_to_num(static, nan=self.nan_fill)

        return dynamic, static
    

def plot_maps_exp(outputs, aleatorics, epistemics, time, path, name):
    fig, ax = plt.subplots(1, 2, figsize=(30, 18))
    fig.suptitle('{}'.format(time), y=0.68)
    plt.subplot(121), plt.imshow(outputs.squeeze(), vmin=0, vmax=1, cmap='Spectral_r')
    plt.title('Predicted Fire Danger')
    # plt.subplot(122), plt.imshow(aleatorics.squeeze(), vmin=0, vmax=1, cmap='Spectral_r')
    # plt.title('Aleatoric Uncertainty')
    plt.subplot(122), plt.imshow(aleatorics.squeeze(), vmin=0, cmap='Spectral_r')
    plt.title('Aleatoric Uncertainty')
    # plt.savefig(path + '/' + name + time + '.png', bbox_inches='tight', dpi=500)
    plt.show()

def plot_maps(input_ds, input, time, path, name, map_type):

    class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
        """ size: length of bar in data units
            extent : height of bar ends in axes units """

        def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                     pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None,
                     frameon=True, linekw={}, **kwargs):
            if not ax:
                ax = plt.gca()
            trans = ax.get_xaxis_transform()
            size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
            line = Line2D([0, size], [0, 0], **linekw)
            vline1 = Line2D([0, 0], [-extent / 2., extent / 2.], **linekw)
            vline2 = Line2D([size, size], [-extent / 2., extent / 2.], **linekw)
            size_bar.add_artist(line)
            size_bar.add_artist(vline1)
            size_bar.add_artist(vline2)
            txt = matplotlib.offsetbox.TextArea(label, textprops=dict(size=7))
            self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar, txt],
                                                     align="center", pad=ppad, sep=sep)
            matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                                                            borderpad=borderpad, child=self.vpac, prop=prop,
                                                            frameon=frameon,
                                                            **kwargs)

    fontprops = fm.FontProperties(size=18)
    # setup Lambert Conformal basemap.
    # set resolution=None to skip processing of boundary datasets.
    fig, ax = plt.subplots(figsize=(14, 10))

    m = Basemap(projection='cyl', llcrnrlat=input_ds['y'][-1].values, urcrnrlat=input_ds['y'][0].values,
                llcrnrlon=input_ds['x'][0].values, urcrnrlon=input_ds['x'][-1].values, resolution='h')

    m.arcgisimage(service='World_Street_Map', xpixels=1500, verbose=True)

    m.drawcountries(linewidth=1)

    lat = input_ds.variables['y'][:].values
    lon = input_ds.variables['x'][:].values
    lon, lat = np.meshgrid(lon, lat)

    m.pcolormesh(lon, lat, input,
                 latlon=True, cmap='Spectral_r')

    parallels = np.arange(34, 42, 2.)  # make latitude lines ever 5 degrees from 30N-50N
    meridians = np.arange(18, 30, 2)  # make longitude lines every 5 degrees from 95W to 70W
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

    x, y, arrow_length = 0.202, 0.3, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=ax.transAxes)

    # x in kpc, return in km
    ob1 = AnchoredHScaleBar(size=0.45045, label="50km", loc=4, frameon=False,
                            pad=0.6, sep=4, linekw=dict(color="k", linewidth=0.8), bbox_to_anchor=(0.22 - 0.0445, 0.05),
                            bbox_transform=ax.transAxes)
    ob2 = AnchoredHScaleBar(size=0.45045, label="100km", loc=4, frameon=False,
                            pad=0.6, sep=4, linekw=dict(color="k", linewidth=0.8), bbox_to_anchor=(0.219, 0.05),
                            bbox_transform=ax.transAxes)
    ob3 = AnchoredHScaleBar(size=0.45045, label="150km", loc=4, frameon=False,
                            pad=0.6, sep=4, linekw=dict(color="k", linewidth=0.8),
                            bbox_to_anchor=(0.2188 + 0.0445, 0.05), bbox_transform=ax.transAxes)
    ob4 = AnchoredHScaleBar(size=0.45045, label="200km", loc=4, frameon=False,
                            pad=0.6, sep=4, linekw=dict(color="k", linewidth=0.8),
                            bbox_to_anchor=(0.2188 + 2 * 0.0445, 0.05), bbox_transform=ax.transAxes)
    ax.add_artist(ob1)
    ax.add_artist(ob2)
    ax.add_artist(ob3)
    ax.add_artist(ob4)

    cbar = plt.colorbar()
    # cbar.set_label(label='Fire Danger', size=16)
    # cbar.set_ticks([0.001, 0.955])
    # cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.tick_params(size=0)

    # plt.title(f"Variance for {time}", fontsize=20)

    plot_path = path + '/' + name + time + map_type + '.png'
    fig.savefig(plot_path, transparent=True, dpi=200, bbox_inches="tight")
    plt.show()

    return


def save_tif(ds, outputs, epistemics, aleatorics, total, time, path, name):
    da = xr.DataArray(
        data=outputs,
        dims=["y", "x"],
        coords=dict(
            x=ds['x'],
            y=ds['y'],
    ))

    # ep = xr.DataArray(
    #     data=epistemics,
    #     dims=["y", "x"],
    #     coords=dict(
    #         x=ds['x'],
    #         y=ds['y'],
    #     ))
    #
    al = xr.DataArray(
        data=aleatorics,
        dims=["y", "x"],
        coords=dict(
            x=ds['x'],
            y=ds['y'],
        ))
    
    ep = xr.DataArray(
        data=epistemics,
        dims=["y", "x"],
        coords=dict(
            x=ds['x'],
            y=ds['y'],
        ))
    
    all = xr.DataArray(
        data=total,
        dims=["y", "x"],
        coords=dict(
            x=ds['x'],
            y=ds['y'],
        ))

    tif_path = path + '/' + name + time + '.tif'
    da.rio.to_raster(tif_path)
    tif_path = path + '/' + name + time + 'aleatorics' + '.tif'
    al.rio.to_raster(tif_path)
    tif_path = path + '/' + name + time + 'epistemics' + '.tif'
    ep.rio.to_raster(tif_path)
    tif_path = path + '/' + name + time + 'total_uncertainty' + '.tif'
    all.rio.to_raster(tif_path)
    return
