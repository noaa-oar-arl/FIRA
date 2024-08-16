"""
Author: whung

This script is used for simple FRP map plotting.
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, fnmatch
from mpl_toolkits.basemap import Basemap

import warnings
warnings.simplefilter(action='ignore')



def fire_flitering(lst, lat, lon, frp, dis_thres):
    mask = np.zeros_like(frp, dtype=bool)
    for x, y in lst:
        dis = np.sqrt((lat-x)**2+(lon-y)**2)
        mask |= dis <= dis_thres

    return np.array(frp*mask)


def main_driver(f_predic, frp_option, cor_option, lat_lim, lon_lim):

    init_time = f_predic[-19:-7]
    f_output  = f_predic[f_predic.rindex('/')+1:-3]
    frp_thres = 15        # small fires with FRP<15MW are not included
    dis_thres = 15/110    # max spread distance (degree), fires beyond this distance are assumed as NEW fires


    '''Reading Data'''
    ## normalization coef
    readin = pd.read_csv('./model/model_normalization_coef.txt')
    readin = np.array(readin)
    a      = readin[0, -1]
    b      = readin[1, -1]
    del readin
    

    ## initial fire mask
    for files in os.listdir('./input/'+init_time):
        if fnmatch.fnmatch(files, f"*{'f00'}*") and os.path.isfile(os.path.join('./input/'+init_time, files)):
            f_init = './input/'+init_time+'/'+files
    
    readin   = Dataset(f_init)
    fsize    = int((readin['input'][:].shape[1]-1)/2)
    lat_init = np.round(readin['frame_lat'][:, fsize, fsize], 3)
    lon_init = np.round(readin['frame_lon'][:, fsize, fsize], 3)
    frp_init = np.array(np.append(np.expand_dims(lat_init, axis=1), np.expand_dims(lon_init, axis=1), axis=1))
    readin.close()
    del [readin, fsize, lat_init, lon_init]
    
        
    ## forecast
    readin     = Dataset(f_predic)
    time       = readin['time'][0]
    lat        = np.round(readin['grid_lat'][:], 3)
    lon        = np.round(readin['grid_lon'][:], 3)
    frp_pre    = readin['grid_predic'][0, :, :]             # lat x alon
    frame_lat  = np.round(readin['frame_lat'][:], 3)
    frame_lon  = np.round(readin['frame_lon'][:], 3)
    frame_pre  = readin['frame_predic_ori'][:, :, :, 0]     # frame x lat x lon
    
    if cor_option == 0:
        frame_post = readin['frame_predic_post'][:, :, :, 0]    # frame x lat x lon
    elif cor_option == 1:
        frame_post = readin['frame_predic_frp'][:, :, :, 0]     # frame x lat x lon
    readin.close()
    del readin
    
    date = time[:8]
    hour = time[8:10]
        

    ## spatial map - prediction
    cmap = cm.get_cmap('jet').copy()
    cmap.set_over('#9400D3')
    
    fig, ax = plt.subplots(figsize=(18, 12))    # unit=100pixel
    h = ax.get_position()
    ax.set_position([h.x0-0.04, h.y0, h.width+0.06, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    
    plt.title(date+' '+hour+'Z Fire Radiative Power - Prediction', fontsize=28, y=1.05)
    
    m = Basemap(llcrnrlon=lon_lim[0],urcrnrlon=lon_lim[-1],llcrnrlat=lat_lim[0],urcrnrlat=lat_lim[-1], projection='mill')
    m.drawcoastlines(color='k', linewidth=1)
    m.drawcountries(color='k', linewidth=1)
    m.drawstates(color='k', linewidth=1)
    m.drawmeridians(np.arange(lon_lim[0], lon_lim[-1]+1, 10), color='none', labels=[0,0,0,1], fontsize=28)
    m.drawparallels(np.arange(lat_lim[0], lat_lim[-1]+1, 5), color='none', labels=[1,0,0,0], fontsize=28)
    
    x, y = m(lon[frp_pre!=0], lat[frp_pre!=0])
    cs   = m.scatter(x, y, marker='o', c=frp_pre[frp_pre!=0], s=120, edgecolor='k', cmap=cmap, vmin=0, vmax=200)
    
    # colorbar
    cb = plt.colorbar(cs, extend='max', orientation='vertical')
    cb.set_ticks(np.arange(0, 200+1, 20))
    cb.set_label('FRP (MW)', fontsize=28, fontweight='bold')
    cb.ax.tick_params(labelsize=28)
    
    plt.savefig('./output/'+init_time+'/'+f_output+'.jpg')
    plt.close()
    del [fig, ax, h, m, x, y, cmap, cs, cb]
    
    return 0
