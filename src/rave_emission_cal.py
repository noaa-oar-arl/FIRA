# -*- coding: utf-8 -*-
"""
Author: whung

This script is used to generate RAVE fire emission estimations based on Li et al. (2022).

Li, F., Zhang, X., Kondragunta, S., Lu, X., Csiszar, I., Schmidt, C. C., 2022. Hourly biomass burning emissions product from blended geostationary and polar-orbiting satellites for air quality forecasting applications. Remote Sens. Environ. 281, 113237, https://doi.org/10.1016/j.rse.2022.113237.

NOTE: THIS SCRIPT DOES NOT INCLUDED IN THE FIRE MODEL WORKFLOW.
"""

from netCDF4 import Dataset
import pandas as pd
import numpy as np
import os

import warnings
warnings.simplefilter(action='ignore')



'''Settings'''
namelist = pd.read_csv('./input/namelist', header=None, delimiter='=')
namelist = namelist[1]

time_start  = str(namelist[4].replace(' ', ''))     # yyyymmddHHMM
path_input  = './input/'+time_start
path_output = './output/'+time_start
path_rave   = str(namelist[21].replace(' ', ''))

lat_lim = [float(namelist[7]), float(namelist[8])]
lon_lim = [float(namelist[9]), float(namelist[10])]

## constants
f_scale    = 1       # FRP scale factor
factor_FRE = 3600    # the ratio of hourly FRE to 5-min-avg FRP
fc         = 0.368   # FRE biomass combustion factor (kg MJ-1)


def emi_estimator(fre, vtype):
    DM = fre * fc       # hourly consumed dry matter (kg)

    EF = np.zeros(DM.shape)             # PM2.5 emission factor (g kg-1)
    EF[vtype==1] = 12.8                 # forest
    EF[(vtype>=2) & (vtype<=4)] = 7.17  # savanna, shrubland, grassland
    EF[vtype==5] = 6.26                 # cropland

    M = (DM * EF) / 1000    # total mass of PM2.5 emission (kg)

    return M



'''Land cover'''
## domain
readin  = Dataset(path_input+'/rave.frp.conus.'+time_start+'.nc')
LAT     = readin['grid_lat'][:]
LON     = readin['grid_lon'][:]
readin.close()


## RAVE land cover
date = time_start[:8]
hour = time_start[8:10]

#fname = 'Hourly_Emissions_3km_'+date+'0000_'+date+'2300.nc'
fname = 'RAVE-HrlyEmiss-3km_v1r3_blend_s'+date+hour+'0000'
f_rave = [f for f in os.listdir(path_rave) if fname in f][0]
f_rave = path_rave + '/' + f_rave

readin = Dataset(f_rave)
yt     = np.flip(readin['grid_latt'][:, 0])
xt     = readin['grid_lont'][0, :]
yt     = np.round(yt, 3)
xt     = np.round(xt, 3)
index1 = np.squeeze(np.argwhere((yt>=lat_lim[0]) & (yt<=lat_lim[1])))
index2 = np.squeeze(np.argwhere((xt>=lon_lim[0]) & (xt<=lon_lim[1])))

VTYPE = np.squeeze(readin['land_cover'][:])
VTYPE = np.flipud(VTYPE)

if (index1[0] == 0) & (index2[0] == 0):
    VTYPE = VTYPE[index1[0]:index1[-1]+2, index2[0]:index2[-1]+2]
elif index1[0] == 0:
    VTYPE = VTYPE[index1[0]:index1[-1]+2, index2[0]-1:index2[-1]+2]
elif index2[0] == 0:
    VTYPE = VTYPE[index1[0]-1:index1[-1]+2, index2[0]:index2[-1]+2]
else:
    VTYPE = VTYPE[index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]

readin.close()
del [readin, yt, xt, index1, index2]

#print(LAT.shape, LON.shape, VTYPE.shape)



'''Fire emission estiamtion'''
filename = [file for file in os.listdir(path_output) if ('fire.pred.conus' in file) and (file.endswith('.nc'))]
filename = np.sort(filename)

NN = len(filename)

for i in np.arange(NN):
    f_time   = int(filename[i][30:32])
    f_input  = path_output+'/'+filename[i]
    f_output = path_output+'/'+(filename[i].replace('pred', 'emi'))

    if f_scale != 1:
        f_output = f_output.replace('.nc', '.scale'+str(f_scale)+'.nc')

    #print(f_time, f_input, f_output)

    readin = Dataset(f_input)
    FRP    = readin['grid_predic'][0, :, :] * f_scale
    FRP_SD = np.ones(FRP.shape)
    FRE    = FRP * factor_FRE
    EMI    = emi_estimator(FRE, VTYPE)
    readin.close()

    print(LAT.shape, LON.shape)
    print(FRP.shape, np.min(FRP[FRP!=0]), np.max(FRP[FRP!=0]))
    print(FRE.shape, np.min(FRE[FRE!=0]), np.max(FRE[FRE!=0]))
    print(EMI.shape, np.min(EMI[EMI!=0]), np.max(EMI[EMI!=0]))

    masked_FRP = np.ma.masked_array(FRP, FRP==0, fill_value=-1)
    masked_SD  = np.ma.masked_array(FRP_SD, FRP_SD==0, fill_value=-1)
    masked_FRE = np.ma.masked_array(FRE, FRE==0, fill_value=-1)
    masked_EMI = np.ma.masked_array(EMI, EMI==0, fill_value=-1)
    

    ## write outputs
    f = Dataset(f_output, 'w')
    f.createDimension('time', 1)
    f.createDimension('grid_yt', LAT.shape[0])
    f.createDimension('grid_xt', LAT.shape[1])

    var_time = f.createVariable('time', 'i4', ('time',))
    var_lat  = f.createVariable('grid_latt', 'f4', ('grid_yt', 'grid_xt'))
    var_lon  = f.createVariable('grid_lont', 'f4', ('grid_yt', 'grid_xt'))
    var_frp  = f.createVariable('FRP_MEAN', 'f4',  ('time', 'grid_yt', 'grid_xt'), fill_value=-1)
    var_sd   = f.createVariable('FRP_SD', 'f4',  ('time', 'grid_yt', 'grid_xt'), fill_value=-1)
    var_fre  = f.createVariable('FRE', 'f4',  ('time', 'grid_yt', 'grid_xt'), fill_value=-1)
    var_emi  = f.createVariable('PM2.5', 'f4',  ('time', 'grid_yt', 'grid_xt'), fill_value=-1)

    var_time[:] = f_time
    var_lat[:]  = np.flip(LAT)
    var_lon[:]  = LON
    var_frp[:]  = np.flipud(masked_FRP)
    var_sd[:]   = np.flipud(masked_SD)
    var_fre[:]  = np.flipud(masked_FRE)
    var_emi[:]  = np.flipud(masked_EMI)
    f.close() 

    print(i+1, '/', NN)
