"""
Author: whung

This script is used for fire intensity prediction. 
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xgboost as xgb
import pickle

import warnings
warnings.simplefilter(action='ignore')



def main_driver(f_input, f_output, forecast_hour):

    '''Reading Data'''
    ## normalization coef
    readin = pd.read_csv('./model/model_normalization_coef.txt')
    readin = np.array(readin)
    a      = readin[0, np.append(0, np.arange(3, 14))]
    b      = readin[1, np.append(0, np.arange(3, 14))]
    
    ## non-scaled input
    readin = Dataset(f_input)
    inputs = readin['input_noscale'][:]
    readin.close()
    del readin
    
    ## model first guess
    readin = Dataset(f_output)
    guess  = readin['frame_predic_post'][:, :, :, 0]
    readin.close()
    del readin
    


    '''Intensity prediction'''
    model = xgb.XGBRegressor()
    model.load_model('./model/intensity_model_rf.json')
    
    fire = np.copy(guess)
    mask = fire == 1
    fire[mask] = model.predict(inputs[mask, :].reshape(-1, inputs.shape[-1]))
    fire[fire < 1] = 0

    
    
    print('---- Intensity prediction:')
    print('Fire location prediction:', guess.shape, np.min(guess), np.max(guess))
    print('Fire intensity prediction:', fire.shape, np.min(fire), np.max(fire))
    
    
    
    ## save prediction
    f = Dataset(f_output, 'a')
    var_fire = f.createVariable('frame_predic_frp', 'float', ('flen', 'xlen', 'ylen', 'num_output'))
    var_fire[:] = np.expand_dims(fire, axis=-1)

    f.close()
