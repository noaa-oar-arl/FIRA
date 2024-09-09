"""
This script is used to generate fire spread forecast with well-trained model.
"""

import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from netCDF4 import Dataset
from tensorflow.keras.optimizers import Adamax

warnings.simplefilter(action="ignore")


def normalization_coef():
    readin = pd.read_csv("./model/model_normalization_coef.txt")
    readin = np.array(readin)
    a1 = readin[0, 0]
    b1 = readin[1, 0]
    a2 = readin[0, -1]
    b2 = readin[1, -1]
    return a1, b1, a2, b2


def lr_tracer(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def frame_upscaling_test(DATA):
    resize = 8
    XX = DATA.shape[1]
    YY = DATA.shape[2]
    return tf.image.resize(DATA, [XX * resize, YY * resize], method="nearest")


def loss_function(w0, normalized_zero):
    def loss(y_true, y_pred):
        y_error = K.square(y_true - y_pred)
        y_error = K.switch(
            K.equal(K.round(y_true * (10**7)), normalized_zero * (10**7)),
            w0 * y_error,
            y_error,
        )
        return K.mean(y_error)

    return loss


def scheduler(epoch, lr):
    if (epoch != 0) & (epoch % 100 == 0):
        return lr * 0.7
    else:
        return lr


def spread_forecast(f_input, f_output):
    # ---- Settings ----
    # constants
    learning_rate = 1e-4
    nweight = 0.6  # loss weight for zeros

    # variable list
    firelist = ["frp"]
    geolist = ["lat", "lon", "elv", "ast", "doy", "hour"]
    veglist = ["fh", "vhi"]
    metlist = ["t2m", "sh2", "prate", "wd", "ws"]
    INPUTLIST = firelist + geolist + veglist + metlist

    # ---- Reading Inputs ----
    readin = Dataset(f_input)
    time_fc = (pd.to_datetime(readin["time"][0]) + pd.Timedelta("1H")).strftime(
        "%Y%m%d%H%M"
    )
    X_lat = readin["frame_lat"][:]
    X_lon = readin["frame_lon"][:]
    X_initial = readin["input"][:]
    INPUTLIST = readin["INPUTLIST"][:]
    NN = len(INPUTLIST)
    readin.close()

    print("---- Variable list:")
    print(INPUTLIST)
    print("---- Input data:")
    print(X_initial.shape, np.min(X_initial), np.max(X_initial))

    # normalization coef
    a1, b1, a2, b2 = normalization_coef()  # 1: input/2: output

    normalized_zero_input = np.round((0 - b1) / a1, 7)
    normalized_zero = np.round((0 - b2) / a2, 7)

    # fire location only
    idx0 = np.round(X_initial[:, :, :, 0], 7) == normalized_zero_input
    idx1 = np.round(X_initial[:, :, :, 0], 7) != normalized_zero_input
    X_initial[idx0, 0] = 0
    X_initial[idx1, 0] = 1
    del [idx0, idx1]

    # ---- Reading Model ----
    print("---- Model configuration:")
    opt = Adamax(
        learning_rate=learning_rate,
        beta_1=0.90,
        beta_2=0.999,
        epsilon=1e-08,
        decay=1e-06,
    )
    lr = lr_tracer(opt)
    model = load_model(
        "./model/spread_model.h5",
        custom_objects={"loss": loss_function(nweight, normalized_zero), "lr": lr},
    )
    model.summary()

    # ---- Forecast ----
    nlat = X_initial.shape[1]
    nlon = X_initial.shape[2]

    X = tf.convert_to_tensor(X_initial)
    X = frame_upscaling_test(X)

    Y = model.predict(X)
    Y = np.array(Y)

    predic = np.copy(Y)
    predic[predic >= 0.2] = 1
    predic[predic < 0.2] = 0

    # downscaling
    Y = tf.convert_to_tensor(Y)
    Y = tf.image.resize(Y, [nlat, nlon], method="nearest")
    Y = Y.numpy()

    predic = tf.convert_to_tensor(predic)
    predic = tf.image.resize(predic, [nlat, nlon], method="nearest")
    predic = predic.numpy()

    print("---- Spread prediction:")
    print("Original output:", Y.shape, np.min(Y), np.max(Y))
    print("Post-processed prediction:", predic.shape, np.min(predic), np.max(predic))

    # save prediction
    f = Dataset(f_output, "w")
    f.createDimension("flen", predic.shape[0])
    f.createDimension("xlen", nlat)
    f.createDimension("ylen", nlon)
    f.createDimension("num_input", NN)
    f.createDimension("num_output", 1)
    f.createDimension("time", 1)
    var_time = f.createVariable("time", str, ("time",))
    var_ori = f.createVariable(
        "frame_predic_ori", "float", ("flen", "xlen", "ylen", "num_output")
    )
    var_post = f.createVariable(
        "frame_predic_post", "float", ("flen", "xlen", "ylen", "num_output")
    )
    var_lat = f.createVariable("frame_lat", "float", ("flen", "xlen", "ylen"))
    var_lon = f.createVariable("frame_lon", "float", ("flen", "xlen", "ylen"))
    var_list = f.createVariable("INPUTLIST", str, ("num_input",))
    var_time[:] = np.array(time_fc).astype(str)
    var_ori[:] = Y
    var_post[:] = predic
    var_lat[:] = X_lat
    var_lon[:] = X_lon
    var_list[:] = np.array(INPUTLIST).astype(str)
    f.close()
