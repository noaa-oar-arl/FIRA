"""
This script is used for fire intensity prediction.
"""

import warnings

import numpy as np
import xgboost as xgb
from netCDF4 import Dataset

warnings.simplefilter(action="ignore")


def main_driver(f_input, f_output, forecast_hour):
    # ---- Reading Data ----
    # non-scaled input
    readin = Dataset(f_input)
    inputs = readin["input_noscale"][:]
    readin.close()
    del readin

    # model first guess
    readin = Dataset(f_output)
    guess = readin["frame_predic_post"][:, :, :, 0]
    readin.close()
    del readin

    # ---- Intensity prediction ----
    model = xgb.XGBRegressor()
    model.load_model("./model/intensity_model_rf.json")

    fire = np.copy(guess)
    mask = fire == 1
    fire[mask] = model.predict(inputs[mask, :].reshape(-1, inputs.shape[-1]))
    fire[fire < 1] = 0

    print("---- Intensity prediction:")
    print("Fire location prediction:", guess.shape, np.min(guess), np.max(guess))
    print("Fire intensity prediction:", fire.shape, np.min(fire), np.max(fire))

    # save prediction
    f = Dataset(f_output, "a")
    var_fire = f.createVariable(
        "frame_predic_frp", "float", ("flen", "xlen", "ylen", "num_output")
    )
    var_fire[:] = np.expand_dims(fire, axis=-1)

    f.close()
