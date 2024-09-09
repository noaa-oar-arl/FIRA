"""
This script is used to generate model input files. Pre-process gridded fire map needed.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from metpy.calc import wind_direction
from metpy.units import units
from netCDF4 import Dataset
from scipy import ndimage
from scipy.interpolate import griddata

warnings.simplefilter(action="ignore")


def file_finder(item, date, initial_hour, hour):
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist[1]

    if item == "elv":
        return str(namelist[22].replace(" ", "")) + "/ELEV_4X_1Y_V1_Yamazaki.nc"
    elif item == "ast":
        return str(namelist[23].replace(" ", "")) + "/VIIRS_AST_2020_grid3km.nc"
    elif item == "fh":
        return str(namelist[24].replace(" ", "")) + "/GLAD_FH_grid3km_2020.nc"
    elif item == "vhi":
        week = datetime(int(date[:4]), int(date[4:6]), int(date[6:])).isocalendar().week
        lastweek = datetime(int(date[:4]), 12, 31).isocalendar().week
        if week == lastweek:
            return (
                str(namelist[25].replace(" ", ""))
                + "/npp/VHP.G04.C07.npp.P2020"
                + ("%03d" % (lastweek - 1))
                + ".VH.nc",
                str(namelist[25].replace(" ", ""))
                + "/j01/VHP.G04.C07.j01.P2020"
                + ("%03d" % (lastweek - 1))
                + ".VH.nc",
            )
        else:
            return (
                str(namelist[25].replace(" ", ""))
                + "/npp/VHP.G04.C07.npp.P2020"
                + ("%03d" % week)
                + ".VH.nc",
                str(namelist[25].replace(" ", ""))
                + "/j01/VHP.G04.C07.j01.P2020"
                + ("%03d" % week)
                + ".VH.nc",
            )
    elif (item == "t2m") or (item == "sh2") or (item == "wd") or (item == "ws"):
        return (
            str(namelist[26].replace(" ", ""))
            + "/hrrr.t"
            + initial_hour
            + "z.wrfsfcf"
            + ("%02d" % hour)
            + ".grib2"
        )
    elif item == "prate":
        return (
            str(namelist[26].replace(" ", ""))
            + "/hrrr.t"
            + initial_hour
            + "z.wrfsfcf"
            + ("%02d" % hour)
            + ".grib2"
        )


def mapping(xgrid, ygrid, data, xdata, ydata, map_method, fill_value):
    output = griddata(
        (xdata, ydata), data, (xgrid, ygrid), method=map_method, fill_value=fill_value
    )
    return output


def wind_conversion(
    LAT, U, V
):  # from HRRR FAQ (https://rapidrefresh.noaa.gov/faq/HRRR.faq.html)
    rotcon_p = 0.622515
    lat_tan_p = 38.5

    angle = rotcon_p * (LAT - lat_tan_p) * 0.017453  # LAMBERT CONFORMAL PROJECTION
    sinx2 = np.sin(angle)
    cosx2 = np.cos(angle)

    U_new = cosx2 * U + sinx2 * V
    V_new = (-1) * sinx2 * U + cosx2 * V
    return U_new, V_new


def normalization(var, NN):
    a = []
    b = []
    for i in np.arange(NN):
        a = np.append(a, np.nanmax(var[:, :, :, i]) - np.nanmin(var[:, :, :, i]))
        b = np.append(b, np.nanmin(var[:, :, :, i]))
    return a, b


def add_manual_fire(frp_map, lat_map, lon_map, flat, flon, flvl):
    for i in np.arange(len(flvl)):
        dis = (lat_map - flat[i]) ** 2 + (lon_map - flon[i]) ** 2

        if frp_map[dis == np.min(dis)] == 0:
            frp_map[dis == np.min(dis)] = flvl[i]

    return frp_map


def main_driver(initial_hour, forecast_hour, f_input, f_output, lat_lim, lon_lim):
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist[1]
    manual_fire = int(namelist[17])

    # ---- Global Settings ----
    # constants
    frp_thres = 15  # fires with low FRP will be removed
    fsize = 2  # extend fire grid

    # variable list
    firelist = ["frp"]
    geolist = ["elv", "ast", "doy", "hour"]
    veglist = ["fh", "vhi"]
    metlist = ["t2m", "sh2", "prate", "wd", "ws"]
    INPUTLIST = firelist + geolist + veglist + metlist

    # ---- Reading Fire Map & Input Settings ----
    if forecast_hour == 0:  # initial fire
        readin = Dataset(f_input)
        time = readin["time"][0]
        LAT = readin["grid_lat"][:]
        LON = readin["grid_lon"][:]
        FRP = readin["frp"][0, :, :]
        readin.close()
        del readin

        if manual_fire == 1:
            man_lat = np.array(str(namelist[18].replace(" ", "")).split(",")).astype(
                float
            )
            man_lon = np.array(str(namelist[19].replace(" ", "")).split(",")).astype(
                float
            )
            man_lvl = np.array(str(namelist[20].replace(" ", "")).split(",")).astype(
                float
            )
            add_manual_fire(FRP, LAT, LON, man_lat, man_lon, man_lvl)
            print("---- Warning!!! Manual fire added!!!")
            del [man_lat, man_lon, man_lvl]
    else:
        readin = Dataset(f_input)
        time = readin["time"][0]
        LAT = readin["grid_lat"][:]
        LON = readin["grid_lon"][:]
        FRP = readin["grid_predic"][0, :, :]
        readin.close()
        del readin

    NN = len(INPUTLIST)
    tt = time  # yyyymmddHHMM
    dd = time[:8]  # yyyymmdd
    hh = time[8:10]  # HH

    INPUT = np.empty([LAT.shape[0], LAT.shape[1], NN])
    print("---- Selecting data...", dd, hh + "Z")

    # ---- Reading Input Variables ----
    # frp, lat, lon
    INPUT[:, :, INPUTLIST.index("frp")] = np.copy(FRP)

    # elv
    filename = file_finder("elv", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "elv", "Terminated!")
        print(filename)
        return 1

    readin = Dataset(filename)
    yt = readin["lat"][:]
    xt = readin["lon"][:]
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = readin["data"][index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = readin["data"][
            index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
        ]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = readin["data"][
            index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2
        ]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = readin["data"][
            index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
        ]
    data[data < 0] = 0

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT, LON, data.flatten(), yt_grid.flatten(), xt_grid.flatten(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("elv")] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]

    # ast
    filename = file_finder("ast", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "ast", "Terminated!")
        print(filename)
        return 1

    readin = Dataset(filename)
    yt = np.flip(readin["lat"][:, 0])
    xt = readin["lon"][0, :]
    yt = np.round(yt, 3)
    xt = np.round(xt, 3)
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    data = np.squeeze(readin["ast"][0, :, :])
    data = np.flipud(data)

    # ast, hour x lat x lon
    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = data[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = data[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = data[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = data[index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT,
        LON,
        data.flatten(),
        yt_grid.flatten(),
        xt_grid.flatten(),
        "nearest",
        np.nan,
    )

    INPUT[:, :, INPUTLIST.index("ast")] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]

    # doy
    data = np.empty(LAT.shape)
    data[:] = int(pd.to_datetime(dd).strftime("%-j"))

    INPUT[:, :, INPUTLIST.index("doy")] = np.copy(data).astype(int)
    del data

    # hour
    readin = Dataset("./fix/timezones_voronoi_1x1.nc")
    yt = np.flip(readin["lat"][:])
    xt = readin["lon"][:]
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    offset = np.squeeze(readin["UTC_OFFSET"][0, :, :])
    offset = np.flipud(offset)

    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        offset = offset[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        offset = offset[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        offset = offset[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        offset = offset[index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    offset_grid = mapping(
        LAT,
        LON,
        offset.flatten(),
        yt_grid.flatten(),
        xt_grid.flatten(),
        "nearest",
        np.nan,
    )

    data = int(hh) + offset_grid
    data[data < 0] = data[data < 0] + 24
    data[data > 24] = data[data > 24] - 24

    INPUT[:, :, INPUTLIST.index("hour")] = np.copy(data).astype(int)

    readin.close()
    del [readin, xt, xt_grid, yt, yt_grid, offset, offset_grid, data]

    # fh
    filename = file_finder("fh", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "fh", "Terminated!")
        print(filename)
        return 1

    readin = Dataset(filename)
    yt = np.flip(readin["lat"][:, 0])
    xt = readin["lon"][0, :]
    yt = np.round(yt, 3)
    xt = np.round(xt, 3)
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    data = np.squeeze(readin["forest_canopy_height"][:])
    data = np.flipud(data)

    # fh, lat x lon
    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = data[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = data[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = data[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = data[index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT, LON, data.flatten(), yt_grid.flatten(), xt_grid.flatten(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan
    data_grid[np.isnan(data_grid)] = 0

    INPUT[:, :, INPUTLIST.index("fh")] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]

    # vhi
    filename_npp, filename_j01 = file_finder("vhi", dd, initial_hour, forecast_hour)

    if (os.path.isfile(filename_npp) is False) or (
        os.path.isfile(filename_j01) is False
    ):
        print("---- Incorrect input file:", "vhi", "Terminated!")
        print(filename_npp, filename_j01)
        return 1

    # npp
    readin = Dataset(filename_npp)
    yt = np.flip(readin["latitude"][:])
    xt = readin["longitude"][:]
    yt = np.round(yt, 3)
    xt = np.round(xt, 3)
    xt[xt < 0] = xt[xt < 0] + 360
    vci_npp = np.flipud(np.asarray(readin["VCI"][:]))
    tci_npp = np.flipud(np.asarray(readin["TCI"][:]))
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        vci_npp = vci_npp[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
        tci_npp = tci_npp[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        vci_npp = vci_npp[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
        tci_npp = tci_npp[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        vci_npp = vci_npp[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
        tci_npp = tci_npp[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        vci_npp = vci_npp[
            index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
        ]
        tci_npp = tci_npp[
            index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
        ]

    vci_npp[vci_npp == -999] = np.nan
    tci_npp[tci_npp == -999] = np.nan
    readin.close()
    del readin

    # j01
    readin = Dataset(filename_j01)
    vci_j01 = np.flipud(np.asarray(readin["VCI"][:]))
    tci_j01 = np.flipud(np.asarray(readin["TCI"][:]))

    if (index1[0] == 0) & (index2[0] == 0):
        vci_j01 = vci_j01[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
        tci_j01 = tci_j01[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        vci_j01 = vci_j01[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
        tci_j01 = tci_j01[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    elif index2[0] == 0:
        vci_j01 = vci_j01[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
        tci_j01 = tci_j01[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
    else:
        vci_j01 = vci_j01[
            index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
        ]
        tci_j01 = tci_j01[
            index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
        ]

    vci_j01[vci_j01 == -999] = np.nan
    tci_j01[tci_j01 == -999] = np.nan
    readin.close()
    del readin

    # combine two satellites
    vci = np.nanmean([vci_npp, vci_j01], axis=0)
    tci = np.nanmean([tci_npp, tci_j01], axis=0)
    vci = vci / 100
    tci = tci / 100
    del [vci_npp, tci_npp, vci_j01, tci_j01]

    # calculate vhi
    data = 0.3 * vci + 0.7 * tci
    data[np.isnan(data)] = -999
    del [vci, tci]

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT,
        LON,
        data.flatten(),
        yt_grid.flatten(),
        xt_grid.flatten(),
        "nearest",
        np.nan,
    )
    data_grid[data_grid == -999] = np.nan

    INPUT[:, :, INPUTLIST.index("vhi")] = np.copy(data_grid)
    del [
        filename_npp,
        filename_j01,
        yt,
        xt,
        yt_grid,
        xt_grid,
        index1,
        index2,
        data,
        data_grid,
    ]

    # t2m
    filename = file_finder("t2m", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "t2m", "Terminated!")
        print(filename)
        return 1

    readin = xr.open_dataset(
        filename,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
    )
    yt = readin["latitude"].data
    xt = readin["longitude"].data
    data = readin["t2m"].data

    data_grid = mapping(
        LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("t2m")] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, data, data_grid]

    # sh2
    filename = file_finder("sh2", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "sh2", "Terminated!")
        print(filename)
        return 1

    readin = xr.open_dataset(
        filename,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
    )
    yt = readin["latitude"].data
    xt = readin["longitude"].data
    data = readin["sh2"].data

    data_grid = mapping(
        LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("sh2")] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, data, data_grid]

    # prate
    filename = file_finder("prate", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "prate", "Terminated!")
        print(filename)
        return 1

    readin = xr.open_dataset(
        filename,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "surface", "stepType": "instant"},
    )
    yt = readin["latitude"].data
    xt = readin["longitude"].data
    data = readin["prate"].data
    data = data * 3600  # kg m-2 s-1 -> hourly accumulation

    data_grid = mapping(
        LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("prate")] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, data, data_grid]

    # ws, wd
    filename = file_finder("ws", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "ws, wd", "Terminated!")
        print(filename)
        return 1

    readin = xr.open_dataset(
        filename,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10},
    )
    yt = readin["latitude"].data
    xt = readin["longitude"].data
    u = readin["u10"].data
    v = readin["v10"].data

    u_new, v_new = wind_conversion(yt, u, v)

    u_grid = mapping(
        LAT, LON, u_new.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )
    v_grid = mapping(
        LAT, LON, v_new.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )

    # for canopy wind calculation
    UGRID = np.copy(u_grid)
    VGRID = np.copy(v_grid)

    # ws, lat x lon
    ws_grid = np.sqrt(u_grid**2 + v_grid**2)

    # wd, lat x lon
    u_grid = units.Quantity(u_grid, "m/s")
    v_grid = units.Quantity(v_grid, "m/s")
    wd_grid = wind_direction(u_grid, v_grid, convention="from")

    INPUT[:, :, INPUTLIST.index("ws")] = np.copy(ws_grid)
    INPUT[:, :, INPUTLIST.index("wd")] = np.copy(wd_grid)
    del [filename, readin, yt, xt, u, v, u_new, v_new, u_grid, v_grid, ws_grid, wd_grid]

    print("Start framing... Original data:", INPUT.shape)

    # ---- Fire Framing ----
    XX = LAT.shape[0]
    YY = LAT.shape[1]

    total = 0
    skip = 0

    # fire mask
    MASK = np.zeros(LAT.shape)
    MASK[FRP != 0] = 1

    # fire location
    lw, num = ndimage.label(MASK)
    lw = lw.astype(float)
    lw[lw == 0] = np.nan

    # fire frame
    for j in np.arange(1, num + 1, 1):
        total = total + 1
        index = np.argwhere(lw == j)

        if index.shape[0] == 1:
            loc = np.squeeze(index)
        else:
            xx = np.mean(LAT[index[:, 0], index[:, 1]])
            yy = np.mean(LON[index[:, 0], index[:, 1]])
            dis = (LAT - xx) ** 2 + (LON - yy) ** 2
            loc = np.squeeze(np.argwhere(dis == np.min(dis)))
            del [xx, yy, dis]

        if loc.size > 2:
            loc = loc[0, :]

        if (
            (loc[0] - fsize < 0)
            or (loc[0] + fsize + 1 > XX)
            or (loc[1] - fsize < 0)
            or (loc[1] + fsize + 1 > YY)
        ):
            skip = skip + 1
            continue

        X_fire = INPUT[
            loc[0] - fsize : loc[0] + fsize + 1, loc[1] - fsize : loc[1] + fsize + 1, :
        ]
        X_lat = LAT[
            loc[0] - fsize : loc[0] + fsize + 1, loc[1] - fsize : loc[1] + fsize + 1
        ]
        X_lon = LON[
            loc[0] - fsize : loc[0] + fsize + 1, loc[1] - fsize : loc[1] + fsize + 1
        ]

        if "INPUTFRAME" in locals():
            INPUTFRAME = np.append(INPUTFRAME, np.expand_dims(X_fire, axis=0), axis=0)
            LATFRAME = np.append(LATFRAME, np.expand_dims(X_lat, axis=0), axis=0)
            LONFRAME = np.append(LONFRAME, np.expand_dims(X_lon, axis=0), axis=0)
        else:
            INPUTFRAME = np.expand_dims(X_fire, axis=0)
            LATFRAME = np.expand_dims(X_lat, axis=0)
            LONFRAME = np.expand_dims(X_lon, axis=0)

        del [index, loc, X_fire, X_lat, X_lon]
    del [lw, num, INPUT, FRP, LAT, LON, MASK]

    if "INPUTFRAME" in locals():
        pass
    else:
        print("---- " + tt + " no available frames.")
        return 2

    # remove frames with NaN
    index = []
    for i in np.arange(NN):
        X = np.sum(INPUTFRAME[:, :, :, i], axis=(1, 2))
        index = np.append(index, np.squeeze(np.argwhere(np.isnan(X))))
        del X

    index = index.astype(int)

    if index.size != 0:
        INPUTFRAME = np.delete(INPUTFRAME, index, axis=0)
        LATFRAME = np.delete(LATFRAME, index, axis=0)
        LONFRAME = np.delete(LONFRAME, index, axis=0)
    nan = index.size
    del index

    # remove isolated small fires
    fire_map = np.copy(INPUTFRAME[:, :, :, 0])
    fire_map[fire_map != 0] = 1
    fire_count = np.sum(fire_map, axis=(1, 2))
    fire_max = np.max(INPUTFRAME[:, :, :, 0], axis=(1, 2))
    index = np.squeeze(np.argwhere((fire_count == 1) & (fire_max < frp_thres)))

    if index.size != 0:
        INPUTFRAME = np.delete(INPUTFRAME, index, axis=0)
        LATFRAME = np.delete(LATFRAME, index, axis=0)
        LONFRAME = np.delete(LONFRAME, index, axis=0)
    small = index.size
    del [fire_map, fire_count, fire_max, index]

    # remove frames with water bodies (AST=17)
    index = np.argwhere(INPUTFRAME[:, :, :, INPUTLIST.index("ast")] == 17)[:, 0]
    index = np.unique(index)

    if index.size != 0:
        INPUTFRAME = np.delete(INPUTFRAME, index, axis=0)
        LATFRAME = np.delete(LATFRAME, index, axis=0)
        LONFRAME = np.delete(LONFRAME, index, axis=0)
    water = index.size
    del index

    print("---- Fire framing complete!")

    if INPUTFRAME.shape[1] == 0:
        print("----No available frames!")
        return 2
    else:
        print("All:", total, "Skip:", skip, "NaN:", nan, "Water", water)
        print("Final:", INPUTFRAME.shape)

    INPUTFRAME_ori = np.copy(INPUTFRAME)

    # ---- Scaling ----
    for X in ["frp", "prate"]:
        i = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1[X1 != 0] = np.log(X1[X1 != 0])
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        del [i, X1]

    for X in ["sh2", "ws"]:
        i = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1 = np.sqrt(X1)
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        del [i, X1]

    for X in ["fh", "elv"]:
        i = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1 = (X1) ** (1 / 3)
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        del [i, X1]

    # ---- Normalization ----
    # normalization coef
    coef = pd.read_csv("./model/model_normalization_coef.txt")
    coef = np.array(coef)
    a = coef[0, np.append(0, np.arange(3, 14))]
    b = coef[1, np.append(0, np.arange(3, 14))]

    for i in np.arange(NN):
        X = np.copy(INPUTFRAME[:, :, :, i])
        X = (X - b[i]) / a[i]
        INPUTFRAME[:, :, :, i] = np.copy(X)
        del X

    # ---- Writing Model Input ----
    INPUTFRAME[np.isnan(INPUTFRAME)] = -999

    f = Dataset(f_output, "w")
    f.createDimension("time", 1)
    f.createDimension("flen", INPUTFRAME.shape[0])
    f.createDimension("xlen", INPUTFRAME.shape[1])
    f.createDimension("ylen", INPUTFRAME.shape[2])
    f.createDimension("num_input", NN)
    var_time = f.createVariable("time", str, ("time",))
    var_input0 = f.createVariable(
        "input_noscale", "float", ("flen", "xlen", "ylen", "num_input")
    )
    var_input = f.createVariable(
        "input", "float", ("flen", "xlen", "ylen", "num_input")
    )
    var_lat = f.createVariable("frame_lat", "float", ("flen", "xlen", "ylen"))
    var_lon = f.createVariable("frame_lon", "float", ("flen", "xlen", "ylen"))
    var_list = f.createVariable("INPUTLIST", str, ("num_input",))

    var_time[:] = np.array(time).astype(str)
    var_input0[:] = INPUTFRAME_ori
    var_input[:] = INPUTFRAME
    var_lat[:] = LATFRAME
    var_lon[:] = LONFRAME
    var_list[:] = np.array(INPUTLIST).astype(str)
    f.close()

    return 0
