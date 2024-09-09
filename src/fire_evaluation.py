"""
This script is used for simple FRP map plotting.
"""

import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

warnings.simplefilter(action="ignore")


def fire_flitering(lst, lat, lon, frp, dis_thres):
    mask = np.zeros_like(frp, dtype=bool)
    for x, y in lst:
        dis = np.sqrt((lat - x) ** 2 + (lon - y) ** 2)
        mask |= dis <= dis_thres

    return np.array(frp * mask)


def main_driver(f_predic, frp_option, cor_option, lat_lim, lon_lim):

    init_time = f_predic[-19:-7]
    f_output = f_predic[(f_predic.rindex("/") + 1) : -3]

    # reading data
    readin = Dataset(f_predic)
    time = readin["time"][0]
    lat = np.round(readin["grid_lat"][:], 3)
    lon = np.round(readin["grid_lon"][:], 3)
    frp_pre = readin["grid_predic"][0, :, :]  # lat x alon
    readin.close()
    del readin

    date = time[:8]
    hour = time[8:10]

    # spatial map - prediction
    cmap = cm.get_cmap("jet").copy()
    cmap.set_over("#9400D3")

    fig, ax = plt.subplots(figsize=(18, 12))  # unit=100pixel
    h = ax.get_position()
    ax.set_position([h.x0 - 0.04, h.y0, h.width + 0.06, h.height])
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)

    plt.title(
        date + " " + hour + "Z Fire Radiative Power - Prediction", fontsize=28, y=1.05
    )

    m = Basemap(
        llcrnrlon=lon_lim[0],
        urcrnrlon=lon_lim[-1],
        llcrnrlat=lat_lim[0],
        urcrnrlat=lat_lim[-1],
        projection="mill",
    )
    m.drawcoastlines(color="k", linewidth=1)
    m.drawcountries(color="k", linewidth=1)
    m.drawstates(color="k", linewidth=1)
    m.drawmeridians(
        np.arange(lon_lim[0], lon_lim[-1] + 1, 10),
        color="none",
        labels=[0, 0, 0, 1],
        fontsize=28,
    )
    m.drawparallels(
        np.arange(lat_lim[0], lat_lim[-1] + 1, 5),
        color="none",
        labels=[1, 0, 0, 0],
        fontsize=28,
    )

    x, y = m(lon[frp_pre != 0], lat[frp_pre != 0])
    cs = m.scatter(
        x,
        y,
        marker="o",
        c=frp_pre[frp_pre != 0],
        s=120,
        edgecolor="k",
        cmap=cmap,
        vmin=0,
        vmax=200,
    )

    # colorbar
    cb = plt.colorbar(cs, extend="max", orientation="vertical")
    cb.set_ticks(np.arange(0, 200 + 1, 20))
    cb.set_label("FRP (MW)", fontsize=28, fontweight="bold")
    cb.ax.tick_params(labelsize=28)

    plt.savefig("./output/" + init_time + "/" + f_output + ".jpg")
    plt.close()
    del [fig, ax, h, m, x, y, cmap, cs, cb]

    return 0
