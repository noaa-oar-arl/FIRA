# Fire Intensity and spRead forecAst (FIRA)

Repository for a fire spread forecast model, generating hourly fire radiative power (FRP) prediction for air quality forecasting applications.

## Required Python libraries

- NumPy
- pandas
- xarray
- SciPy
- MetPy
- netCDF4
- Matplotlib (for output plotting only)
- Tensorflow
- Keras
- XGBoost

## Components

Trained machine learning models are available in `model/fira_models.zip`

| **Source Code (Python Script)**   | **Script Description**                                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `fire_model`                      | Main driver of FIRA                                                                                                |
| `rave_preprocessor`               | Data preprocessor for initial FRP data from RAVE                                                                   |
| `fire_inputgen`                   | Model input generator, including data re-gridding, fire frame selection and filtering, scaling, and normalization  |
| `fire_forecast`                   | Driver for fire spread prediction model                                                                            |
| `fire_intensity`                  | Driver for fire intensity prediction model                                                                         |
| `fire_mapgen`                     | Fire mapper for grided FRP prediction generation                                                                   |
| `fire_evaluation`                 | Visualization tool for grided FRP prediction                                                                       |

## Namelist settings

| **Namelist Option** | **Option Description**                                                 |
| ------------------- | ---------------------------------------------------------------------- |
| Filename specification                                                                       |
| `frp_input`         | Filename of FRP input files                                            |
| `model_input`       | Filename of FIRA input files                                           |
| `model_output`      | Filename of FIRA output files                                          |
| FRP source option                                                                            |
| `frp_source`        | Option of the source of initial FRP data (0: RAVE)                     |
| Simulation time settings                                                                     |
| `time_start`        | Simulation start time in UTC (YYYYMMDDHH00)                            |
| `time_end`          | Simulation end time in UTC (YYYYMMDDHH00)                              |
| `time_freq`         | Temporal frequency (in hour) of simulation                             |
| Simulation domain settings                                                                   |
| `lat_lower_left`    | Latitude of the lower left point of model domain (degree)              |
| `lat_upper_right`   | Latitude of the upper right point of model domain (degree)             |
| `lon_lower_left`    | Longitude of the lower left point of model domain (degree)             |
| `lon_upper_right`   | Longitude of the upper right point of model domain (degree)            |
| Model options                                                                                |
| `opt_frpgen`        | Option for FRP preprocessor (0: off, 1: on)                            |
| `opt_inputgen`      | Option for input generator (0: off, 1: on)                             |
| `opt_forecast`      | Option for fire spread prediction model (0: off, 1: on)                |
| `opt_intensity`     | Option for fire intensity prediction model (0: off, 1: on)             |
| `opt_mapgen`        | Option for fire mapper (0: off, 1: on)                                 |
| `opt_evaluation`    | Option for result visualization (0: off, 1: on)                        |
| Manual/Fake fire settings                                                                    |
| `man_fire`          | Option for adding manual fire (0: off, 1: on)                          |
| `man_lat`           | Latitude of manual fires, separated by `,` (degree)                    |
| `man_lon`           | Longitude of manual fires, separated by `,` (degree)                   |
| `man_lvl`           | Fire intensity (FRP) of manual fires, separated by `,` (MW)            |
| Path for required input files                                                                |
| `path_frp`          | Path for initial FRP files                                             |
| `path_elv`          | Path for terrain elevation files                                       |
| `path_ast`          | Path for surface type files                                            |
| `path_fh`           | Path for forest height files                                           |
| `path_vhi`          | Path for vegetation health index files                                 |
| `path_mete`         | Path for meteorological data files                                     |

## Inputs

1. Filename format: `[input name].[start time].f[simuluation time].nc`
   `input name` and `start time` can be specified in namelist.

2. Input netCDF file components:

| **Variable Name**   | **Variable Description**                                                           |
| ------------------- | ---------------------------------------------------------------------------------- |
| `time`              | Time in UTC                                                                        |
| `INPUTLIST`         | List of input variables                                                            |
| `input_noscale`     | Input variables per fire frames before scaling and normalization (units vary)      |
| `input`             | Input variables per fire frames after scaling and normalization (dimensionless)    |
| `frame_lat`         | Latitude of fire frames (degree)                                                   |
| `frame_lon`         | Longitude of fire frames (degree)                                                  |

3. List of input variables:

| **Variable Name** | **Variable Description**                  | **Data Source**                                                                 |
| ----------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| Fire characteristics                                                                                                                            |
| `frp`             | Fire radiative power (MW)                 | [RAVE](https://sites.google.com/view/rave-emission/)                            |
| Fuel/surface characteristics                                                                                                                    |
| `elv`             | Terrain elevation (m)                     | GriddingMachine/[Yamazaki et al. (2017)](https://doi.org/10.1002/2017GL072874)  |
| `fh`              | Forest height (m)                         | GEDI+Landsat ([GLAD](https://glad.umd.edu/dataset/GLCLUC2020))                  |
| `vhi`             | Vegetation health index (dimensionless)   | [VIIRS VHP](https://www.ospo.noaa.gov/products/land/vvhp/)                      |
| `ast`             | Surface type (dimensionless)              | [VIIRS AST](https://www.star.nesdis.noaa.gov/jpss/st.php)                       |
| Date/Time                                                                                                                                       |
| `doy`             | Day of year (dimensionless)               | N/A                                                                             |
| `hour`            | Hour of the day                           | N/A                                                                             |
| Meteorological conditions                                                                                                                       |
| `t2m`             | 2-meter temperature                       | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |
| `sh2 `            | 2-meter specific humidity                 | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |
| `prate`           | Precipitation rate (mm h-1)               | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |
| `wd`              | 10-meter wind direction (degree)          | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |
| `ws`              | 10-meter wind speed (m s-1)               | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |

## Outputs

1. Filename format: `[output name].[start time].f[forecast hour].nc`
   `output name` and `start time` can be specified in namelist.

2. Output netCDF file components:

| **Variable Name**   | **Variable Description**                                               |
| ------------------- | ---------------------------------------------------------------------- |
| `time`              | Time of FRP prediction in UTC                                          |
| `INPUTLIST`         | List of input variables                                                |
| `frame_predic_ori`  | Original fire spread prediction per fire frames (dimensionless)        |
| `frame_predic_post` | Post-processed fire spread prediction per fire frames (dimensionless)  |
| `frame_predic_frp`  | Fire intensity (FRP) prediction per fire frames (MW)                   |
| `frame_lat`         | Latitude of fire frames (degree)                                       |
| `frame_lon`         | Longitude of fire frames (degree)                                      |
| `grid_predic`       | Gridded FRP prediction (MW)                                            |
| `grid_lat`          | Latitude of gridded FRP prediction (degree)                            |
| `grid_lon`          | Longitude of gridded FRP prediction (degree)                           |
