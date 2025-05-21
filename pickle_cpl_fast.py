#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""
import dask
import datetime
import calendar
import numpy as np
import os
import sys
import xarray as xr
import time
import pandas as pd
import pickle

import namelist
from intensity import coupled_fast, ocean
from thermo import calc_thermo
from track import env_wind
from wind import tc_wind
from util import basins, input, mat

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

b = basins.TC_Basin(sys.argv[1])
year = int(sys.argv[2])

# Load thermodynamic and ocean variables.
if namelist.debug: print(f'Currently working on {year}...')
sys.stdout.flush()    
if namelist.gnu_parallel == True: fn_th = calc_thermo.get_fn_thermo(year)
else: fn_th = calc_thermo.get_fn_thermo()
        
ds = xr.open_dataset(fn_th)
dt_year_start = datetime.datetime(year-1, 12, 31)
dt_year_end = datetime.datetime(year, 12, 31)
dt_bounds = input.convert_from_datetime(ds, [dt_year_start, dt_year_end])
ds = ds.sel(time=slice(dt_bounds[0], dt_bounds[1])).load()
lon = ds['lon'].data
lat = ds['lat'].data
mld = ocean.mld_climatology(year, basins.TC_Basin('GL'))
strat = ocean.strat_climatology(year, basins.TC_Basin('GL'))    # Make sure latitude is increasing.
vpot = ds['vmax'] * namelist.PI_reduc * np.sqrt(namelist.Ck / namelist.Cd)
rh_mid = ds['rh_mid']
chi = ds['chi']

if (lat[0] - lat[1]) > 0:
    vpot = vpot.reindex({'lat': lat[::-1]})
    rh_mid = rh_mid.reindex({'lat': lat[::-1]})
    chi = chi.reindex({'lat': lat[::-1]})
    lat = lat[::-1]

# Load the basin bounds and genesis points.
basin_ids = np.array(sorted([k for k in namelist.basin_bounds if k != 'GL']))
f_basins = {}
for basin_id in basin_ids:
    ds_b = xr.open_dataset('land/%s.nc' % basin_id)
    basin_mask = ds_b['basin']
    f_basins[basin_id] = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)

# In case basin is "GL", we load again.
ds_b = xr.open_dataset('land/%s.nc' % b.basin_id)
basin_mask = ds_b['basin']
f_b = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)
b_bounds = b.get_bounds()

# To randomly seed in both space and time, load data for each month in the year.
To_s = namelist.output_interval_s
T_s = namelist.total_track_time_days * 24 * 60 * 60     # total time to run tracks
if (namelist.gnu_parallel) and (namelist.wind_ts == 'monthly'):
    fn_wnd_stat = [env_wind.get_env_wnd_fn(year), env_wind.get_env_wnd_fn(year + 1)]
    ds_wnd = xr.open_mfdataset(fn_wnd_stat)
else:
    fn_wnd_stat = env_wind.get_env_wnd_fn(year)
    ds_wnd = xr.open_dataset(fn_wnd_stat)

cpl_fast = [0] * 1460
m_init_fx = [0] * 1460
n_seeds = np.zeros((len(basin_ids), 1460))

# Create array of dates to draw from and remove leap day
start_date = datetime.datetime(year, 1, 1, 0)
end_date = datetime.datetime(year, 12, 31, 18)
interval = datetime.timedelta(hours = 6)  # 6-hourly time interval

dates = [start_date + i * interval for i in range(int((end_date - start_date) / interval) + 1)
         if (((start_date + i * interval).month != 2) or (((start_date + i * interval).day) != 29))]

if namelist.debug: print(f'Initializing CoupledFAST objects...')
sys.stdout.flush()
idxs = range(0, 1460)

for i in idxs:
    dt_6hr = dates[i]
    ds_dt_6hr = input.convert_from_datetime(ds_wnd, [dt_6hr])[0]
    print(i, ds_dt_6hr)
    sys.stdout.flush()
    rh_mid_6hr = rh_mid.interp(time = ds_dt_6hr).data
    m_init_fx[i] = mat.interp2_fx(lon, lat, rh_mid_6hr)

    os.makedirs(f'{namelist.output_directory}/pickle_files', exist_ok=True)
    pickle_fn = os.path.join(namelist.output_directory, f'pickle_files/cpl_fast_{year}_{i}.pkl')
    if not os.path.exists(pickle_fn):
        month_index = int(dt_6hr.month - 1)
        sys.stdout.flush()
        vpot_6hr = np.nan_to_num(vpot.interp(time = ds_dt_6hr).data, 0)
        rh_mid_6hr = rh_mid.interp(time = ds_dt_6hr).data
        chi_6hr = chi.interp(time = ds_dt_6hr).data
        chi_6hr[np.isnan(chi_6hr)] = 5
        chi_6hr = np.maximum(np.minimum(np.exp(np.log(chi_6hr + 1e-3) + namelist.log_chi_fac) + namelist.chi_fac, 5), 1e-5)
    
        mld_month = mat.interp_2d_grid(mld['lon'], mld['lat'], np.nan_to_num(mld[:, :, month_index]), lon, lat)
        strat_month = mat.interp_2d_grid(strat['lon'], strat['lat'], np.nan_to_num(strat[:, :, month_index]), lon, lat)
        #cpl_fast[i] = coupled_fast.Coupled_FAST(fn_wnd_stat, b, ds_dt_6hr, To_s, T_s)
        #cpl_fast[i].init_fields(lon, lat, chi_6hr, vpot_6hr, mld_month, strat_month)
        # Create Coupled_FAST object
        cpl_fast = coupled_fast.Coupled_FAST(fn_wnd_stat, b, ds_dt_6hr, namelist.output_interval_s, T_s)
        cpl_fast.init_fields(lon, lat, chi_6hr, vpot_6hr, mld_month, strat_month)
        
        # Serialize and save object to a file
        with open(pickle_fn, 'wb') as f:
            pickle.dump(cpl_fast, f)
