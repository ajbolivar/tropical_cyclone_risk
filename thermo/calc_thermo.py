#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""

import dask
import datetime
import os
import glob
import namelist
import numpy as np
import xarray as xr
import pandas as pd

from util import input, mat
from thermo import thermo

def get_fn_thermo(year = 999):
    if namelist.gnu_parallel == True:
        fn_th = '%s/thermo_%s_%d01_%d12.nc' % (namelist.output_directory, namelist.exp_prefix,
                                               year, year)
    else:
        fn_th = '%s/thermo_%s_%d%02d_%d%02d.nc' % (namelist.output_directory, namelist.exp_prefix,
                                                   namelist.start_year, namelist.start_month,
                                                   namelist.end_year, namelist.end_month)
    return(fn_th)

def running_mean_xr(data, var, dim, n):
    var_sm = data[var].rolling({dim: n}, min_periods=1, center=True).mean()
    data[var] = var_sm

    return data

def compute_thermo(dt_start, dt_end):
    ds_sst = input.load_sst(dt_start, dt_end).load()
    ds_psl = input.load_mslp(dt_start, dt_end).load()
    ds_ta = input.load_temp(dt_start, dt_end).load()
    ds_hus = input.load_sp_hum(dt_start, dt_end).load()
    lon_ky = input.get_lon_key()
    lat_ky = input.get_lat_key()
    sst_ky = input.get_sst_key()
    
    # Check to make sure all necessary timesteps of t, sst, psl, and q are available for
    # computation of thermodynamic quantities
    for ds in [ds_sst, ds_psl, ds_ta, ds_hus]:
        nTime = len(ds['time'])

        if namelist.thermo_ts == 'sub_monthly':
            if nTime < 365:
                print(f'Error! Data should be length 365 but is instead length {nTime}. Please check your input data to ensure you have all necessary timesteps of thermodynamic data available for calculations.')
                return
        if namelist.thermo_ts == 'monthly':
            if nTime < 12:
                print(f'Error! Data should be length 12 but is instead length {nTime}. Please check your input data to ensure you have all necessary timesteps of thermodynamic data available for calculations.')
                return

    # Compute running means for all input variables
    if namelist.thermo_ts == 'sub-monthly':
        print('Computing running means...')
        ds_sst = running_mean_xr(ds_sst, input.get_sst_key(), 'time', namelist.window)
        ds_psl = running_mean_xr(ds_psl, input.get_mslp_key(), 'time', namelist.window)
        ds_ta  = running_mean_xr(ds_ta, input.get_temp_key(), 'time', namelist.window)
        ds_hus = running_mean_xr(ds_hus, input.get_sp_hum_key(), 'time', 31)

    vmax = np.zeros(ds_psl[input.get_mslp_key()].shape)
    chi = np.zeros(ds_psl[input.get_mslp_key()].shape)
    rh_mid = np.zeros(ds_psl[input.get_mslp_key()].shape)
    for i in range(nTime):
        # Convert all variables to the atmospheric grid.
        sst_interp = mat.interp_2d_grid(ds_sst[lon_ky], ds_sst[lat_ky],
                                        np.nan_to_num(ds_sst[sst_ky][i, :, :].data),
                                        ds_ta[lon_ky], ds_ta[lat_ky])
        if 'C' in ds_sst[sst_ky].units:
            sst_interp = sst_interp + 273.15

        psl = ds_psl[input.get_mslp_key()][i, :, :]
        ta = ds_ta[input.get_temp_key()][i, :, :, :]
        hus = ds_hus[input.get_sp_hum_key()][i, :, :, :]
        lvl = ds_ta[input.get_lvl_key()]
        lvl_d = np.copy(ds_ta[input.get_lvl_key()].data)

        # Ensure lowest model level is first.
        # Here we assume the model levels are in pressure.
        if (lvl[0] - lvl[1]) < 0:
            ta = ta.reindex({input.get_lvl_key(): lvl[::-1]})
            hus = hus.reindex({input.get_lvl_key(): lvl[::-1]})
            lvl_d = lvl_d[::-1]
    
        p_midlevel = namelist.p_midlevel                    # Pa
        if lvl.units in ['millibars', 'hPa']:
            lvl_d *= 100                                    # needs to be in Pa
            p_midlevel = namelist.p_midlevel / 100          # hPa
        
        lvl_mid = lvl.sel({input.get_lvl_key(): p_midlevel}, method = 'nearest')

        # TODO: Check units of psl, ta, and hus
        vmax_args = (sst_interp, psl.data, lvl_d, ta.data, hus.data)
        vmax[i, :, :] = thermo.CAPE_PI_vectorized(*vmax_args)
        ta_midlevel = ta.sel({input.get_lvl_key(): p_midlevel}, method = 'nearest').data
        hus_midlevel = hus.sel({input.get_lvl_key(): p_midlevel}, method = 'nearest').data

        p_midlevel_Pa = float(lvl_mid) * 100 if lvl_mid.units in ['millibars', 'hPa'] else float(lvl_mid)
        chi_args = (sst_interp, psl.data, ta_midlevel,
                    p_midlevel_Pa, hus_midlevel)
        chi[i, :, :] = np.minimum(np.maximum(thermo.sat_deficit(*chi_args), 0), 10)
        rh_mid[i, :, :] = thermo.conv_q_to_rh(ta_midlevel, hus_midlevel, p_midlevel_Pa)
    
    return (vmax, chi, rh_mid)

def gen_thermo(year = 999):
    # Get thermo file name
    if namelist.gnu_parallel == True: fn_out = get_fn_thermo(year)
    else: fn_out = get_fn_thermo()
    # TODO: Assert all of the datasets have the same length in time.
    # Check if file exists
    if os.path.exists(fn_out):
        print(f"File {fn_out} exists. Skipping...")
        return

    # Load datasets metadata. Since SST is split into multiple files and can
    # cause parallel reads with open_mfdataset to hang, save as a single file.
    if namelist.gnu_parallel == True: 
        dt_start = datetime.datetime(year, 1, 1, 0)
        dt_end = datetime.datetime(year, 12, 31, 18)
    else: dt_start, dt_end = input.get_bounding_times()

    start_year = dt_start.year
    end_year = dt_end.year
    start_month = dt_start.month
    end_month = dt_end.month
    ds = input.load_mslp()
    ct_bounds = [dt_start, dt_end]
    ds_times = input.convert_from_datetime(ds,
                   np.array([x for x in input.convert_to_datetime(ds, ds['time'].values)
                             if x >= ct_bounds[0] and x <= ct_bounds[1]]))
    # If specific thermo file does not exist but data is contained in larger thermo file,
    # save subset of data from larger file.
    if glob.glob('%s/thermo*.nc' % namelist.output_directory):
        thermo_files = glob.glob('%s/thermo*.nc' % namelist.output_directory)
        for file in thermo_files:
            part = file.split('/')[-1].split('_') # Isolate file name from path
            syr = int(part[-2][0:4]) # Isolate start year from file name
            eyr = int(part[-1].split('.')[0][0:4]) # Isolate end year from file name
            # Check if year is in this range, save subset from existing thermo file
            if year in range(syr,eyr):
                thermo = xr.open_dataset(file)
                thermo_ss = thermo.sel(time=slice(f'{start_year}-{start_month:02}',
                                                  f'{end_year}-{end_month:02}'))
                thermo_ss.to_netcdf(fn_out)
            
    n_chunks = namelist.n_procs
    chunks = np.array_split(ds_times, np.minimum(n_chunks, np.floor(len(ds_times) / 2)))
    lazy_results = []
    for i in range(len(chunks)):
        lazy_result = dask.delayed(compute_thermo)(chunks[i][0], chunks[i][-1])
        lazy_results.append(lazy_result)
    out = dask.compute(*lazy_results, scheduler = 'processes', num_workers = n_chunks)

    # Clean up and process output.
    if namelist.thermo_ts == 'sub-monthly':
        ds_times = input.convert_from_datetime(ds,
                    np.array([datetime.datetime(x.year, x.month, x.day) for x in
                             [x for x in input.convert_to_datetime(ds, ds['time'].values)
                             if x >= ct_bounds[0] and x <= ct_bounds[1]]]))
    elif namelist.thermo_ts == 'monthly':
        # Ensure monthly timestamps have middle-of-the-month days.
        ds_times = input.convert_from_datetime(ds,
                    np.array([datetime.datetime(x.year, x.month, 15) for x in
                             [x for x in input.convert_to_datetime(ds, ds['time'].values)
                             if x >= ct_bounds[0] and x <= ct_bounds[1]]]))
    
    vmax = np.concatenate([x[0] for x in out], axis = 0)
    chi = np.concatenate([x[1] for x in out], axis = 0)
    rh_mid = np.concatenate([x[2] for x in out], axis = 0)
    
    ds_thermo = xr.Dataset(data_vars = dict(vmax = (['time', 'lat', 'lon'], vmax),
                                            chi = (['time', 'lat', 'lon'], chi),
                                            rh_mid = (['time', 'lat', 'lon'], rh_mid)),
                           coords = dict(lon = ("lon", ds[input.get_lon_key()].data),
                                         lat = ("lat", ds[input.get_lat_key()].data),
                                         time = ("time", ds_times)))
    ds_thermo.to_netcdf(fn_out)
    print('Saved %s' % fn_out)
